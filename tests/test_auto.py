from pathlib import Path
from typing import Any, Dict, Optional, Type, Union

import pytest
from huggingface_hub.constants import DEFAULT_ETAG_TIMEOUT

from pie_core import Auto, Registrable
from pie_core.hf_hub_mixin import HFHubMixin
from tests import FIXTURES_ROOT

PRETRAINED_PATH = FIXTURES_ROOT / "pretrained" / "auto"
PRETRAINED_PATH_NO_AUTO_TYPE = FIXTURES_ROOT / "pretrained" / "auto_no_auto_type"


class TestHFHubMixin(HFHubMixin):
    config_name: str = "auto_config.json"
    config_type_key: str = "auto_type"

    @classmethod
    def _from_pretrained(
        cls: Type[HFHubMixin],
        *,
        model_id: str,
        subfolder: Optional[str] = None,
        repo_type: Optional[str] = None,
        revision: Optional[str] = None,
        library_name: Optional[str] = None,
        library_version: Optional[str] = None,
        cache_dir: Union[str, Path, None] = None,
        local_dir: Union[str, Path, None] = None,
        user_agent: Union[Dict, str, None] = None,
        force_download: bool = False,
        proxies: Optional[Dict] = None,
        etag_timeout: float = DEFAULT_ETAG_TIMEOUT,
        token: Union[bool, str, None] = None,
        local_files_only: bool = False,
        headers: Optional[Dict[str, str]] = None,
        endpoint: Optional[str] = None,
        config: Optional[dict] = None,
        **kwargs,
    ) -> HFHubMixin:
        return cls.from_config(config=config or {}, **kwargs)


class Test(Registrable, TestHFHubMixin):
    foo: str = None

    def __init__(self, *args, foo: Optional[str] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.foo = foo


@Test.register()
class Sub(Test):

    pass


@Test.register()
class Sub2(Test):

    pass


class AutoTest(TestHFHubMixin, Auto[Test]):
    BASE_CLASS = Test


@pytest.fixture(scope="module")
def config_as_dict() -> Dict[str, Any]:
    return {"auto_type": "Sub", "foo": "Test"}


def test_from_config(config_as_dict):
    sub = AutoTest.from_config(config=config_as_dict)
    assert isinstance(sub, Sub)
    assert sub.foo == "Test"


def test_from_config_with_kwargs_override(config_as_dict):
    sub = AutoTest.from_config(config=config_as_dict, auto_type="Sub2", foo="Test2")
    assert isinstance(sub, Sub2)
    assert sub.foo == "Test2"


def test_from_config_without_auto_type(config_as_dict):
    config = config_as_dict.copy()
    config.pop("auto_type")
    with pytest.raises(ValueError) as e:
        AutoTest.from_config(config=config)
    assert (
        str(e.value) == "Missing required key 'auto_type' to select a concrete Test for AutoTest. "
        "Provide it in the config or pass it as a keyword argument (auto_type=...). "
        "Received config: {'foo': 'Test'}"
    )


def test_from_config_with_external_auto_type(config_as_dict):
    config = config_as_dict.copy()
    config.pop("auto_type")
    sub = AutoTest.from_config(config=config, auto_type="Sub")
    assert isinstance(sub, Sub)
    assert sub.foo == "Test"


def test_from_pretrained():
    sub = AutoTest.from_pretrained(PRETRAINED_PATH)
    assert isinstance(sub, Sub)
    assert sub.is_from_pretrained
    assert sub.foo == "Test"


def test_from_pretrained_with_kwargs_override():
    sub = AutoTest.from_pretrained(PRETRAINED_PATH, auto_type="Sub2", foo="Test2")
    assert isinstance(sub, Sub2)
    assert sub.is_from_pretrained
    assert sub.foo == "Test2"


def test_from_pretrained_without_auto_type():
    with pytest.raises(ValueError) as e:
        AutoTest.from_pretrained(PRETRAINED_PATH_NO_AUTO_TYPE)
    assert (
        str(e.value) == "Missing required key 'auto_type' to select a concrete Test for AutoTest. "
        "Provide it in the config or pass it as a keyword argument (auto_type=...). "
        "Received config: {'foo': 'Test'}"
    )


def test_from_pretrained_with_external_auto_type():
    sub = AutoTest.from_pretrained(PRETRAINED_PATH_NO_AUTO_TYPE, auto_type="Sub")
    assert isinstance(sub, Sub)
    assert sub.is_from_pretrained
    assert sub.foo == "Test"
