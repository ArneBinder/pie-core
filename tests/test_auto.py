from pathlib import Path
from typing import Any, Dict, Optional, Type, Union

import pytest

from pie_core import Auto, Registrable
from pie_core.hf_hub_mixin import HFHubMixin, T
from tests import FIXTURES_ROOT

CONFIG_PATH = FIXTURES_ROOT / "configs"


class TestHFHubMixin(HFHubMixin):
    config_name: str = "test_config.json"
    config_type_key: str = "test_type"

    @classmethod
    def _from_pretrained(
        cls: Type[HFHubMixin],
        *,
        model_id: str,
        revision: Optional[str],
        cache_dir: Optional[Union[str, Path]],
        force_download: bool,
        proxies: Optional[Dict],
        resume_download: bool,
        local_files_only: bool,
        token: Optional[Union[str, bool]],
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
    return {"test_type": "Sub", "foo": "Test"}


def test_from_config(config_as_dict):
    sub = AutoTest.from_config(config=config_as_dict)
    assert isinstance(sub, Sub)
    assert sub.foo == "Test"


def test_from_config_with_kwargs_override(config_as_dict):
    sub = AutoTest.from_config(config=config_as_dict, test_type="Sub2", foo="Test2")
    assert isinstance(sub, Sub2)
    assert sub.foo == "Test2"


def test_from_pretrained():
    sub = AutoTest.from_pretrained(CONFIG_PATH)
    assert isinstance(sub, Sub)
    assert sub.is_from_pretrained
    assert sub.foo == "Test"


def test_from_pretrained_with_kwargs_override():
    sub = AutoTest.from_pretrained(CONFIG_PATH, test_type="Sub2", foo="Test2")
    assert isinstance(sub, Sub2)
    assert sub.is_from_pretrained
    assert sub.foo == "Test2"
