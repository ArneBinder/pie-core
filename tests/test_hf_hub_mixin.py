import logging
from pathlib import Path
from typing import Any, Dict, Optional, Type, TypeVar, Union

import pytest

from pie_core.hf_hub_mixin import PieBaseHFHubMixin
from pie_core.utils.dictionary import TNestedBoolDict

logger = logging.getLogger(__name__)

THFHubMixin = TypeVar("THFHubMixin", bound="HFHubObject")
CONFIG_PATH = Path(__file__).parent / "fixtures" / "configs"


class HFHubObject(PieBaseHFHubMixin):
    config_name = "hf_hub_config.json"
    config_type_key = "hf_hub_config_type"

    def __init__(self, *args, foo: Optional[str] = None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.foo = foo

    def _config(self):
        return {"foo": "bar"}

    def _save_pretrained(self, save_directory) -> None:
        # We cast save_directory to string, else it would include class type
        # which is platform dependent.
        logger.info(f"_save_pretrained() called with arguments {str(save_directory)=}")
        return None

    @classmethod
    def _from_pretrained(
        cls: Type[THFHubMixin],
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
    ) -> THFHubMixin:
        return cls.from_config(config=config or {}, **kwargs)


@pytest.fixture(scope="module")
def hf_hub_object() -> HFHubObject:
    return HFHubObject()


@pytest.fixture(scope="module")
def config_as_dict() -> Dict[str, Any]:
    return {"foo": "bar"}


@pytest.fixture(scope="module")
def config_as_json() -> str:
    return '{\n  "foo": "bar"\n}'


def test_is_from_pretrained(hf_hub_object):
    assert not hf_hub_object.is_from_pretrained


def test_has_config(hf_hub_object):
    assert hf_hub_object.has_config


def test_config(hf_hub_object, config_as_dict):
    assert hf_hub_object.config == config_as_dict


def test_save_pretrained(hf_hub_object, caplog, tmp_path, config_as_json):
    with caplog.at_level(logging.INFO):
        hf_hub_object.save_pretrained(save_directory=tmp_path)
    assert (
        f"_save_pretrained() called with arguments str(save_directory)='{tmp_path}'" in caplog.text
    )

    assert tmp_path.joinpath(hf_hub_object.config_name).exists()
    with open(tmp_path.joinpath(hf_hub_object.config_name)) as f:
        file_contents = f.read()
    assert file_contents == config_as_json


def test_retrieve_config_file(hf_hub_object):
    config, kwargs = hf_hub_object.retrieve_config_file(CONFIG_PATH)
    assert config is not None
    assert config == str(CONFIG_PATH / "hf_hub_config.json")
    # TODO: Test loading from hub?


def test_from_pretrained():
    pretrained = HFHubObject.from_pretrained(CONFIG_PATH)
    assert pretrained.is_from_pretrained
    assert pretrained.foo == "bar"


def test_push_to_hub(hf_hub_object):
    pass


def test_from_config(hf_hub_object):
    pass
