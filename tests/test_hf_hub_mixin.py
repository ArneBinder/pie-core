import logging
from pathlib import Path
from typing import Any, Dict, Optional, Type, Union

import pytest
from huggingface_hub.hf_api import HfApi

from pie_core.hf_hub_mixin import HFHubMixin

logger = logging.getLogger(__name__)

CONFIG_PATH = Path(__file__).parent / "fixtures" / "configs"
WRONG_CONFIG_PATH = Path(__file__).parent / "fixtures"
HF_PATH = "rainbowrivey/HF_Hub_Test"
HF_WRITE_PATH = "rainbowrivey/HF_Hub_Write_Test"
WRONG_HF_PATH = "rainbowrivey/HF_Hub_Test_Wrong"


def cleanup_hf_temp_repo():
    api = HfApi()
    api.delete_repo(HF_WRITE_PATH, missing_ok=True)

class HFHubObject(HFHubMixin):
    config_name = "hf_hub_config.json"
    config_type_key = "hf_hub_config_type"

    def __init__(self, *args, foo: Optional[str] = None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.foo = foo

    def _config(self):
        return {"foo": self.foo}

    def _save_pretrained(self, save_directory) -> None:
        # We cast save_directory to string, else it would include class type
        # which is platform dependent.
        logger.info(f"_save_pretrained() called with arguments {str(save_directory)=}")
        return None

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


@pytest.fixture(scope="module")
def hf_hub_object() -> HFHubObject:
    return HFHubObject(foo="bar")


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


@pytest.mark.slow
def test_save_pretrained_push_to_hub(hf_hub_object, caplog, tmp_path, config_as_json):
    try:
        with caplog.at_level(logging.INFO):
            hf_hub_object.save_pretrained(save_directory=tmp_path, push_to_hub=True, repo_id=HF_WRITE_PATH)

        assert (
                f"_save_pretrained() called with arguments str(save_directory)='{tmp_path}'" in caplog.text
        )

        assert tmp_path.joinpath(hf_hub_object.config_name).exists()
        with open(tmp_path.joinpath(hf_hub_object.config_name)) as f:
            file_contents = f.read()
        assert file_contents == config_as_json



    finally:
        cleanup_hf_temp_repo()


def test_retrieve_config_file_local():
    config, kwargs = HFHubObject.retrieve_config_file(CONFIG_PATH)
    assert config is not None
    assert config == str(CONFIG_PATH / "hf_hub_config.json")


def test_retrieve_config_file_local_wrong_path(caplog):
    with caplog.at_level(logging.WARNING):
        config, kwargs = HFHubObject.retrieve_config_file(WRONG_CONFIG_PATH)
    assert caplog.messages == [f"{HFHubObject.config_name} not found in {Path(WRONG_CONFIG_PATH).resolve()}"]


@pytest.mark.slow
def test_retrieve_config_file_hf():
    config, kwargs = HFHubObject.retrieve_config_file(HF_PATH)
    assert config is not None
    assert Path(config).is_file()


@pytest.mark.slow
def test_retrieve_config_file_hf_wrong_path(caplog):
    with caplog.at_level(logging.WARNING):
        config, kwargs = HFHubObject.retrieve_config_file(WRONG_HF_PATH)
    assert caplog.messages == [f"{HFHubObject.config_name} not found in HuggingFace Hub."]


def test_from_pretrained_local_config_file():
    pretrained = HFHubObject.from_pretrained(CONFIG_PATH)
    assert pretrained.is_from_pretrained
    assert pretrained.foo == "bar"


@pytest.mark.slow
def test_from_pretrained_hf(config_as_dict):
    pretrained = HFHubObject.from_pretrained(HF_PATH)
    assert pretrained.is_from_pretrained
    assert pretrained.foo == "bar"


@pytest.mark.slow
def test_push_to_hub(hf_hub_object):
    try:
        hf_hub_object.push_to_hub(HF_WRITE_PATH)
        pretrained = HFHubObject.from_pretrained(HF_WRITE_PATH)
        assert pretrained.is_from_pretrained
        assert pretrained.foo == "bar"

    finally:
        cleanup_hf_temp_repo()

def test_from_config(hf_hub_object, config_as_dict):
    new_hf_hub_object = HFHubObject.from_config(config=hf_hub_object.config)
    assert new_hf_hub_object.config == hf_hub_object.config
