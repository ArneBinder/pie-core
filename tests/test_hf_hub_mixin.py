import logging
from pathlib import Path
from typing import Any, Dict, Optional, Type, Union

import pytest
from huggingface_hub.hf_api import HfApi

from pie_core.hf_hub_mixin import HFHubMixin
from tests import FIXTURES_ROOT

logger = logging.getLogger(__name__)

PRETRAINED_PATH = FIXTURES_ROOT / "pretrained" / "hf_hub_mixin"
HF_USERNAME = "rainbowrivey"
HF_PATH = f"{HF_USERNAME}/HF_Hub_Test"
HF_WRITE_PATH = f"{HF_USERNAME}/HF_Hub_Write_Test"
WRONG_HF_PATH = f"{HF_USERNAME}/HF_Hub_Test_Wrong"
HF_NO_ACCESS_MSG = (
    "Not enough permissions to HuggingFace repository. "
    f"Provide a token with write access to '{HF_WRITE_PATH}'"
)

hf_api = HfApi()
hf_has_write_access = hf_api.repo_exists(HF_WRITE_PATH)


class HFHubObject(HFHubMixin):
    config_name = "hf_hub_config.json"
    config_type_key = "hf_hub_type"

    def __init__(self, *args, foo: Optional[str] = None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.foo = foo

    def _config(self):
        return {"foo": self.foo}

    def _save_pretrained(self, save_directory: Path) -> None:
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


@pytest.mark.test_hf_access
def test_hf_access():
    assert hf_has_write_access


def test_is_from_pretrained(hf_hub_object):
    assert not hf_hub_object.is_from_pretrained


def test_has_config(hf_hub_object):
    assert hf_hub_object.has_config


def test_config(hf_hub_object, config_as_dict):
    assert hf_hub_object.config == config_as_dict


def test_config_not_implemented():
    class Test(HFHubMixin):
        pass

    assert Test().config == {}


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


def test_save_pretrained_not_implemented(tmp_path):
    class Test(HFHubMixin):
        pass

    with pytest.raises(NotImplementedError):
        Test().save_pretrained(save_directory=tmp_path)


@pytest.mark.slow
@pytest.mark.skipif(not hf_has_write_access, reason=HF_NO_ACCESS_MSG)
def test_save_pretrained_push_to_hub(hf_hub_object, caplog, tmp_path, config_as_json):
    try:
        with caplog.at_level(logging.INFO):
            hf_hub_object.save_pretrained(
                save_directory=tmp_path, push_to_hub=True, repo_id=HF_WRITE_PATH
            )

        assert (
            f"_save_pretrained() called with arguments str(save_directory)='{tmp_path}'"
            in caplog.text
        )

        assert tmp_path.joinpath(hf_hub_object.config_name).exists()
        with open(tmp_path.joinpath(hf_hub_object.config_name)) as f:
            file_contents = f.read()
        assert file_contents == config_as_json

    finally:
        hf_api.delete_file(hf_hub_object.config_name, HF_WRITE_PATH)


@pytest.mark.slow
@pytest.mark.skipif(not hf_has_write_access, reason=HF_NO_ACCESS_MSG)
def test_save_pretrained_push_to_hub_no_repo_id(hf_hub_object, caplog, tmp_path, config_as_json):
    #   If no repo_id was provided, save_directory will be used instead.
    # We need to name folder exactly as the Repository Name.
    #   Token used for this tests should be issued by Repo owner, not anyone else with access,
    # since repo will be looked up/created under current User's name.
    #   In this case path also contains Username, which is not necessary, it will be ignored;
    # But this is a simple solution to make sure folder has the right name if we change repos.
    path = tmp_path.joinpath(HF_WRITE_PATH)
    path.mkdir(parents=True, exist_ok=True)

    try:
        with caplog.at_level(logging.INFO):
            hf_hub_object.save_pretrained(save_directory=path, push_to_hub=True)

        assert (
            f"_save_pretrained() called with arguments str(save_directory)='{path}'" in caplog.text
        )

        assert path.joinpath(hf_hub_object.config_name).exists()
        with open(path.joinpath(hf_hub_object.config_name)) as f:
            file_contents = f.read()
        assert file_contents == config_as_json

    finally:
        hf_api.delete_file(hf_hub_object.config_name, HF_WRITE_PATH)


def test_retrieve_config_file_local():
    path_to_config, kwargs = HFHubObject.retrieve_config_file(PRETRAINED_PATH)
    assert path_to_config is not None
    assert path_to_config == str(PRETRAINED_PATH / "hf_hub_config.json")


def test_retrieve_config_file_local_wrong_path(caplog, tmp_path):
    with caplog.at_level(logging.WARNING):
        HFHubObject.retrieve_config_file(tmp_path)
    assert caplog.messages == [
        f"{HFHubObject.config_name} not found in {Path(tmp_path).resolve()}"
    ]


def test_retrieve_config_file_hf():
    path_to_config, kwargs = HFHubObject.retrieve_config_file(HF_PATH)
    assert path_to_config is not None
    assert Path(path_to_config).is_file()


def test_retrieve_config_file_hf_wrong_path(caplog):
    with caplog.at_level(logging.WARNING):
        HFHubObject.retrieve_config_file(WRONG_HF_PATH)
    assert caplog.messages == [f"{HFHubObject.config_name} not found in HuggingFace Hub."]


@pytest.mark.parametrize("config_path", [PRETRAINED_PATH, HF_PATH])
def test_from_pretrained(config_as_dict, config_path):
    pretrained = HFHubObject.from_pretrained(config_path)
    assert pretrained.is_from_pretrained
    assert pretrained.config == config_as_dict


@pytest.mark.parametrize("config_path", [PRETRAINED_PATH, HF_PATH])
def test_from_pretrained_not_implemented(config_path):
    class Test(HFHubMixin):
        pass

    with pytest.raises(NotImplementedError):
        Test.from_pretrained(config_path)


@pytest.mark.parametrize("config_path", [PRETRAINED_PATH, HF_PATH])
def test_from_pretrained_with_kwargs_override(config_as_dict, config_path):
    pretrained = HFHubObject.from_pretrained(
        config_path, foo="test", hf_hub_type="will_be_discarded"
    )
    assert pretrained.is_from_pretrained
    config = config_as_dict.copy()
    config.update(foo="test")
    assert pretrained.config == config


@pytest.mark.slow
@pytest.mark.skipif(not hf_has_write_access, reason=HF_NO_ACCESS_MSG)
def test_push_to_hub(hf_hub_object, config_as_dict):
    try:
        hf_hub_object.push_to_hub(HF_WRITE_PATH)
        pretrained = HFHubObject.from_pretrained(HF_WRITE_PATH)
        assert pretrained.is_from_pretrained
        assert pretrained.config == config_as_dict

    finally:
        hf_api.delete_file(hf_hub_object.config_name, HF_WRITE_PATH)


def test_from_config(hf_hub_object):
    new_hf_hub_object = HFHubObject.from_config(config=hf_hub_object.config)
    assert new_hf_hub_object.config == hf_hub_object.config


def test_from_config_with_kwargs_override(hf_hub_object):
    new_hf_hub_object = HFHubObject.from_config(
        config=hf_hub_object.config, foo="test", hf_hub_type="will_be_discarded"
    )
    config = hf_hub_object.config.copy()
    config.update(foo="test")
    assert new_hf_hub_object.config == config
