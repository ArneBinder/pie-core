import json
import logging
import os
from typing import List

import pytest
from huggingface_hub import HfApi

from pie_core import AutoModel, Model
from tests import FIXTURES_ROOT

logger = logging.getLogger(__name__)

PRETRAINED_PATH = FIXTURES_ROOT / "pretrained" / "model"
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


@Model.register()
class TestModel(Model):
    weights_file_name = "model.json"
    param: List[int]

    def __init__(self, param=None, **kwargs):
        super().__init__(**kwargs)
        if param is None:
            param = [0, 0, 0]
        self.param = param

    def save_model_file(self, model_file: str) -> None:
        state_dict = {"param": self.param}
        json.dump(state_dict, open(model_file, "w"), indent=2)

    def load_model_file(
        self, model_file: str, map_location: str = "cpu", strict: bool = False
    ) -> None:
        state_dict = json.load(open(model_file))
        self.param = state_dict["param"]


@pytest.fixture
def model():
    return TestModel()


@pytest.fixture
def config_as_json() -> str:
    return '{\n  "model_type": "TestModel"\n}'


@pytest.fixture
def weights_as_json() -> str:
    return '{\n  "param": [\n    0,\n    0,\n    0\n  ]\n}'


@pytest.mark.test_hf_access
def test_hf_access():
    assert hf_has_write_access


def test_save_pretrained(model, tmp_path, config_as_json, weights_as_json) -> None:
    model.save_pretrained(tmp_path)
    assert os.path.exists(tmp_path / model.config_name)
    with open(tmp_path / model.config_name) as f:
        assert f.read() == config_as_json
    assert os.path.exists(tmp_path / model.weights_file_name)
    with open(tmp_path / model.weights_file_name) as f:
        assert f.read() == weights_as_json


@pytest.mark.skipif(not hf_has_write_access, reason=HF_NO_ACCESS_MSG)
def test_save_pretrained_push_to_hub(model, caplog, tmp_path, config_as_json, weights_as_json):
    try:
        model.save_pretrained(save_directory=tmp_path, push_to_hub=True, repo_id=HF_WRITE_PATH)

        assert tmp_path.joinpath(model.config_name).exists()
        with open(tmp_path.joinpath(model.config_name)) as f:
            file_contents = f.read()
        assert file_contents == config_as_json

        assert tmp_path.joinpath(model.weights_file_name).exists()
        with open(tmp_path.joinpath(model.weights_file_name)) as f:
            file_contents = f.read()
        assert file_contents == weights_as_json

    finally:
        hf_api.delete_file(model.config_name, HF_WRITE_PATH)
        hf_api.delete_file(model.weights_file_name, HF_WRITE_PATH)


@pytest.mark.skipif(not hf_has_write_access, reason=HF_NO_ACCESS_MSG)
def test_push_to_hub(model):
    try:
        model.push_to_hub(HF_WRITE_PATH)
        pretrained = TestModel.from_pretrained(HF_WRITE_PATH)
        assert pretrained.is_from_pretrained
        assert pretrained.config == model.config
        assert pretrained.param == model.param

    finally:
        hf_api.delete_file(model.config_name, HF_WRITE_PATH)
        hf_api.delete_file(model.weights_file_name, HF_WRITE_PATH)


@pytest.mark.parametrize("config_path", [PRETRAINED_PATH, HF_PATH])
def test_from_pretrained(model, config_path):
    pretrained = TestModel.from_pretrained(config_path)
    assert isinstance(pretrained, TestModel)
    assert pretrained.is_from_pretrained
    assert pretrained.config == model.config
    assert pretrained.param == model.param


def test_save_and_from_pretrained(model, tmp_path) -> None:
    model.save_pretrained(tmp_path)
    pretrained = TestModel.from_pretrained(tmp_path)
    assert isinstance(pretrained, TestModel)
    assert pretrained.config == model.config
    assert pretrained.param == model.param


def test_auto_model_from_pretrained(model) -> None:
    pretrained = AutoModel.from_pretrained(PRETRAINED_PATH)
    assert type(pretrained) is TestModel
    assert pretrained.config == model.config
    assert pretrained.param == model.param


def test_from_pretrained_warnings(model, caplog) -> None:
    with caplog.at_level(logging.WARNING):
        TestModel.from_pretrained(PRETRAINED_PATH, map_location="cpu", strict=False)
    assert caplog.messages == [
        'map_location is deprecated. Use load_model_file={"map_location": "cpu"} instead.',
        'strict is deprecated. Use load_model_file={"strict": False} instead.',
    ]


def test_config_warning(caplog) -> None:
    class UnregisteredTestModel(Model):
        pass

    unregistered_model = UnregisteredTestModel()
    assert unregistered_model.BASE_CLASS is Model
    with caplog.at_level(logging.WARNING):
        unregistered_model._config()
    assert caplog.messages == [
        "UnregisteredTestModel is not registered. It will not work"
        " with AutoModel.from_pretrained() or"
        " AutoModel.from_config(). Consider to annotate the class with"
        " @Model.register() or @Model.register(name='...') to register it at as a Model"
        " which will allow to load it via AutoModel."
    ]


def test_config_warning_no_base_class(caplog) -> None:
    class UnregisteredTestModelWithNoBaseClass(Model):
        BASE_CLASS = None

    unregistered_model = UnregisteredTestModelWithNoBaseClass()
    assert unregistered_model.BASE_CLASS is None
    with caplog.at_level(logging.WARNING):
        unregistered_model._config()
    assert caplog.messages == [
        "UnregisteredTestModelWithNoBaseClass is not registered. It will not work"
        " with AutoModel.from_pretrained() or"
        " AutoModel.from_config(). Consider to annotate the class with"
        " @Model.register() or @Model.register(name='...') to register it at as a Model"
        " which will allow to load it via AutoModel."
    ]


def test_config_registered_but_no_base_class(caplog) -> None:

    @Model.register()
    class RegisteredTestModelWithNoBaseClass(Model):
        BASE_CLASS = None

    model = RegisteredTestModelWithNoBaseClass()
    assert model.BASE_CLASS is Model
    with caplog.at_level(logging.WARNING):
        model._config()
    assert caplog.messages == []
