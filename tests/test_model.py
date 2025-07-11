import json
import logging
import os
from typing import List

import pytest

from pie_core import AutoModel, Model
from tests import FIXTURES_ROOT

logger = logging.getLogger(__name__)

CONFIG_PATH = FIXTURES_ROOT / "pretrained" / "model"


@Model.register()
class TestModel(Model):
    param: List[int]

    def __init__(self, param=None, **kwargs):
        super().__init__(**kwargs)
        if param is None:
            param = [0, 0, 0]
        self.param = param

    def save_model_file(self, model_file: str) -> None:
        state_dict = {"param": self.param}
        json.dump(state_dict, open(model_file, "w"))

    def load_model_file(
        self, model_file: str, map_location: str = "cpu", strict: bool = False
    ) -> None:
        state_dict = json.load(open(model_file))
        self.param = state_dict["param"]


@pytest.fixture
def model():
    return TestModel()


def config_as_dict() -> dict:
    return {"model_type": "TestModel"}


def config_as_json() -> str:
    return '{\n  "model_type": "TestModel"\n}'


def test_save_pretrained(model, tmp_path) -> None:
    model.save_pretrained(tmp_path)
    assert os.path.exists(tmp_path / model.config_name)
    with open(tmp_path / model.config_name) as f:
        assert f.read() == config_as_json()
    assert os.path.exists(tmp_path / model.weights_file_name)


def test_from_pretrained(model) -> None:
    test_model = TestModel.from_pretrained(CONFIG_PATH)
    assert test_model.param == model.param


def test_auto_model_from_pretrained(model) -> None:
    test_model = AutoModel.from_pretrained(CONFIG_PATH)
    assert test_model.param == model.param


def test_from_pretrained_warnings(model, caplog) -> None:
    with caplog.at_level(logging.WARNING):
        TestModel.from_pretrained(CONFIG_PATH, map_location="cpu", strict=False)
    assert caplog.messages == [
        'map_location is deprecated. Use load_model_file={"map_location": "cpu"} instead.',
        'strict is deprecated. Use load_model_file={"strict": False} instead.',
    ]


def test_config_warning(caplog) -> None:
    class UnregisteredTestModel(Model):
        pass

    unregistered_model = UnregisteredTestModel()

    with caplog.at_level(logging.WARNING):
        unregistered_model._config()
    assert caplog.messages == [
        "UnregisteredTestModel is not registered. It will not work"
        " with AutoModel.from_pretrained() or"
        " AutoModel.from_config(). Consider to annotate the class with"
        " @Model.register() or @Model.register(name='...') to register it at as a Model"
        " which will allow to load it via AutoModel."
    ]
