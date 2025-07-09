import logging
import os

import pytest
import torch
from torch import nn

from pie_core import AutoModel, Model
from tests import FIXTURES_ROOT

logger = logging.getLogger(__name__)

CONFIG_PATH = FIXTURES_ROOT / "configs" / "test-model"


@Model.register("TestModel")
class TestModel(Model, nn.Module):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.param = nn.Parameter(torch.Tensor([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]))
        self.linear = nn.Linear(4, 5)
        with torch.no_grad():
            self.linear.weight.copy_(torch.zeros(self.linear.weight.shape))
        torch.nn.init.zeros_(self.linear.bias)

    def forward(self, x):
        return self.linear(x + self.param)

    def save_model_file(self, model_file: str) -> None:
        torch.save(self.state_dict(), model_file)

    def load_model_file(
        self, model_file: str, map_location: str = "cpu", strict: bool = False
    ) -> None:
        state_dict = torch.load(model_file, map_location=torch.device(map_location))
        self.load_state_dict(state_dict, strict=strict)


class UnimplementedTestModel(Model):
    BASE_CLASS = None


@pytest.fixture
def model():
    return TestModel()


@pytest.fixture
def unimplemented_model():
    return UnimplementedTestModel()


def config_as_dict() -> dict:
    return {"model_type": "TestModel"}


def config_as_json() -> str:
    return '{\n  "model_type": "TestModel"\n}'


def test_save_pretrained(model, tmp_path) -> None:
    model.save_pretrained(tmp_path)
    assert os.path.exists(tmp_path / "config.json")
    with open(tmp_path / "config.json") as f:
        assert f.read() == config_as_json()
    assert os.path.exists(tmp_path / "pytorch_model.bin")


def test_from_pretrained(model) -> None:
    test_model = TestModel.from_pretrained(CONFIG_PATH)
    assert test_model.state_dict().keys() == model.state_dict().keys()

    for key in model.state_dict().keys():
        assert torch.equal(test_model.state_dict()[key], model.state_dict()[key])


def test_auto_model_from_pretrained(model) -> None:
    test_model = AutoModel.from_pretrained(CONFIG_PATH)
    assert test_model.state_dict().keys() == model.state_dict().keys()

    for key in model.state_dict().keys():
        assert torch.equal(test_model.state_dict()[key], model.state_dict()[key])


def test_from_pretrained_warnings(model, caplog) -> None:
    with caplog.at_level(logging.WARNING):
        TestModel.from_pretrained(CONFIG_PATH, map_location="cpu", strict=False)
    assert caplog.messages == [
        'map_location is deprecated. Use load_model_file={"map_location": "cpu"} instead.',
        'strict is deprecated. Use load_model_file={"strict": False} instead.',
    ]


def test_config_warning(unimplemented_model, caplog) -> None:
    with caplog.at_level(logging.WARNING):
        unimplemented_model._config()
    assert caplog.messages == [
        "UnimplementedTestModel does not have a base class. It will not work"
        " with AutoModel.from_pretrained() or"
        " AutoModel.from_config(). Consider to annotate the class with"
        " @Model.register() or @Model.register(name='...') to register it at as a Model"
        " which will allow to load it via AutoModel."
    ]
