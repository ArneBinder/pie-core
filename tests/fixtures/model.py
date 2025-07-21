import json
from typing import Any, Dict, List

from typing_extensions import TypeAlias

from pie_core import AutoTaskModule, Model, TaskModule

ModelInputType: TypeAlias = List[int]
ModelOutputType: TypeAlias = Dict[str, Any]


@Model.register()
class TestModel(Model):
    weights_file_name = "model.json"
    param: List[int]

    def __init__(self, param=None, taskmodule=None, **kwargs):
        super().__init__(**kwargs)
        if param is None:
            param = [0, 0, 0]
        self.param = param
        if isinstance(taskmodule, dict):
            self.taskmodule = AutoTaskModule.from_config(taskmodule)
        elif isinstance(taskmodule, TaskModule):
            self.taskmodule = taskmodule

    def save_model_file(self, model_file: str) -> None:
        state_dict = {"param": self.param}
        json.dump(state_dict, open(model_file, "w"), indent=2)

    def load_model_file(
        self, model_file: str, map_location: str = "cpu", strict: bool = False
    ) -> None:
        state_dict = json.load(open(model_file))
        self.param = state_dict["param"]

    def forward(self, model_input: ModelInputType) -> ModelOutputType:
        output = [[x + y for x, y in zip(self.param, model_input)] * 2]
        return {"logits": output}
