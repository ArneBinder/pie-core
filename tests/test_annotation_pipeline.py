import json
from typing import List, Sequence, Union, overload

from pie_core import AnnotationPipeline, Document, Model
from tests.fixtures.taskmodules import TestTaskModule


# TODO: Replace with existing Model when tests/model will be merged
@Model.register("TestPipelineModel")
class TestModel(Model):
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


@AnnotationPipeline.register("TestPipeline")
class TestAnnotationPipeline(AnnotationPipeline[TestModel, TestTaskModule]):

    @overload
    def __call__(
        self,
        documents: Document,
        inplace: bool = True,
        *args,
        **kwargs,
    ) -> Document: ...

    @overload
    def __call__(
        self,
        documents: Sequence[Document],
        inplace: bool = True,
        *args,
        **kwargs,
    ) -> Sequence[Document]: ...

    def __call__(
        self, documents: Union[Document, Sequence[Document]], inplace: bool = True, *args, **kwargs
    ) -> Union[Document, Sequence[Document]]:
        pass


def test_annotation_pipeline() -> None:
    pipeline = TestAnnotationPipeline()
    pass
