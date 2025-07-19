from typing import Sequence, Union

import pytest

from pie_core import AnnotationPipeline, Document
from tests.fixtures.model import TestModel
from tests.fixtures.taskmodule import TestDocumentWithLabel, TestTaskModule
from tests.fixtures.types import Label


@AnnotationPipeline.register("TestPipeline")
class TestAnnotationPipeline(AnnotationPipeline[TestModel, TestTaskModule]):
    def __call__(
        self, documents: Union[Document, Sequence[Document]], inplace: bool = True, *args, **kwargs
    ) -> Union[Document, Sequence[Document]]:
        task_encodings = self.taskmodule.encode(
            documents, as_task_encoding_sequence=True, encode_target=True
        )
        model_outputs = [
            self.model.forward(task_encoding.inputs) for task_encoding in task_encodings
        ]
        task_outputs = [
            self.taskmodule.unbatch_output(model_output)[0] for model_output in model_outputs
        ]
        output_documents = self.taskmodule.decode(task_encodings, task_outputs, inplace=inplace)
        return output_documents

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


@pytest.fixture(scope="module")
def documents() -> list[TestDocumentWithLabel]:
    """
    - Creates example documents with predefined texts.
    - Assigns labels to the documents for testing purposes.

    """
    doc1 = TestDocumentWithLabel(text="May your code be bug-free and your algorithms optimized!")
    doc2 = TestDocumentWithLabel(
        text="A cascading failure occurred, resulting in a complete system crash and irreversible data loss."
    )
    # add labels
    doc1.label.append(Label(label="Positive"))
    doc2.label.append(Label(label="Negative"))
    return [doc1, doc2]


def test_annotation_pipeline(documents) -> None:

    model = TestModel()

    taskmodule = TestTaskModule()
    taskmodule.prepare(documents)

    pipeline = TestAnnotationPipeline(model=model, taskmodule=taskmodule)
    pipeline(documents, inplace=True)
    assert documents[0].label.predictions.resolve() == ["Positive"]
    assert documents[1].label.predictions.resolve() == ["Positive"]
