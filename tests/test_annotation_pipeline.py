import os
from typing import Sequence, Union

import pytest

from pie_core import AnnotationPipeline, AutoAnnotationPipeline, Document
from tests import FIXTURES_ROOT
from tests.fixtures.model import TestModel
from tests.fixtures.taskmodule import TestDocumentWithLabel, TestTaskModule
from tests.fixtures.types import Label

PRETRAINED_PATH = FIXTURES_ROOT / "pretrained" / "annotation_pipeline"


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
        if isinstance(documents, Document):
            if len(output_documents) == 1:
                return output_documents[0]
            else:
                raise ValueError(
                    "Pipeline should return a single document if input is a single document."
                )

        return output_documents

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


@pytest.fixture(scope="function")
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


@pytest.fixture(scope="function")
def taskmodule(documents) -> TestTaskModule:
    taskmodule = TestTaskModule()
    taskmodule.prepare(documents)
    return taskmodule


@pytest.fixture(scope="function")
def model() -> TestModel:
    return TestModel()


def assert_pipeline_output(
    documents: Union[Document, Sequence[Document]],
    output_documents: Union[Document, Sequence[Document]],
    inplace,
) -> None:
    assert type(output_documents) is type(documents)
    if inplace:
        assert output_documents == documents
    else:
        assert output_documents != documents
    assert output_documents[0].label.predictions.resolve() == ["Positive"]
    assert output_documents[1].label.predictions.resolve() == ["Positive"]


@pytest.mark.parametrize("inplace", [True, False])
def test_annotation_pipeline(documents, model, taskmodule, inplace) -> None:
    pipeline = TestAnnotationPipeline(model=model, taskmodule=taskmodule)
    output_documents = pipeline(documents, inplace=inplace)
    assert_pipeline_output(documents, output_documents, inplace)


@pytest.mark.parametrize("inplace", [True, False])
def test_annotation_pipeline_single_document(documents, model, taskmodule, inplace) -> None:
    documents = documents[0]
    pipeline = TestAnnotationPipeline(model=model, taskmodule=taskmodule)
    output_documents = pipeline(documents, inplace=inplace)

    assert type(output_documents) is type(documents)
    if inplace:
        assert output_documents == documents
    else:
        assert output_documents != documents
    assert output_documents.label.predictions.resolve() == ["Positive"]


@pytest.mark.parametrize("inplace", [True, False])
def test_annotation_pipeline_taskmodule_passed_in_model(
    documents, model, taskmodule, inplace
) -> None:
    model.taskmodule = taskmodule
    pipeline = TestAnnotationPipeline(model=model)
    output_documents = pipeline(documents, inplace=inplace)

    assert_pipeline_output(documents, output_documents, inplace)


def test_save_pretrained(documents, model, taskmodule, tmp_path) -> None:
    pipeline = TestAnnotationPipeline(model=model, taskmodule=taskmodule)
    pipeline.save_pretrained(tmp_path)
    for config in [
        pipeline.config_name,
        model.config_name,
        model.weights_file_name,
        taskmodule.config_name,
    ]:
        assert os.path.exists(os.path.join(tmp_path, config))


def test_from_pretrained(documents) -> None:
    pipeline = TestAnnotationPipeline.from_pretrained(PRETRAINED_PATH)
    assert isinstance(pipeline, TestAnnotationPipeline)
    assert isinstance(pipeline.taskmodule, TestTaskModule)
    assert isinstance(pipeline.model, TestModel)

    output_documents = pipeline(documents)
    assert_pipeline_output(documents, output_documents, True)


@pytest.mark.parametrize("inplace", [True, False])
def test_save_and_from_pretrained(documents, model, taskmodule, tmp_path, inplace) -> None:
    pipeline = TestAnnotationPipeline(model=model, taskmodule=taskmodule)
    pipeline.save_pretrained(tmp_path)
    from_pretrained_pipeline = TestAnnotationPipeline.from_pretrained(tmp_path)

    assert isinstance(from_pretrained_pipeline, TestAnnotationPipeline)
    assert isinstance(from_pretrained_pipeline.taskmodule, TestTaskModule)
    assert isinstance(from_pretrained_pipeline.model, TestModel)

    output_documents = from_pretrained_pipeline(documents, inplace=inplace)
    assert_pipeline_output(documents, output_documents, inplace=inplace)


@pytest.mark.parametrize("inplace", [True, False])
def test_auto_annotation_pipeline_from_pretrained(
    documents, model, taskmodule, tmp_path, inplace
) -> None:
    pipeline = AutoAnnotationPipeline.from_pretrained(PRETRAINED_PATH)
    assert type(pipeline) is TestAnnotationPipeline
    assert type(pipeline.taskmodule) is TestTaskModule
    assert type(pipeline.model) is TestModel

    output_documents = pipeline(documents, inplace=inplace)
    assert_pipeline_output(documents, output_documents, inplace=inplace)
