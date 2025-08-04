import copy
import json
import logging
import os
from collections.abc import Generator, Sequence
from typing import Iterable, Iterator, List, Optional, Union

import pytest

from pie_core import AutoTaskModule, TaskEncoding
from pie_core.taskmodule import (
    DocumentType,
    InputEncoding,
    TargetEncoding,
)
from tests import FIXTURES_ROOT
from tests.fixtures.taskmodule import TestDocumentWithLabel, TestTaskModule
from tests.fixtures.types import Label

logger = logging.getLogger(__name__)

PRETRAINED_PATH = FIXTURES_ROOT / "pretrained" / "taskmodule"


@pytest.fixture(scope="module")
def unprepared_taskmodule() -> TestTaskModule:
    return TestTaskModule()


def test_taskmodule(unprepared_taskmodule) -> None:
    assert unprepared_taskmodule is not None
    assert not unprepared_taskmodule.is_prepared


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


@pytest.fixture(scope="module")
def taskmodule(unprepared_taskmodule, documents) -> TestTaskModule:
    """
    - Prepares the task module with the given documents, i.e. collect available label values.
    - Calls the necessary methods to prepare the task module with the documents.
    - Calls _prepare(documents) and then _post_prepare()

    """
    unprepared_taskmodule.prepare(documents)
    return unprepared_taskmodule


@pytest.fixture(scope="module")
def label_ids(taskmodule, documents) -> List[int]:
    return [taskmodule.label_to_id[label.label] for doc in documents for label in doc.label]


def test_label_ids(taskmodule, label_ids) -> None:
    assert label_ids == [2, 1]  # Positive for doc1, Negative for doc2
    assert [taskmodule.id_to_label[label_id] for label_id in label_ids] == [
        "Positive",
        "Negative",
    ]


@pytest.fixture(scope="module")
def config_as_dict() -> dict:
    return {"taskmodule_type": "TestTaskModule", "labels": ["Negative", "Positive"]}


@pytest.fixture(scope="module")
def config_as_json(config_as_dict) -> str:
    return json.dumps(config_as_dict, indent=2)


def test_prepare(taskmodule, config_as_dict) -> None:
    assert taskmodule is not None
    assert taskmodule.is_prepared
    assert taskmodule.config == config_as_dict
    assert_is_post_prepared(taskmodule)


def assert_is_post_prepared(taskmodule) -> None:
    assert taskmodule.label_to_id == {"Negative": 1, "O": 0, "Positive": 2}
    assert taskmodule.id_to_label == {0: "O", 1: "Negative", 2: "Positive"}


def test_from_config(taskmodule) -> None:
    taskmodule_from_config = TestTaskModule.from_config(taskmodule.config)
    assert type(taskmodule_from_config) is TestTaskModule
    assert taskmodule_from_config.is_prepared
    assert taskmodule_from_config.config == taskmodule.config
    assert_is_post_prepared(taskmodule_from_config)


@pytest.fixture(scope="module")
def task_encoding_without_targets(taskmodule, documents):
    """
    - Generates input encodings for a specific task from a document, but without associated targets.

    """
    return taskmodule.encode_input(documents[0])


def test_encode_input(task_encoding_without_targets, documents, taskmodule) -> None:
    assert task_encoding_without_targets is not None
    assert task_encoding_without_targets.document == documents[0]
    assert not task_encoding_without_targets.has_targets

    input_ids = task_encoding_without_targets.inputs
    assert input_ids == [1, 2, 3, 4, 5, 6, 2, 7, 8]
    input_tokens = taskmodule.token_ids2tokens(input_ids)
    assert input_tokens == [
        "May",
        "your",
        "code",
        "be",
        "bug-free",
        "and",
        "your",
        "algorithms",
        "optimized!",
    ]

    assert task_encoding_without_targets.metadata == {}
    assert not task_encoding_without_targets.has_targets


@pytest.fixture(scope="module")
def targets(taskmodule, task_encoding_without_targets) -> int:
    """
    - Encodes the target for a given task encoding.
    - Generates encoded targets for a specific task encoding.

    """
    return taskmodule.encode_target(task_encoding_without_targets)


def test_targets(targets, taskmodule) -> None:
    expected_label = "Positive"
    label_tokens = taskmodule.id_to_label[targets]
    assert label_tokens == expected_label


@pytest.fixture(scope="module")
def task_encodings(taskmodule, documents):
    return taskmodule.encode(documents, encode_target=True)


@pytest.fixture(scope="module")
def task_encoding(taskmodule, task_encodings, targets):
    return task_encodings[0]


def test_task_encoding(task_encoding, task_encoding_without_targets, targets):
    assert task_encoding is not None
    assert task_encoding.document == task_encoding_without_targets.document
    assert task_encoding.inputs == task_encoding_without_targets.inputs
    assert task_encoding.has_targets
    assert task_encoding.targets == targets


def test_encode_from_iterable(taskmodule, documents, task_encodings):

    def documents_iterator():
        yield from documents

    task_encodings_iter = taskmodule.encode(documents_iterator(), encode_target=True)
    assert not isinstance(task_encodings_iter, Sequence)
    assert isinstance(task_encodings_iter, Generator)
    task_encodings_list = list(task_encodings_iter)
    assert len(task_encodings_list) == len(task_encodings)
    for task_encoding1, task_encoding2 in zip(task_encodings, task_encodings_list):
        assert task_encoding1.document == task_encoding2.document
        assert task_encoding1.inputs == task_encoding2.inputs
        assert task_encoding1.targets == task_encoding2.targets


def test_collate_without_targets(taskmodule, task_encoding_without_targets):
    batch_without_targets = taskmodule.collate(
        [task_encoding_without_targets, task_encoding_without_targets]
    )
    assert batch_without_targets is not None
    inputs, targets = batch_without_targets
    assert targets is None
    assert inputs == [[1, 2, 3, 4, 5, 6, 2, 7, 8], [1, 2, 3, 4, 5, 6, 2, 7, 8]]


def test_collate(taskmodule, task_encoding):
    batch = taskmodule.collate([task_encoding, task_encoding])
    assert batch is not None
    inputs, targets = batch
    assert inputs == [[1, 2, 3, 4, 5, 6, 2, 7, 8], [1, 2, 3, 4, 5, 6, 2, 7, 8]]
    assert targets == [2, 2]


@pytest.fixture(scope="module")
def task_outputs(taskmodule):
    """
    - Converts model outputs from batched to unbatched format.
    - Helps in further processing of model outputs for individual task encodings.
    - Model output can be created with model_predict_output fixture above.

    """
    model_output = {"logits": [[0.0513, 0.7510, -0.3345], [0.7510, 0.0513, -0.3345]]}
    return taskmodule.unbatch_output(model_output)


def test_unpatch_output(task_outputs):
    assert task_outputs is not None
    assert task_outputs == [
        {"label": "Negative", "probability": 0.5451},
        {"label": "O", "probability": 0.5451},
    ]


@pytest.fixture(scope="module")
def annotations_from_output(taskmodule, task_encoding_without_targets, task_outputs):
    """Converts the inputs (task_encoding_without_targets) and the respective model outputs
    (unbatched_outputs) into human-readable  annotations."""
    task_encodings = [task_encoding_without_targets, task_encoding_without_targets]
    assert len(task_encodings) == len(task_outputs)
    named_annotations = []
    for _task_encoding, _task_output in zip(task_encodings, task_outputs):
        annotations = taskmodule.create_annotations_from_output(_task_encoding, _task_output)
        named_annotations.extend(annotations)
    return named_annotations


def test_annotations_from_output(annotations_from_output):
    assert annotations_from_output is not None
    assert len(annotations_from_output) == 2
    assert annotations_from_output[0] == (
        "label",
        Label(label="Negative", score=0.5451174378395081),
    )
    assert annotations_from_output[1] == ("label", Label(label="O", score=0.5451174378395081))


@pytest.mark.parametrize("inplace", [True, False])
def test_decode(task_outputs, taskmodule, documents, inplace) -> None:
    # create a copy of the documents to not modify the original documents
    documents = copy.deepcopy(documents)
    task_encodings = taskmodule.encode(documents, encode_target=False)
    # use inplace=False to not modify the original documents
    docs_with_predictions = taskmodule.decode(task_encodings, task_outputs, inplace=inplace)
    if inplace:
        # check if the documents are the same
        for doc, doc_with_pred in zip(documents, docs_with_predictions):
            assert doc is doc_with_pred
    else:
        # check if the documents are different
        for doc, doc_with_pred in zip(documents, docs_with_predictions):
            assert doc is not doc_with_pred

    # check annotations
    labels_resolved = [doc.label.resolve() for doc in documents]
    labels_predictions_resolved = [
        doc_with_pred.label.predictions.resolve() for doc_with_pred in docs_with_predictions
    ]
    assert labels_resolved == [["Positive"], ["Negative"]]
    assert labels_predictions_resolved == [["Negative"], ["O"]]


def test_from_config_unregistered_taskmodule(documents, caplog):
    # Unregistered test class
    class UnregisteredTaskModule(TestTaskModule):
        pass

    unregistered_taskmodule = UnregisteredTaskModule()
    unregistered_taskmodule.prepare(documents)
    assert (
        unregistered_taskmodule.base_class().registered_name_for_class(UnregisteredTaskModule)
        is None
    )

    with caplog.at_level(logging.WARNING):
        unregistered_taskmodule._config()
    assert caplog.messages == [
        "UnregisteredTaskModule is not registered. It will not work "
        "with AutoTaskModule.from_pretrained() or "
        "AutoTaskModule.from_config(). Consider to annotate the class with "
        "@TaskModule.register() or @TaskModule.register(name='...') "
        "to register it as a TaskModule which will allow to load it via AutoTaskModule."
    ]


def assert_task_encodings(
    task_encodings: Iterable[TaskEncoding],
    expected_documents: Optional[List],
    expected_inputs: List[List[int]],
    expected_targets: Optional[List[int]] = None,
) -> None:
    for idx, task_encoding in enumerate(task_encodings):
        assert task_encoding.inputs == expected_inputs[idx]
        if expected_targets is not None:
            assert task_encoding.targets == expected_targets[idx]
        else:
            assert not task_encoding.has_targets
        assert task_encoding.document == expected_documents[idx]

        assert task_encoding.metadata == {}


@pytest.mark.parametrize(
    ["batch_size", "encode_target", "show_progress"], [[None, False, False], [1, True, True]]
)
def test_encoding_iterator(
    documents, label_ids, taskmodule, batch_size, encode_target, show_progress, caplog
):
    encodings_iterator = taskmodule._encoding_iterator(
        documents, encode_target=encode_target, batch_size=batch_size, show_progress=show_progress
    )
    assert isinstance(encodings_iterator, Iterator)

    assert_task_encodings(
        task_encodings=encodings_iterator,
        expected_documents=documents,
        expected_inputs=[
            [1, 2, 3, 4, 5, 6, 2, 7, 8],
            [9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 6, 19, 20, 21],
        ],
        expected_targets=label_ids if encode_target else None,
    )

    if batch_size is not None and show_progress:
        assert caplog.messages == [
            "do not show document encoding progress because we encode lazily with an iterator"
        ]


def test_encode_single_document(taskmodule, documents, label_ids):
    document = documents[0]
    encodings = taskmodule.encode(document, encode_target=True)
    assert_task_encodings(
        task_encodings=encodings,
        expected_documents=[document],
        expected_inputs=[[1, 2, 3, 4, 5, 6, 2, 7, 8]],
        expected_targets=[label_ids[0]],
    )


def test_encode_multiple_documents(taskmodule, documents, label_ids):
    documents = copy.deepcopy(documents)
    # Following Documents will be discarded with current encode_target() implementation
    documents.extend(
        [
            TestDocumentWithLabel(text="Some text with empty labels list"),
            TestDocumentWithLabel(text="Some other text with no label"),
        ]
    )
    # And a document with no text that will be discarded by encode_input()
    documents.extend(
        [
            TestDocumentWithLabel(text=""),
        ]
    )
    documents[3].label = []
    encodings = taskmodule.encode(documents, encode_target=True)
    assert_task_encodings(
        task_encodings=encodings,
        expected_documents=documents,
        expected_inputs=[
            [1, 2, 3, 4, 5, 6, 2, 7, 8],
            [9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 6, 19, 20, 21],
        ],
        expected_targets=label_ids,
    )


def test_encode_as_iterator_as_task_encoding_sequence(taskmodule, documents):
    with pytest.raises(ValueError) as excinfo:
        taskmodule.encode(
            documents, encode_target=False, as_iterator=True, as_task_encoding_sequence=True
        )
    assert excinfo.value.args[0] == "can not return a TaskEncodingSequence as Iterator"


def test_save_pretrained(taskmodule, tmp_path, config_as_json):
    taskmodule.save_pretrained(tmp_path)
    assert os.path.exists(tmp_path / "taskmodule_config.json")
    with open(tmp_path / "taskmodule_config.json") as f:
        config = f.read()
    assert config == config_as_json


def test_from_pretrained(taskmodule):
    from_pretrained_taskmodule = TestTaskModule.from_pretrained(PRETRAINED_PATH)
    assert isinstance(from_pretrained_taskmodule, TestTaskModule)
    assert from_pretrained_taskmodule.is_from_pretrained
    assert from_pretrained_taskmodule.is_prepared
    config = {"is_from_pretrained": True}
    config.update(taskmodule.config)
    assert from_pretrained_taskmodule.config == config


def test_save_and_from_pretrained(taskmodule, tmp_path):
    taskmodule.save_pretrained(tmp_path)
    from_pretrained_taskmodule = TestTaskModule.from_pretrained(tmp_path)
    assert isinstance(from_pretrained_taskmodule, TestTaskModule)
    assert from_pretrained_taskmodule.is_from_pretrained
    config = {"is_from_pretrained": True}
    config.update(taskmodule.config)
    assert from_pretrained_taskmodule.config == config


def test_encode_inputs_with_encode_input_returns_none() -> None:
    class NoneOrListEncoderTaskModule(TestTaskModule):
        """This taskmodules encode_input() returns either an empty List or None.

        We need to test how encode_inputs() handles both cases.
        """

        def encode_input(
            self,
            document: DocumentType,
        ) -> Optional[
            Union[
                TaskEncoding[DocumentType, InputEncoding, TargetEncoding],
                Sequence[TaskEncoding[DocumentType, InputEncoding, TargetEncoding]],
            ]
        ]:
            if document.text == "":
                return None
            else:
                return []

    # set labels to mitigate calling prepare()
    taskmodule = NoneOrListEncoderTaskModule(labels=[])
    documents = [TestDocumentWithLabel(""), TestDocumentWithLabel("ABC")]
    task_encodings_empty = taskmodule.encode(documents=documents)
    assert_task_encodings(
        task_encodings=task_encodings_empty,
        expected_documents=documents,
        expected_inputs=[],
        expected_targets=[],
    )


def test_decode_targets_as_list(task_outputs, taskmodule, documents) -> None:
    # create a copy of the documents to not modify the original documents
    task_encodings = taskmodule.encode(
        documents, encode_target=False, as_task_encoding_sequence=False
    )
    assert isinstance(task_encodings, List)

    docs_with_predictions = taskmodule.decode(task_encodings, task_outputs, inplace=False)

    # check annotations
    labels_resolved = [doc.label.resolve() for doc in documents]
    labels_predictions_resolved = [
        doc_with_pred.label.predictions.resolve() for doc_with_pred in docs_with_predictions
    ]
    assert labels_resolved == [["Positive"], ["Negative"]]
    assert labels_predictions_resolved == [["Negative"], ["O"]]


def test_configure_model_metric(taskmodule, caplog):
    with caplog.at_level(logging.WARNING):
        assert taskmodule.configure_model_metric("Test") is None
    assert caplog.messages == [
        "TaskModule TestTaskModule does not implement a model metric. "
        "Override configure_model_metric(stage: str) to configure a metric for stage 'Test'."
    ]


def test_encode_inputs_with_encode_input_returns_list(documents):
    class DoubledEncodingTaskModule(TestTaskModule):
        """With encode_input() returns each encoding twice."""

        def encode_input(
            self, document: DocumentType
        ) -> Optional[Sequence[TaskEncoding[DocumentType, InputEncoding, TargetEncoding]]]:
            """Create one or multiple task encodings for the given document."""

            task_encoding = super().encode_input(document)
            return [task_encoding, task_encoding]

    taskmodule = DoubledEncodingTaskModule()
    taskmodule.prepare(documents)
    encodings = taskmodule.encode(documents=documents, encode_target=False)
    assert_task_encodings(
        task_encodings=encodings,
        expected_documents=[documents[0], documents[0], documents[1], documents[1]],
        expected_inputs=[
            [1, 2, 3, 4, 5, 6, 2, 7, 8],
            [1, 2, 3, 4, 5, 6, 2, 7, 8],
            [9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 6, 19, 20, 21],
            [9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 6, 19, 20, 21],
        ],
    )


def test_autotaskmodule(taskmodule, tmp_path):
    taskmodule.save_pretrained(tmp_path)
    from_pretrained_taskmodule = AutoTaskModule.from_pretrained(tmp_path)
    assert type(from_pretrained_taskmodule) is TestTaskModule
    assert from_pretrained_taskmodule.is_from_pretrained
    assert from_pretrained_taskmodule.is_prepared
    config = {"is_from_pretrained": True}
    config.update(taskmodule.config)
    assert from_pretrained_taskmodule.config == config
