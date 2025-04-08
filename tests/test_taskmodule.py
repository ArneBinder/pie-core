from copy import copy
from dataclasses import dataclass
from typing import Any, Dict

import pytest

from pie_core import AnnotationLayer, Document, annotation_field
from tests.common.taskmodules import SimpleTransformerTextClassificationTaskModule
from tests.common.types import Label


def _config_to_str(cfg: Dict[str, Any]) -> str:
    # Converts a configuration dictionary to a string representation
    result = "-".join([f"{k}={cfg[k]}" for k in sorted(cfg)])
    return result


CONFIGS = [
    {},
]

CONFIGS_DICT = {_config_to_str(cfg): cfg for cfg in CONFIGS}


@pytest.fixture(scope="module", params=CONFIGS_DICT.keys())
def config(request) -> Dict[str, Any]:
    """
    - Provides taskmodule configuration for testing.
    - Yields config dictionaries from the CONFIGS list to produce clean test case identifiers.

    """
    return CONFIGS_DICT[request.param]


@pytest.fixture(scope="module")
def unprepared_taskmodule(config) -> SimpleTransformerTextClassificationTaskModule:
    """
    - Prepares a task module with the specified tokenizer and configuration.
    - Sets up the task module with an unprepared state for testing purposes.

    """
    return SimpleTransformerTextClassificationTaskModule(**config)


def test_taskmodule(unprepared_taskmodule) -> None:
    assert unprepared_taskmodule is not None
    assert not unprepared_taskmodule.is_prepared


@dataclass
class ExampleDocument(Document):
    text: str
    label: AnnotationLayer[Label] = annotation_field()


@pytest.fixture(scope="module")
def documents() -> list[ExampleDocument]:
    """
    - Creates example documents with predefined texts.
    - Assigns labels to the documents for testing purposes.

    """
    doc1 = ExampleDocument(text="May your code be bug-free and your algorithms optimized!")
    doc2 = ExampleDocument(
        text="A cascading failure occurred, resulting in a complete system crash and irreversible data loss."
    )
    # assign label
    label1 = Label(label="Positive")
    label2 = Label(label="Negative")
    # add label
    doc1.label.append(label1)
    doc2.label.append(label2)
    return [doc1, doc2]


@pytest.fixture(scope="module")
def taskmodule(unprepared_taskmodule, documents) -> SimpleTransformerTextClassificationTaskModule:
    """
    - Prepares the task module with the given documents, i.e. collect available label values.
    - Calls the necessary methods to prepare the task module with the documents.
    - Calls _prepare(documents) and then _post_prepare()

    """
    unprepared_taskmodule.prepare(documents)
    return unprepared_taskmodule


def test_prepare(taskmodule) -> None:
    assert taskmodule is not None
    assert taskmodule.is_prepared
    assert taskmodule.config == {
        "taskmodule_type": "SimpleTransformerTextClassificationTaskModule",
        "labels": ["Negative", "Positive"],
    }


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


def test_decode(task_encodings, task_outputs, taskmodule, documents) -> None:
    # use inplace=False to not modify the original documents
    docs_with_predictions = taskmodule.decode(task_encodings, task_outputs, inplace=False)
    assert len(docs_with_predictions) == 2
    # check if the documents are different
    assert docs_with_predictions[0] is not documents[0]
    assert docs_with_predictions[1] is not documents[1]
    # assert docs_with_predictions[0] == documents[0]
    assert documents[0].label.resolve() == ["Positive"]
    assert docs_with_predictions[0].label.predictions.resolve() == ["Negative"]
    # assert docs_with_predictions[1] == documents[1]
    assert documents[1].label.resolve() == ["Negative"]
    assert docs_with_predictions[1].label.predictions.resolve() == ["O"]
