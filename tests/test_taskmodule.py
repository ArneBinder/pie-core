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
    assert taskmodule.label_to_id == {"O": 0, "Negative": 1, "Positive": 2}
    assert taskmodule.id_to_label == {0: "O", 1: "Negative", 2: "Positive"}


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
def target(taskmodule, task_encoding_without_targets) -> int:
    """
    - Encodes the target for a given task encoding.
    - Generates encoded targets for a specific task encoding.

    """
    return taskmodule.encode_target(task_encoding_without_targets)


def test_target(target, taskmodule) -> None:
    expected_label = "Positive"
    label_tokens = taskmodule.id_to_label[target]
    assert label_tokens == expected_label


@pytest.fixture(scope="module")
def task_encoding(task_encoding_without_targets, target):
    """
    - Combines the task encoding with the associated target.
    - Creates a new task encoding by copying the original and including the target.

    """
    result = copy(task_encoding_without_targets)
    result.targets = target
    return result


def test_task_encoding(task_encoding):
    assert task_encoding is not None


@pytest.fixture(scope="module")
def batch(taskmodule, task_encoding_without_targets):
    """
    - Collates a list of task encodings into a batch.
    - Prepares a batch of task encodings for efficient processing.

    """
    task_encodings = [task_encoding_without_targets, task_encoding_without_targets]
    return taskmodule.collate(task_encodings)


def test_collate(batch, taskmodule):
    assert batch is not None
    assert len(batch) == 2
    inputs, targets = batch
    assert targets is None
    assert inputs == [[1, 2, 3, 4, 5, 6, 2, 7, 8], [1, 2, 3, 4, 5, 6, 2, 7, 8]]


@pytest.fixture(scope="module")
def unbatched_outputs(taskmodule):
    """
    - Converts model outputs from batched to unbatched format.
    - Helps in further processing of model outputs for individual task encodings.
    - Model output can be created with model_predict_output fixture above.

    """
    model_output = {"logits": [[0.0513, 0.7510, -0.3345], [0.7510, 0.0513, -0.3345]]}
    return taskmodule.unbatch_output(model_output)


def test_unpatch_output(unbatched_outputs):
    assert unbatched_outputs is not None
    assert unbatched_outputs == [
        {"label": "Negative", "probability": 0.5451},
        {"label": "O", "probability": 0.5451},
    ]


@pytest.fixture(scope="module")
def annotations_from_output(taskmodule, task_encoding_without_targets, unbatched_outputs):
    """Converts the inputs (task_encoding_without_targets) and the respective model outputs
    (unbatched_outputs) into human-readable  annotations."""
    task_encodings = [task_encoding_without_targets, task_encoding_without_targets]
    assert len(task_encodings) == len(unbatched_outputs)
    named_annotations = []
    for task_encoding, task_output in zip(task_encodings, unbatched_outputs):
        annotations = taskmodule.create_annotations_from_output(task_encoding, task_output)
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
