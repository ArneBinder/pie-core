import logging

import pytest

from tests.common.taskmodules import TestDocumentWithLabel, TestTaskModule
from tests.common.types import Label

logger = logging.getLogger(__name__)


@pytest.fixture(scope="module")
def unprepared_taskmodule() -> TestTaskModule:
    return TestTaskModule()


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
def prepared_taskmodule(documents) -> TestTaskModule:
    """
    - Prepares the task module with the given documents, i.e. collect available label values.
    - Calls the necessary methods to prepare the task module with the documents.
    - Calls _prepare(documents) and then _post_prepare()

    """
    taskmodule = TestTaskModule()
    taskmodule.prepare(documents)
    return taskmodule


def test_prepared_attributes(unprepared_taskmodule, prepared_taskmodule, documents):

    assert not unprepared_taskmodule.is_prepared
    with pytest.raises(Exception) as excinfo:
        attrs = unprepared_taskmodule.prepared_attributes
    assert str(excinfo.value) == "The module is not prepared."

    assert prepared_taskmodule.is_prepared
    assert prepared_taskmodule.prepared_attributes == {"labels": ["Negative", "Positive"]}


def test_assert_is_prepared(unprepared_taskmodule):
    with pytest.raises(Exception) as excinfo:
        unprepared_taskmodule.assert_is_prepared()
    assert str(excinfo.value) == " Required attributes that are not set: ['labels']"


def test_prepare_prepared_taskmodule(prepared_taskmodule, documents, caplog):
    with caplog.at_level(logging.WARNING):
        prepared_taskmodule.prepare(documents)
    assert (
        "The TestTaskModule is already prepared, do not prepare again.\nlabels = ['Negative', 'Positive']"
        in caplog.text
    )
