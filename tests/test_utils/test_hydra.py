from importlib import import_module

import pytest

from pie_core import Document
from pie_core.utils.hydra import (
    InstantiationException,
    resolve_optional_document_type,
    resolve_target,
    resolve_type,
    serialize_type,
)
from tests.fixtures.types import (
    TestDocument,
    TestDocumentWithEntities,
    TestDocumentWithSentences,
    TextBasedDocument,
)


def test_resolve_target_string():
    target_str = "pie_core.utils.hydra.resolve_target"
    target = resolve_target(target_str)
    assert target == resolve_target


def test_resolve_target_not_found():
    with pytest.raises(InstantiationException):
        resolve_target("does.not.exist", full_key="full_key")


def test_resolve_target_empty_path():
    with pytest.raises(InstantiationException):
        resolve_target("")


def test_resolve_target_empty_part():
    with pytest.raises(InstantiationException):
        resolve_target("pie_core..hydra.resolve_target")


def test_resolve_target_from_src():
    resolve_target("src.pie_core.utils.hydra.resolve_target")


def test_resolve_target_from_src_not_found():
    with pytest.raises(InstantiationException):
        resolve_target("tests.fixtures.not_loadable")


def test_resolve_target_not_loadable(monkeypatch):
    # Normally, import_module will raise ModuleNotFoundError, but we want to test the case
    # in _locate where it raises a different exception.
    # So we mock the import_module function to raise a different exception on the second call
    # (the first call is important to succeed because otherwise we just check the first try/except block).
    class MockImportModule:
        def __init__(self):
            self.counter = 0

        def __call__(self, path):
            if self.counter < 1:
                self.counter += 1
                return import_module(path)
            raise Exception("Custom exception")

    # Apply the monkeypatch to replace import_module with our mock function
    monkeypatch.setattr("importlib.import_module", MockImportModule())

    with pytest.raises(Exception):
        resolve_target("src.invalid_attr")


def test_resolve_target_not_callable_with_full_key():
    with pytest.raises(InstantiationException):
        resolve_target("pie_utils.hydra", full_key="full_key")


def test_resolve_optional_document_type():

    assert resolve_optional_document_type(Document) == Document
    assert resolve_optional_document_type("pie_core.Document") == Document

    assert resolve_optional_document_type(TextBasedDocument) == TextBasedDocument
    assert (
        resolve_optional_document_type("tests.fixtures.types.TextBasedDocument")
        == TextBasedDocument
    )


def test_resolve_optional_document_type_none():
    assert resolve_optional_document_type(None) is None


class NoDocument:
    pass


def test_resolve_optional_document_type_no_document():
    with pytest.raises(TypeError) as excinfo:
        resolve_optional_document_type(NoDocument)
    assert (
        str(excinfo.value)
        == "type must be a subclass of <class 'pie_core.document.Document'> or a string "
        "that resolves to that, but got <class 'test_hydra.NoDocument'>"
    )

    with pytest.raises(TypeError) as excinfo:
        resolve_optional_document_type("tests.test_utils.test_hydra.NoDocument")
    assert (
        str(excinfo.value)
        == "type must be a subclass of <class 'pie_core.document.Document'> or a string "
        "that resolves to that, but got <class 'tests.test_utils.test_hydra.NoDocument'>"
    )


def test_serialize_type():

    serialized_dt = serialize_type(TestDocument)
    assert serialized_dt == "tests.fixtures.types.TestDocument"
    resolved_dt = resolve_optional_document_type(serialized_dt)
    assert resolved_dt == TestDocument


def test_resolve_document_type():
    assert resolve_type(TestDocumentWithEntities) == TestDocumentWithEntities
    assert (
        resolve_type("tests.fixtures.types.TestDocumentWithEntities") == TestDocumentWithEntities
    )
    with pytest.raises(TypeError) as exc_info:
        resolve_type("tests.test_utils.test_hydra.test_resolve_document_type")
    assert str(exc_info.value).startswith(
        "type must be a subclass of None or a string that resolves to that, but got "
        "<function test_resolve_document_type"
    )

    assert (
        resolve_type(TestDocumentWithEntities, expected_super_type=TextBasedDocument)
        == TestDocumentWithEntities
    )
    with pytest.raises(TypeError) as exc_info:
        resolve_type(TestDocumentWithEntities, expected_super_type=TestDocumentWithSentences)
    assert (
        str(exc_info.value)
        == f"type must be a subclass of {TestDocumentWithSentences} or a string "
        f"that resolves to that, but got {TestDocumentWithEntities}"
    )
