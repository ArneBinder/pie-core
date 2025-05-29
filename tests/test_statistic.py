from dataclasses import dataclass

import pytest

from pie_core import (
    AnnotationLayer,
    Document,
    DocumentStatistic,
    annotation_field,
)
from tests.common.types import LabeledSpan, TextBasedDocument


@pytest.fixture
def documents():
    @dataclass
    class TextDocumentWithEntities(TextBasedDocument):
        entities: AnnotationLayer[LabeledSpan] = annotation_field(target="text")

    # a test sentence with two entities
    doc1 = TextDocumentWithEntities(
        text="The quick brown fox jumps over the lazy dog.",
    )
    doc1.entities.append(LabeledSpan(start=4, end=19, label="animal"))
    doc1.entities.append(LabeledSpan(start=35, end=43, label="animal"))
    assert str(doc1.entities[0]) == "quick brown fox"
    assert str(doc1.entities[1]) == "lazy dog"

    # a second test sentence with a different text and a single entity (a company)
    doc2 = TextDocumentWithEntities(text="Apple is a great company.")
    doc2.entities.append(LabeledSpan(start=0, end=5, label="company"))
    assert str(doc2.entities[0]) == "Apple"

    documents = [doc1, doc2]

    return documents


class WordCountCollector(DocumentStatistic):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _collect(self, doc: Document) -> int:
        return len(doc.text)


def test_WordCountCollector(documents):
    statistic = WordCountCollector()
    values = statistic(documents)
    assert values == {"mean": 34.5, "std": 9.5, "min": 25, "max": 44}


def test_median_aggregated_function(documents):
    statistic = WordCountCollector(aggregation_functions=["median"])
    values = statistic(documents)
    assert values == {"median": 44}
