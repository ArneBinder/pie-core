import logging
from typing import List, Optional

import pytest

from pie_core import Document, DocumentStatistic
from tests import FIXTURES_ROOT
from tests.fixtures.types import LabeledSpan, TestDocumentWithEntities


@pytest.fixture
def documents():

    # a test sentence with two entities
    doc1 = TestDocumentWithEntities(
        text="The quick brown fox jumps over the lazy dog.",
    )
    doc1.entities.append(LabeledSpan(start=4, end=19, label="animal"))
    doc1.entities.append(LabeledSpan(start=35, end=43, label="animal"))
    assert str(doc1.entities[0]) == "quick brown fox"
    assert str(doc1.entities[1]) == "lazy dog"

    # a second test sentence with a different text and a single entity (a company)
    doc2 = TestDocumentWithEntities(text="Apple is a great company.")
    doc2.entities.append(LabeledSpan(start=0, end=5, label="company"))
    assert str(doc2.entities[0]) == "Apple"

    documents = [doc1, doc2]

    return documents


class CharacterCountCollector(DocumentStatistic):

    def _collect(self, doc: Document) -> int:
        return len(doc.text)


def test_CharacterCountCollector(documents):
    statistic = CharacterCountCollector()
    values = statistic(documents)
    assert values == {"mean": 34.5, "std": 9.5, "min": 25, "max": 44}


def test_CharacterCountCollector_with_empty_input():
    statistic = CharacterCountCollector()
    values = statistic([])
    assert values == {}


def test_median_aggregated_function(documents):
    statistic = CharacterCountCollector(aggregation_functions=["median"])
    values = statistic(documents)
    assert values == {"median": 44}


def test_builtin_aggregated_funtion(documents):
    statistic = CharacterCountCollector(aggregation_functions=["all"])
    values = statistic(documents)
    assert values == {"all": True}


def calculate_product(values: List[float]) -> Optional[float]:
    if len(values) == 0:
        return None
    result = 1
    for value in values:
        result *= value
    return result


def test_selfbuilt_aggregated_funtion(documents):
    statistic = CharacterCountCollector(
        aggregation_functions=["tests.test_statistic.calculate_product"]
    )
    values = statistic(documents)
    assert values == {"tests.test_statistic.calculate_product": 1100}


def test_selfbuilt_invalid_funtion(documents):
    with pytest.raises(ImportError) as error:
        statistic = CharacterCountCollector(
            aggregation_functions=["tests.test_statistic.invalid_function"]
        )
    assert "Cannot resolve aggregation function: tests.test_statistic.invalid_function" in str(
        error.value
    )


class ListCharacterCountCollector(DocumentStatistic):

    def _collect(self, doc: Document) -> int:
        return [factor * len(doc.text) for factor in range(5)]


def test_ListCharacterCountCollector(documents):
    statistic = ListCharacterCountCollector()
    values = statistic(documents)
    assert values == {"mean": 69.0, "std": 54.055527006958314, "min": 0, "max": 176}


def test_aggregated_function_with_lists(documents):
    statistic = CharacterCountCollector(
        aggregation_functions=["median", "all", "tests.test_statistic.calculate_product"]
    )
    values = statistic(documents)
    assert values == {"median": 44, "all": True, "tests.test_statistic.calculate_product": 1100}


class DictCharacterCountCollector(DocumentStatistic):

    def _collect(self, doc: Document) -> int:
        return {char: doc.text.count(char) for char in "aeiou"}


def test_DictCharacterCountCollector(documents):
    statistic = DictCharacterCountCollector()
    values = statistic(documents)
    assert values == {
        "a": {"mean": 2.0, "std": 1.0, "min": 1, "max": 3},
        "e": {"mean": 2.5, "std": 0.5, "min": 2, "max": 3},
        "i": {"mean": 1.0, "std": 0.0, "min": 1, "max": 1},
        "o": {"mean": 2.5, "std": 1.5, "min": 1, "max": 4},
        "u": {"mean": 1.0, "std": 1.0, "min": 0, "max": 2},
    }


def test_aggregated_function_with_dicts(documents):
    statistic = CharacterCountCollector(
        aggregation_functions=["median", "all", "tests.test_statistic.calculate_product"]
    )
    values = statistic(documents)
    assert values == {"median": 44, "all": True, "tests.test_statistic.calculate_product": 1100}


class DictVowelIndecesCollector(DocumentStatistic):

    def _collect(self, doc: Document) -> int:
        result = {}
        for idx, char in enumerate(doc.text):
            if char in "aeiou":
                result.setdefault(char, []).append(idx)
        return result


def test_DictVowelIndecesCollector(documents):
    statistic = DictVowelIndecesCollector()
    values = statistic(documents)
    assert values == {
        "e": {"mean": 16.0, "std": 12.505998560690786, "min": 2, "max": 33},
        "u": {"mean": 13.0, "std": 8.0, "min": 5, "max": 21},
        "i": {"mean": 6.0, "std": 0.0, "min": 6, "max": 6},
        "o": {"mean": 22.8, "std": 10.146920715172657, "min": 12, "max": 41},
        "a": {"mean": 20.0, "std": 10.173494974687902, "min": 9, "max": 36},
    }


def test_aggregated_function_with_dicts_of_lists(documents):
    statistic = DictVowelIndecesCollector(
        aggregation_functions=["median", "all", "tests.test_statistic.calculate_product"]
    )
    values = statistic(documents)
    assert values == {
        "e": {"median": 13, "all": True, "tests.test_statistic.calculate_product": 96096},
        "u": {"median": 21, "all": True, "tests.test_statistic.calculate_product": 105},
        "i": {"median": 6, "all": True, "tests.test_statistic.calculate_product": 36},
        "o": {"median": 18, "all": True, "tests.test_statistic.calculate_product": 3914352},
        "a": {"median": 21, "all": True, "tests.test_statistic.calculate_product": 95256},
    }


def test_show_as_markdown(documents, caplog):
    statistic = CharacterCountCollector(show_as_markdown=True)
    with caplog.at_level(logging.INFO):
        values = statistic(documents)

    assert caplog.messages == [
        "CharacterCountCollector (2 documents)\n"
        "|      |    0 |\n"
        "|:-----|-----:|\n"
        "| mean | 34.5 |\n"
        "| std  |  9.5 |\n"
        "| min  | 25   |\n"
        "| max  | 44   |"
    ]


def test_show_histogram(documents, capsys):
    statistic = CharacterCountCollector(show_histogram=True)
    statistic(documents)

    # get the captured output
    captured = capsys.readouterr()

    fixture_file = FIXTURES_ROOT / "statistic" / "show_histogram_plotext.txt"

    # Check if the output matches the expected output
    with open(fixture_file) as f:
        expected_output = f.read()

    # assert that the captured output matches the expected output
    assert captured.out == expected_output


@pytest.fixture
def collection(documents):
    return {"test": documents[0], "train": documents[1]}


def test_collection_as_input(collection):
    statistic = CharacterCountCollector()
    values = statistic(collection)

    assert values == {
        "test": {"mean": 44.0, "std": 0.0, "min": 44, "max": 44},
        "train": {"mean": 25.0, "std": 0.0, "min": 25, "max": 25},
    }
