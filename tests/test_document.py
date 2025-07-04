import dataclasses
import json
import re
from typing import Dict, List, Optional, Tuple

import pytest

from pie_core import Annotation
from pie_core.document import (
    AnnotationLayer,
    Document,
    _contains_annotation_type,
    _enumerate_dependencies,
    _get_reference_fields_and_container_types,
    _is_annotation_type,
    _is_optional_annotation_type,
    _is_optional_type,
    _is_tuple_of_annotation_types,
    annotation_field,
)
from tests.fixtures.types import (
    BinaryRelation,
    Label,
    LabeledSpan,
    Span,
    TextBasedDocument,
    TokenBasedDocument,
)


def _test_annotation_reconstruction(
    annotation: Annotation, annotation_store: Optional[Dict[int, Annotation]] = None
):
    ann_str = json.dumps(annotation.asdict())
    annotation_reconstructed = type(annotation).fromdict(
        json.loads(ann_str), annotation_store=annotation_store
    )
    assert annotation_reconstructed == annotation


def test_is_optional_type():
    @dataclasses.dataclass(eq=True, frozen=True)
    class Dummy(Annotation):
        a: int
        b: Tuple[int, ...]
        c: Optional[List[int]]
        d: Optional[int] = None

    fields = {field.name: field for field in dataclasses.fields(Dummy)}
    assert not _is_optional_type(fields["a"].type)
    assert not _is_optional_type(fields["b"].type)
    assert _is_optional_type(fields["c"].type)
    assert _is_optional_type(fields["d"].type)


def test_is_optional_annotation_type():
    @dataclasses.dataclass(eq=True, frozen=True)
    class Dummy(Annotation):
        a: int
        b: Optional[int]
        c: Optional[Span]
        d: Optional[Tuple[Span, ...]] = None

    fields = {field.name: field for field in dataclasses.fields(Dummy)}
    assert not _is_optional_annotation_type(fields["a"].type)
    assert not _is_optional_annotation_type(fields["b"].type)
    assert _is_optional_annotation_type(fields["c"].type)
    assert not _is_optional_annotation_type(fields["d"].type)


def test_is_annotation_type():
    @dataclasses.dataclass(eq=True, frozen=True)
    class Dummy(Annotation):
        a: Span
        b: Annotation
        c: Optional[List[int]]
        d: Optional[int] = None

    fields = {field.name: field for field in dataclasses.fields(Dummy)}
    assert _is_annotation_type(fields["a"].type)
    assert _is_annotation_type(fields["b"].type)
    assert not _is_annotation_type(fields["c"].type)
    assert not _is_annotation_type(fields["d"].type)


def test_contains_annotation_type():
    @dataclasses.dataclass(eq=True, frozen=True)
    class Dummy(Annotation):
        a: Span
        b: Annotation
        c: int
        d: Optional[List[Tuple[Optional[Span], ...]]]
        e: Optional[int] = None

    fields = {field.name: field for field in dataclasses.fields(Dummy)}
    assert _contains_annotation_type(
        fields["a"].type
    ), 'field "a" does not contain an annotation type'
    assert _contains_annotation_type(
        fields["b"].type
    ), 'field "b" does not contain an annotation type'
    assert not _contains_annotation_type(
        fields["c"].type
    ), 'field "c" does contain an annotation type'
    assert _contains_annotation_type(
        fields["d"].type
    ), 'field "d" does not contain an annotation type'
    assert not _contains_annotation_type(
        fields["e"].type
    ), 'field "e" does not contain an annotation type'


def test_is_tuple_of_annotation_types():
    @dataclasses.dataclass(eq=True, frozen=True)
    class Dummy(Annotation):
        # a: no
        a: Annotation
        # b: no
        b: int
        # c: no, contains optional elements
        c: Tuple[Optional[Span], ...]
        # d: yes
        d: Tuple[Span, ...]
        # e: no, is optional
        e: Optional[Tuple[Span, ...]]
        # f: yes
        f: Tuple[Span, Span]
        # g: yes
        g: Tuple[Span, ...]
        # h: raise exception because it is mixed with non-Annotation type
        h: Tuple[Span, int]
        # i: raise no exception because it is mixed just with Annotation type subclasses
        i: Tuple[Span, Annotation]

    fields = {field.name: field for field in dataclasses.fields(Dummy)}
    assert not _is_tuple_of_annotation_types(
        fields["a"].type
    ), 'field "a" is a pure tuple of annotation type'
    assert not _is_tuple_of_annotation_types(
        fields["b"].type
    ), 'field "b" is a pure tuple of annotation type'
    assert not _is_tuple_of_annotation_types(
        fields["c"].type
    ), 'field "c" is a pure tuple of annotation type'
    assert _is_tuple_of_annotation_types(
        fields["d"].type
    ), 'field "d" is not a pure tuple of annotation type'
    assert not _is_tuple_of_annotation_types(
        fields["e"].type
    ), 'field "e" is a pure tuple of annotation type'
    assert _is_tuple_of_annotation_types(
        fields["f"].type
    ), 'field "f" is not a pure tuple of annotation type'
    assert _is_tuple_of_annotation_types(
        fields["g"].type
    ), 'field "g" is not a pure tuple of annotation type'
    with pytest.raises(TypeError):
        _is_tuple_of_annotation_types(fields["h"].type)
    assert _is_tuple_of_annotation_types(
        fields["i"].type
    ), 'field "i" is not a pure tuple of annotation type'


def test_get_reference_fields_and_container_types():
    @dataclasses.dataclass(eq=True, frozen=True)
    class Dummy(Annotation):
        a: Annotation
        # This causes an exception because the type of by contains an Annotation subclass (Span),
        # but it is embedded *twice* in a Tuple which is not allowed.
        b: Tuple[Tuple[Span, ...]]

    with pytest.raises(TypeError):
        _get_reference_fields_and_container_types(Dummy)


def test_annotation_with_optional_reference():
    @dataclasses.dataclass(eq=True, frozen=True)
    class BinaryRelationWithOptionalTrigger(Annotation):
        head: Span
        tail: Span
        label: str
        trigger: Optional[Span] = None
        score: float = dataclasses.field(default=1.0, compare=False)

    head = Span(start=1, end=2)
    tail = Span(start=3, end=4)
    trigger = Span(start=5, end=7)

    binary_relation1 = BinaryRelationWithOptionalTrigger(head=head, tail=tail, label="label1")
    assert binary_relation1.head == head
    assert binary_relation1.tail == tail
    assert binary_relation1.label == "label1"
    assert binary_relation1.score == pytest.approx(1.0)

    assert binary_relation1.asdict() == {
        "_id": binary_relation1._id,
        "head": head._id,
        "tail": tail._id,
        "trigger": None,
        "label": "label1",
        "score": 1.0,
    }

    binary_relation2 = BinaryRelationWithOptionalTrigger(
        head=head, tail=tail, label="label2", score=0.5, trigger=trigger
    )
    assert binary_relation2.head == head
    assert binary_relation2.tail == tail
    assert binary_relation2.trigger == trigger
    assert binary_relation2.label == "label2"
    assert binary_relation2.score == pytest.approx(0.5)

    assert binary_relation2.asdict() == {
        "_id": binary_relation2._id,
        "head": head._id,
        "tail": tail._id,
        "trigger": trigger._id,
        "label": "label2",
        "score": 0.5,
    }

    annotation_store = {
        head._id: head,
        tail._id: tail,
        trigger._id: trigger,
    }
    _test_annotation_reconstruction(binary_relation1, annotation_store=annotation_store)
    _test_annotation_reconstruction(binary_relation2, annotation_store=annotation_store)


def test_annotation_with_tuple_of_references():
    @dataclasses.dataclass(eq=True, frozen=True)
    class BinaryRelationWithEvidence(Annotation):
        head: Span
        tail: Span
        label: str
        evidence: Tuple[Span, ...]
        score: float = dataclasses.field(default=1.0, compare=False)

    head = Span(start=1, end=2)
    tail = Span(start=3, end=4)
    evidence1 = Span(start=5, end=7)
    evidence2 = Span(start=9, end=10)

    relation = BinaryRelationWithEvidence(
        head=head, tail=tail, label="label1", evidence=(evidence1, evidence2)
    )
    assert relation.head == head
    assert relation.tail == tail
    assert relation.label == "label1"
    assert relation.score == pytest.approx(1.0)
    assert relation.evidence == (evidence1, evidence2)

    assert relation.asdict() == {
        "_id": relation._id,
        "head": head._id,
        "tail": tail._id,
        "evidence": [evidence1._id, evidence2._id],
        "label": "label1",
        "score": 1.0,
    }

    annotation_store = {
        head._id: head,
        tail._id: tail,
        evidence1._id: evidence1,
        evidence2._id: evidence2,
    }
    _test_annotation_reconstruction(relation, annotation_store=annotation_store)


def test_annotation_sort():
    @dataclasses.dataclass(eq=True, frozen=True)
    class Dummy(Annotation):
        a: int
        b: Optional[int] = None
        c: int = 0

    dummy1 = Dummy(a=1, c=2)
    dummy2 = Dummy(a=1, c=3)
    dummy3 = Dummy(a=2, c=1)
    dummy4 = Dummy(a=2, c=2)

    assert sorted([dummy1, dummy2, dummy3, dummy4]) == [dummy1, dummy2, dummy3, dummy4]

    @dataclasses.dataclass(eq=True, frozen=True)
    class DummyWithNestedAnnotation(Annotation):
        a: int
        n: Dummy

    dummy_nested1 = DummyWithNestedAnnotation(a=1, n=Dummy(a=1, c=2))
    dummy_nested2 = DummyWithNestedAnnotation(a=2, n=Dummy(a=2, c=3))
    dummy_nested3 = DummyWithNestedAnnotation(a=2, n=Dummy(a=1, c=4))
    dummy_nested4 = DummyWithNestedAnnotation(a=1, n=Dummy(a=2, c=2))

    assert sorted([dummy_nested1, dummy_nested2, dummy_nested3, dummy_nested4]) == [
        dummy_nested1,
        dummy_nested4,
        dummy_nested3,
        dummy_nested2,
    ]

    with pytest.raises(ValueError) as excinfo:
        sorted([dummy1, dummy_nested1])
    assert (
        str(excinfo.value) == "comparison field names do not match: ['a', 'n'] != ['a', 'b', 'c']"
    )


def test_annotation_resolve():
    @dataclasses.dataclass(eq=True, frozen=True)
    class Dummy(Annotation):
        a: int

    dummy = Dummy(a=1)
    with pytest.raises(NotImplementedError) as excinfo:
        dummy.resolve()
    assert str(excinfo.value) == f"resolve() is not implemented for {Dummy}"


def test_annotation_is_attached():
    @dataclasses.dataclass
    class MyDocument(Document):
        text: str
        words: AnnotationLayer[Span] = annotation_field(target="text")

    document = MyDocument(text="Hello world!")
    word = Span(start=0, end=5)
    assert not word.is_attached
    document.words.append(word)
    assert word.is_attached
    document.words.pop()
    assert not word.is_attached


def test_annotation_copy():
    @dataclasses.dataclass(eq=True, frozen=True)
    class Attribute(Annotation):
        annotation: Annotation
        label: str

        def __repr__(self):
            return f"Attribute(annotation={self.annotation}, label={self.label})"

    @dataclasses.dataclass
    class MyDocument(Document):
        text: str
        words: AnnotationLayer[Span] = annotation_field(target="text")
        attributes: AnnotationLayer[Attribute] = annotation_field(target="words")

    document = MyDocument(text="Hello world!")
    word = Span(start=0, end=5)
    attribute = Attribute(annotation=word, label="label")
    # both annotations are not yet attached
    assert not word.is_attached
    assert not attribute.is_attached
    # copy the annotations
    attribute_copy0 = attribute.copy()
    word_copy0 = word.copy()
    # now attach the annotations
    document.words.append(word)
    document.attributes.append(attribute)
    assert word.is_attached
    assert attribute.is_attached
    # copy the annotations again
    word_copy1 = word.copy()
    attribute_copy1 = attribute.copy()
    # check that the copies are not attached
    assert not word_copy1.is_attached
    assert not attribute_copy1.is_attached
    # check that the copies have the same values as the originals
    assert word_copy1.start == word.start
    assert word_copy1.end == word.end
    assert attribute_copy1.annotation == attribute.annotation
    assert attribute_copy1.label == attribute.label
    # check that the copies before attaching the originals are the same as the copies after attaching the originals
    assert word_copy1 == word_copy0
    assert attribute_copy1 == attribute_copy0

    # create a copy of the attribute, but let it point to a new word, i.e. overwrite a field
    new_word = Span(start=6, end=11)
    document.words.append(new_word)
    attribute_copy2 = attribute.copy(annotation=new_word)
    document.attributes.append(attribute_copy2)
    assert len(document.attributes) == 2
    assert str(document.attributes[0]) == "Attribute(annotation=Hello, label=label)"
    assert str(document.attributes[1]) == "Attribute(annotation=world, label=label)"


def test_document_annotation_fields():
    @dataclasses.dataclass
    class MyDocument(Document):
        text: str
        words: AnnotationLayer[Span] = annotation_field(target="text")

    annotation_fields = MyDocument.annotation_fields()
    annotation_field_names = {field.name for field in annotation_fields}
    assert annotation_field_names == {"words"}


def test_document_target_names():
    @dataclasses.dataclass
    class MyDocument(Document):
        text: str
        words: AnnotationLayer[Span] = annotation_field(target="text")
        sentences: AnnotationLayer[Span] = annotation_field(target="text")
        belongs_to: AnnotationLayer[BinaryRelation] = annotation_field(
            targets=["words", "sentences"]
        )

    # request target names for annotation field
    assert MyDocument.target_names("words") == {"text"}
    assert MyDocument.target_name("words") == "text"

    # requested field is not an annotation field
    with pytest.raises(ValueError) as excinfo:
        MyDocument.target_names("text")
    assert str(excinfo.value) == f"'text' is not an annotation field of {MyDocument.__name__}."

    # requested field has two targets
    assert MyDocument.target_names("belongs_to") == {"words", "sentences"}
    with pytest.raises(ValueError) as excinfo:
        MyDocument.target_name("belongs_to")
    assert (
        str(excinfo.value)
        == "The annotation field 'belongs_to' has more or less than one target, can not return a single target name: ['sentences', 'words']"
    )


def test_document_copy():
    @dataclasses.dataclass
    class MyDocument(Document):
        text: str
        words: AnnotationLayer[Span] = annotation_field(target="text")

    document = MyDocument(text="Hello world!")
    word = Span(start=0, end=5)
    document.words.append(word)
    document_copy = document.copy()
    assert document_copy == document

    # copy without annotations
    document_copy = document.copy(with_annotations=False)
    assert document_copy != document
    annotation_fields = document_copy.annotation_fields()
    assert len(annotation_fields) > 0
    for field in dataclasses.fields(document):
        if field in annotation_fields:
            assert getattr(document_copy, field.name) != getattr(document, field.name)
        else:
            assert getattr(document_copy, field.name) == getattr(document, field.name)


def test_document_target_name_and_target():
    @dataclasses.dataclass
    class MyDocument(Document):
        text: str
        words: AnnotationLayer[Span] = annotation_field(target="text")

    document = MyDocument(text="Hello world!")
    assert document.words.target_name == "text"
    assert document.words.target == document.text == "Hello world!"

    class DoubleSpan(Annotation):
        start1: int
        end1: int
        start2: int
        end2: int

    @dataclasses.dataclass
    class MyDocumentTwoTargets(Document):
        text1: str
        text2: str
        words: AnnotationLayer[DoubleSpan] = annotation_field(targets=["text1", "text2"])

    document = MyDocumentTwoTargets(text1="Hello world!", text2="Hello world again!")
    with pytest.raises(ValueError) as excinfo:
        document.words.target_name
    assert (
        str(excinfo.value)
        == "The annotation layer has more or less than one target, can not return a single target name: "
        "['text1', 'text2']"
    )
    with pytest.raises(ValueError) as excinfo:
        document.words.target
    assert (
        str(excinfo.value)
        == "The annotation layer has more or less than one target, can not return a single target name: "
        "['text1', 'text2']"
    )

    assert document.words.target_names == ["text1", "text2"]
    assert document.words.targets == {"text1": "Hello world!", "text2": "Hello world again!"}


def test_deduplicate_annotations():
    @dataclasses.dataclass
    class MyDocument(Document):
        text: str
        entities: AnnotationLayer[LabeledSpan] = annotation_field(target="text")
        relations: AnnotationLayer[BinaryRelation] = annotation_field(target="entities")

    document = MyDocument(text="Hello world!")
    document.entities.append(LabeledSpan(start=0, end=5, label="entity"))
    document.entities.append(LabeledSpan(start=0, end=5, label="entity", score=0.5))
    document.entities.append(LabeledSpan(start=6, end=11, label="entity"))
    document.entities.append(LabeledSpan(start=6, end=11, label="entity_2"))
    assert document.entities.resolve() == [
        ("entity", "Hello"),
        ("entity", "Hello"),
        ("entity", "world"),
        ("entity_2", "world"),
    ]
    document.entities.predictions.append(LabeledSpan(start=0, end=5, label="entity", score=0.9))

    document.relations.append(
        BinaryRelation(head=document.entities[0], tail=document.entities[2], label="relation")
    )
    document.relations.append(
        BinaryRelation(head=document.entities[0], tail=document.entities[2], label="relation")
    )
    document.relations.append(
        BinaryRelation(head=document.entities[1], tail=document.entities[2], label="relation")
    )
    document.relations.append(
        BinaryRelation(head=document.entities[1], tail=document.entities[3], label="relation")
    )
    assert document.relations.resolve() == [
        ("relation", (("entity", "Hello"), ("entity", "world"))),
        ("relation", (("entity", "Hello"), ("entity", "world"))),
        ("relation", (("entity", "Hello"), ("entity", "world"))),
        ("relation", (("entity", "Hello"), ("entity_2", "world"))),
    ]

    document.relations.predictions.append(
        BinaryRelation(head=document.entities[1], tail=document.entities[3], label="relation")
    )
    document.relations.predictions.append(
        BinaryRelation(
            head=document.entities[1], tail=document.entities[3], label="relation", score=0.5
        )
    )
    document.relations.predictions.append(
        BinaryRelation(
            head=document.entities.predictions[0],
            tail=document.entities[3],
            label="relation",
            score=0.8,
        )
    )

    tp = len(set(document.relations.predictions) & set(document.relations))
    fp = len(set(document.relations.predictions) - set(document.relations))
    fn = len(set(document.relations) - set(document.relations.predictions))
    assert tp == 1
    assert fp == 0
    assert fn == 1

    deduplicated_doc = document.deduplicate_annotations()
    assert len(deduplicated_doc.entities) == 3
    assert {ann.copy() for ann in deduplicated_doc.entities} == {
        LabeledSpan(start=0, end=5, label="entity", score=1.0),
        LabeledSpan(start=6, end=11, label="entity", score=1.0),
        LabeledSpan(start=6, end=11, label="entity_2", score=1.0),
    }
    assert len(deduplicated_doc.entities.predictions) == 1
    assert {ann.copy() for ann in deduplicated_doc.entities.predictions} == {
        LabeledSpan(start=0, end=5, label="entity", score=0.9)
    }

    assert len(deduplicated_doc.relations) == 2
    assert {ann.copy() for ann in deduplicated_doc.relations} == {
        BinaryRelation(
            head=deduplicated_doc.entities[0], tail=deduplicated_doc.entities[1], label="relation"
        ),
        BinaryRelation(
            head=deduplicated_doc.entities[0], tail=deduplicated_doc.entities[2], label="relation"
        ),
    }
    assert len(deduplicated_doc.relations.predictions) == 1
    assert {ann.copy() for ann in deduplicated_doc.relations.predictions} == {
        BinaryRelation(
            head=deduplicated_doc.entities[0], tail=deduplicated_doc.entities[2], label="relation"
        ),
    }

    assert deduplicated_doc.relations.resolve() == [
        ("relation", (("entity", "Hello"), ("entity", "world"))),
        ("relation", (("entity", "Hello"), ("entity_2", "world"))),
    ]
    assert deduplicated_doc.relations.predictions.resolve() == [
        ("relation", (("entity", "Hello"), ("entity_2", "world")))
    ]
    tp = len(set(deduplicated_doc.relations.predictions) & set(deduplicated_doc.relations))
    fp = len(set(deduplicated_doc.relations.predictions) - set(deduplicated_doc.relations))
    fn = len(set(deduplicated_doc.relations) - set(deduplicated_doc.relations.predictions))
    assert tp == 1
    assert fp == 0
    assert fn == 1


def test_text_document():
    document1 = TextBasedDocument(text="text1")
    assert document1.text == "text1"
    assert document1.id is None
    assert document1.metadata == {}

    document1.asdict() == {
        "id": None,
        "text": "text1",
    }

    assert document1 == TextBasedDocument.fromdict(document1.asdict())

    document2 = TextBasedDocument(text="text2", id="test_id", metadata={"key": "value"})
    assert document2.text == "text2"
    assert document2.id == "test_id"
    assert document2.metadata == {"key": "value"}

    document2.asdict() == {
        "id": "test_id",
        "text": "text1",
        "metadata": {
            "key": "value",
        },
    }

    assert document2 == TextBasedDocument.fromdict(document2.asdict())


def test_document_with_annotations():
    @dataclasses.dataclass
    class TestDocument(TextBasedDocument):
        sentences: AnnotationLayer[Span] = annotation_field(target="text")
        entities: AnnotationLayer[LabeledSpan] = annotation_field(target="text")
        relations: AnnotationLayer[BinaryRelation] = annotation_field(target="entities")
        label: AnnotationLayer[Label] = annotation_field()

    document1 = TestDocument(text="text1 and some more text.")
    assert isinstance(document1.sentences, AnnotationLayer)
    assert isinstance(document1.entities, AnnotationLayer)
    assert isinstance(document1.relations, AnnotationLayer)
    assert len(document1.sentences) == 0
    assert len(document1.entities) == 0
    assert len(document1.relations) == 0
    assert len(document1.sentences.predictions) == 0
    assert len(document1.entities.predictions) == 0
    assert len(document1.relations.predictions) == 0
    assert set(document1._annotation_graph.keys()) == {
        "sentences",
        "relations",
        "entities",
        "_artificial_root",
    }
    assert set(document1._annotation_graph["sentences"]) == {"text"}
    assert set(document1._annotation_graph["relations"]) == {"entities"}
    assert set(document1._annotation_graph["entities"]) == {"text"}
    assert set(document1._annotation_graph["_artificial_root"]) == {
        "sentences",
        "relations",
        "label",
    }

    span1 = Span(start=0, end=5)
    span2 = Span(start=6, end=9)

    document1.sentences.append(span1)
    document1.sentences.append(span2)
    assert len(document1.sentences) == 2
    assert document1.sentences[:2] == [span1, span2]
    assert document1.sentences[0].target == document1.text
    resolved_sentences = document1.sentences.resolve()
    assert resolved_sentences == ["text1", "and"]

    labeled_span1 = LabeledSpan(start=0, end=5, label="label1")
    labeled_span2 = LabeledSpan(start=6, end=9, label="label2")
    document1.entities.append(labeled_span1)
    document1.entities.append(labeled_span2)
    assert len(document1.entities) == 2
    assert document1.sentences[0].target == document1.text
    resolved_entities = document1.entities.resolve()
    assert resolved_entities == [("label1", "text1"), ("label2", "and")]

    relation1 = BinaryRelation(head=labeled_span1, tail=labeled_span2, label="label1")
    relation2 = BinaryRelation(head=labeled_span1, tail=labeled_span2, label="label1")
    relation3 = BinaryRelation(head=labeled_span2, tail=labeled_span1, label="label1")
    assert relation1._id == relation2._id
    assert relation1._id != relation3._id

    document1.relations.append(relation1)
    assert len(document1.relations) == 1
    assert document1.relations[0].target == document1.entities
    resolved_relations = document1.relations.resolve()
    assert resolved_relations == [("label1", (("label1", "text1"), ("label2", "and")))]

    assert document1 == TestDocument.fromdict(document1.asdict())

    assert len(document1) == 4
    assert len(document1["sentences"]) == 2
    assert document1["sentences"][0].target == document1.text

    with pytest.raises(
        KeyError, match=re.escape("Document has no attribute 'non_existing_annotation'.")
    ):
        document1["non_existing_annotation"]

    span3 = Span(start=10, end=14)
    span4 = Span(start=15, end=19)

    document1.sentences.predictions.append(span3)
    document1.sentences.predictions.append(span4)
    resolved_sentences_predictions = document1.sentences.predictions.resolve()
    assert resolved_sentences_predictions == ["some", "more"]
    # add a prediction that is also an annotation
    # remove the annotation to allow reassigning it
    relation1_popped = document1.relations.pop(0)
    assert relation1_popped == relation1
    document1.relations.predictions.append(relation1)

    assert len(document1.sentences.predictions) == 2
    assert document1.sentences.predictions[1].target == document1.text
    assert len(document1["sentences"].predictions) == 2
    assert document1["sentences"].predictions[1].target == document1.text

    document1.label.append(Label(label="test_label", score=1.0))

    assert document1 == TestDocument.fromdict(document1.asdict())

    # number of annotation fields
    assert len(document1) == 4
    # actual annotation fields (tests __iter__)
    assert set(document1) == {"sentences", "entities", "relations", "label"}


def test_document_with_same_annotations():
    @dataclasses.dataclass
    class TestDocument(Document):
        text: str
        text2: str
        text3: str
        tokens0: AnnotationLayer[Span] = annotation_field(target="text")
        tokens1: AnnotationLayer[Span] = annotation_field(target="text")
        tokens2: AnnotationLayer[Span] = annotation_field(target="text2")
        tokens3: AnnotationLayer[Span] = annotation_field(target="text3")

    doc = TestDocument(text="test1", text2="test1", text3="test2")
    start = 0
    end = len(doc.text)
    token0 = Span(start=start, end=end)
    token1 = Span(start=start, end=end)
    token2 = Span(start=start, end=end)
    token3 = Span(start=start, end=end)
    token0_id = token0._id
    token1_id = token1._id
    token2_id = token2._id
    token3_id = token3._id
    # all spans are identical, so are there ids
    assert token1_id == token0_id
    assert token2_id == token0_id
    assert token3_id == token0_id
    doc.tokens0.append(token0)
    doc.tokens1.append(token1)
    doc.tokens2.append(token2)
    doc.tokens3.append(token3)

    # test reconstruction
    doc_dict = doc.asdict()
    doc_reconstructed = TestDocument.fromdict(doc_dict)
    assert doc == doc_reconstructed


def test_as_type():
    @dataclasses.dataclass
    class TestDocument1(TextBasedDocument):
        sentences: AnnotationLayer[Span] = annotation_field(target="text")
        entities: AnnotationLayer[LabeledSpan] = annotation_field(target="text")

    @dataclasses.dataclass
    class TestDocument2(TextBasedDocument):
        sentences: AnnotationLayer[Span] = annotation_field(target="text")
        ents: AnnotationLayer[LabeledSpan] = annotation_field(target="text")

    @dataclasses.dataclass
    class TestDocument3(TextBasedDocument):
        entities: AnnotationLayer[LabeledSpan] = annotation_field(target="text")
        relations: AnnotationLayer[BinaryRelation] = annotation_field(target="entities")

    # create input document with "sentences" and "relations"
    document1 = TestDocument1(text="test1")
    span1 = Span(start=1, end=2)
    span2 = Span(start=3, end=4)
    document1.sentences.append(span1)
    document1.sentences.append(span2)
    labeled_span1 = LabeledSpan(start=1, end=2, label="label1")
    labeled_span2 = LabeledSpan(start=3, end=4, label="label2")
    document1.entities.append(labeled_span1)
    document1.entities.append(labeled_span2)

    # convert rename "entities" to "ents"
    document2 = document1.as_type(new_type=TestDocument2, field_mapping={"entities": "ents"})
    assert set(document2) == {"sentences", "ents"}
    assert document2.sentences == document1.sentences
    assert document2.ents == document1.entities

    # remove "sentences", but add "relations"
    document3 = document1.as_type(new_type=TestDocument3)
    assert set(document3) == {"entities", "relations"}
    rel = BinaryRelation(head=span1, tail=span2, label="rel")
    document3.relations.append(rel)
    assert len(document3.relations) == 1


def test_enumerate_dependencies():
    # annotation field -> targets
    graph = {"a": ["b"], "b": ["c"], "d": ["c", "a"], "e": ["f"], "g": ["e"], "h": ["e"]}
    root_nodes = ["d", "g", "h"]
    resolved = []
    _enumerate_dependencies(resolved=resolved, dependency_graph=graph, nodes=root_nodes)

    for i, node in enumerate(resolved):
        already_resolved = resolved[:i]
        targets = graph.get(node, [])
        for t in targets:
            assert t in already_resolved


def test_enumerate_dependencies_with_circle():
    graph = {"a": ["b"], "b": ["c"], "c": ["b"], "d": ["e"]}
    root_nodes = ["a", "d"]
    resolved = []
    with pytest.raises(ValueError, match=re.escape("circular dependency detected at node: b")):
        _enumerate_dependencies(resolved=resolved, dependency_graph=graph, nodes=root_nodes)


def test_annotation_list_wrong_target():
    @dataclasses.dataclass
    class TestDocument(TextBasedDocument):
        entities: AnnotationLayer[LabeledSpan] = annotation_field(target="does_not_exist")

    with pytest.raises(
        TypeError,
        match=re.escape(
            'annotation target "does_not_exist" is not in field names of the document: '
        ),
    ):
        document = TestDocument(text="text")


def test_annotation_list():
    @dataclasses.dataclass
    class TestDocument(TextBasedDocument):
        entities: AnnotationLayer[LabeledSpan] = annotation_field(target="text")

    document = TestDocument(text="Entity A works at B.")

    entity1 = LabeledSpan(start=0, end=8, label="PER")
    entity2 = LabeledSpan(start=18, end=19, label="ORG")
    assert not entity1.is_attached
    assert not entity2.is_attached

    document.entities.append(entity1)
    document.entities.append(entity2)

    entity3 = LabeledSpan(start=18, end=19, label="PRED-ORG")
    entity4 = LabeledSpan(start=0, end=8, label="PRED-PER")
    assert not entity3.is_attached
    assert not entity4.is_attached

    document.entities.predictions.append(entity3)
    document.entities.predictions.append(entity4)

    assert isinstance(document.entities, AnnotationLayer)
    assert len(document.entities) == 2
    assert document.entities[0] == entity1
    assert document.entities[1] == entity2
    assert document.entities[0].target == document.text
    assert document.entities[1].target == document.text
    assert entity1.target == document.text
    assert entity2.target == document.text
    assert str(document.entities[0]) == "Entity A"
    assert str(document.entities[1]) == "B"

    assert len(document.entities.predictions) == 2
    assert document.entities.predictions[0] == entity3
    assert document.entities.predictions[1] == entity4
    assert document.entities.predictions[0].target == document.text
    assert document.entities.predictions[1].target == document.text
    assert entity3.target == document.text
    assert entity4.target == document.text
    assert str(document.entities.predictions[0]) == "B"
    assert str(document.entities.predictions[1]) == "Entity A"

    document.entities.clear()
    assert len(document.entities) == 0
    assert not entity1.is_attached
    assert not entity2.is_attached

    document.entities.predictions.clear()
    assert len(document.entities.predictions) == 0
    assert not entity3.is_attached
    assert not entity4.is_attached


def test_annotation_list_with_multiple_targets():
    @dataclasses.dataclass
    class TestDocument(TextBasedDocument):
        entities1: AnnotationLayer[LabeledSpan] = annotation_field(target="text")
        entities2: AnnotationLayer[LabeledSpan] = annotation_field(target="text")
        relations: AnnotationLayer[BinaryRelation] = annotation_field(
            targets=["entities1", "entities2"]
        )
        label: AnnotationLayer[Label] = annotation_field()

    doc = TestDocument(text="test1")

    assert set(doc._annotation_graph.keys()) == {
        "entities1",
        "entities2",
        "relations",
        "_artificial_root",
    }
    assert set(doc._annotation_graph["entities1"]) == {"text"}
    assert set(doc._annotation_graph["entities2"]) == {"text"}
    assert set(doc._annotation_graph["relations"]) == {"entities1", "entities2"}
    assert set(doc._annotation_graph["_artificial_root"]) == {
        "relations",
        "label",
    }

    span1 = LabeledSpan(0, 2, label="a")
    assert not span1.is_attached
    doc.entities1.append(span1)
    assert doc.entities1[0] == span1
    assert span1.target == doc.text

    span2 = LabeledSpan(2, 4, label="b")
    assert not span2.is_attached
    doc.entities2.append(span2)
    assert doc.entities2[0] == span2
    assert span2.target == doc.text

    relation = BinaryRelation(head=span1, tail=span2, label="relation")
    assert not relation.is_attached
    doc.relations.append(relation)
    assert doc.relations[0] == relation
    with pytest.raises(
        ValueError,
        match=re.escape("annotation has multiple targets, target is not defined in this case"),
    ):
        relation.target
    assert relation.targets == (doc.entities1, doc.entities2)

    label = Label("label")
    assert not label.is_attached
    doc.label.append(label)
    assert doc.label[0] == label
    with pytest.raises(ValueError, match=re.escape("annotation has no target")):
        label.target
    assert label.targets == ()


@dataclasses.dataclass(eq=True, frozen=True)
class DoubleTextSpan(Annotation):
    TARGET_NAMES = (
        "text1",
        "text2",
    )
    start1: int
    end1: int
    start2: int
    end2: int

    def __str__(self) -> str:
        if not self.is_attached:
            return ""
        text1: str = self.named_targets["text1"]  # type: ignore
        text2: str = self.named_targets["text2"]  # type: ignore
        return str(text1[self.start1 : self.end1]) + "|" + str(text2[self.start2 : self.end2])


def test_annotation_list_with_named_targets():
    @dataclasses.dataclass
    class TestDocument(Document):
        texta: str
        textb: str
        entities1: AnnotationLayer[LabeledSpan] = annotation_field(target="texta")
        entities2: AnnotationLayer[LabeledSpan] = annotation_field(target="textb")
        # note that the entries in targets do not follow the order of DoubleTextSpan.TARGET_NAMES
        crossrefs: AnnotationLayer[DoubleTextSpan] = annotation_field(
            named_targets={"text2": "textb", "text1": "texta"}
        )

    doc = TestDocument(texta="text1", textb="text2")

    assert set(doc._annotation_graph.keys()) == {
        "entities1",
        "entities2",
        "crossrefs",
        "_artificial_root",
    }
    assert set(doc._annotation_graph["entities1"]) == {"texta"}
    assert set(doc._annotation_graph["entities2"]) == {"textb"}
    assert set(doc._annotation_graph["crossrefs"]) == {"texta", "textb"}
    assert set(doc._annotation_graph["_artificial_root"]) == {
        "entities1",
        "entities2",
        "crossrefs",
    }

    span1 = LabeledSpan(0, 2, label="a")
    assert not span1.is_attached
    doc.entities1.append(span1)
    assert doc.entities1[0] == span1
    assert span1.target == doc.texta

    span2 = LabeledSpan(2, 4, label="b")
    assert not span2.is_attached
    doc.entities2.append(span2)
    assert doc.entities2[0] == span2
    assert span2.target == doc.textb

    doublespan = DoubleTextSpan(0, 2, 1, 5)
    assert not doublespan.is_attached
    doc.crossrefs.append(doublespan)
    assert doc.crossrefs[0] == doublespan
    assert doublespan.named_targets == {"text1": doc.texta, "text2": doc.textb}
    assert str(doublespan) == "te|ext2"  # codespell:ignore te


def test_annotation_list_with_named_targets_mismatch_error():
    @dataclasses.dataclass(eq=True, frozen=True)
    class TextSpan(Annotation):
        TARGET_NAMES = ("text",)
        start: int
        end: int

        def __str__(self) -> str:
            if not self.is_attached:
                return ""
            text: str = self.named_targets["text"]  # type: ignore
            return str(text[self.start : self.end])

    @dataclasses.dataclass
    class TestDocument(Document):
        text: str
        entities1: AnnotationLayer[TextSpan] = annotation_field(named_targets={"textx": "text"})

    with pytest.raises(
        TypeError,
        match=re.escape("keys of targets ['textx'] do not match TextSpan.TARGET_NAMES ['text']"),
    ):
        doc = TestDocument(text="text1")


def test_annotation_list_with_missing_target_names():
    @dataclasses.dataclass
    class TestDocument(Document):
        texta: str
        textb: str
        # note that the entries in targets do not follow the order of DoubleTextSpan.TARGET_NAMES
        crossrefs: AnnotationLayer[DoubleTextSpan] = annotation_field(targets=["textb", "texta"])

    with pytest.raises(TypeError) as excinfo:
        doc = TestDocument(texta="text1", textb="text2")
    assert str(excinfo.value) == (
        "A target name mapping is required for AnnotationLayers containing Annotations with TARGET_NAMES, but "
        'AnnotationLayer "crossrefs" has no target_names. You should pass the named_targets dict containing the '
        "following keys (see Annotation \"DoubleTextSpan\") to annotation_field: ('text1', 'text2')"
    )


def test_annotation_list_number_of_targets_mismatch_error():
    @dataclasses.dataclass
    class TestDocument(Document):
        texta: str
        textb: str
        crossrefs: AnnotationLayer[DoubleTextSpan] = annotation_field(target="texta")

    with pytest.raises(
        TypeError,
        match=re.escape(
            "number of targets ['texta'] does not match number of entries in DoubleTextSpan.TARGET_NAMES: "
            "['text1', 'text2']"
        ),
    ):
        doc = TestDocument(texta="text1", textb="text2")


def test_annotation_list_artificial_root_error():
    @dataclasses.dataclass
    class TestDocument(Document):
        text: str
        _artificial_root: AnnotationLayer[LabeledSpan] = annotation_field(target="text")

    with pytest.raises(
        ValueError,
        match=re.escape(
            'Failed to add the "_artificial_root" node to the annotation graph because it already exists. Note '
            "that AnnotationLayer entries with that name are not allowed."
        ),
    ):
        doc = TestDocument(text="text1")


def test_annotation_list_targets():
    @dataclasses.dataclass
    class TestDocument(Document):
        text: str
        entities1: AnnotationLayer[LabeledSpan] = annotation_field(target="text")
        entities2: AnnotationLayer[LabeledSpan] = annotation_field(target="text")
        relations1: AnnotationLayer[BinaryRelation] = annotation_field(target="entities1")
        relations2: AnnotationLayer[BinaryRelation] = annotation_field(
            targets=["entities1", "entities2"]
        )

    doc = TestDocument(text="text1")

    # test getting all targets
    assert doc.entities1.targets == {"text": doc.text}
    assert doc.entities2.targets == {"text": doc.text}
    assert doc.relations1.targets == {"entities1": doc.entities1}
    assert doc.relations2.targets == {"entities1": doc.entities1, "entities2": doc.entities2}

    # test getting a single target
    assert doc.entities1.target == doc.text
    assert doc.entities2.target == doc.text
    assert doc.relations1.target == doc.entities1
    # check that the target of relations2 is not set because it has more than one target
    with pytest.raises(ValueError) as excinfo:
        doc.relations2.target
    assert (
        str(excinfo.value)
        == "The annotation layer has more or less than one target, can not return a single target name: "
        "['entities1', 'entities2']"
    )

    # test getting all target layers
    assert doc.entities1.target_layers == {}
    assert doc.entities2.target_layers == {}
    assert doc.relations1.target_layers == {"entities1": doc.entities1}
    assert doc.relations2.target_layers == {"entities1": doc.entities1, "entities2": doc.entities2}

    # test getting a single target layer
    with pytest.raises(ValueError) as excinfo:
        doc.entities1.target_layer
    assert str(excinfo.value) == "The annotation layer has more or less than one target layer: []"
    with pytest.raises(ValueError) as excinfo:
        doc.entities2.target_layer
    assert str(excinfo.value) == "The annotation layer has more or less than one target layer: []"
    assert doc.relations1.target_layer == doc.entities1
    with pytest.raises(ValueError) as excinfo:
        doc.relations2.target_layer
    assert (
        str(excinfo.value)
        == "The annotation layer has more or less than one target layer: ['entities1', 'entities2']"
    )


def test_annotation_compare():
    @dataclasses.dataclass(eq=True, frozen=True)
    class TestAnnotation(Annotation):
        value: str
        # Note that the score field is marked as not comparable
        score: Optional[float] = dataclasses.field(compare=False, default=None)

    annotation0 = TestAnnotation(value="test")
    annotation1 = TestAnnotation(value="test", score=0.9)
    annotation2 = TestAnnotation(value="test", score=0.5)

    assert hash(annotation0) == hash(annotation1) == hash(annotation2)
    assert annotation0 == annotation1 and annotation0 == annotation2

    # annotation id is equal if the annotation is the same
    assert annotation0._id == TestAnnotation(value="test")._id
    assert annotation1._id == TestAnnotation(value="test", score=0.9)._id
    assert annotation2._id == TestAnnotation(value="test", score=0.5)._id
    # annotation id is different if the annotation is different (just in non-comparable fields)
    assert annotation0._id != annotation1._id and annotation0._id != annotation2._id

    # assert that nothing changes when adding the annotation to a document
    @dataclasses.dataclass
    class TestDocument(TextBasedDocument):
        annotations: AnnotationLayer[TestAnnotation] = annotation_field(target="text")

    id0 = annotation0._id
    hash0 = hash(annotation0)
    doc = TestDocument(text="test")
    doc.annotations.append(annotation0)
    assert annotation0._id == id0
    assert hash(annotation0) == hash0

    # The score field of built-in Annotations is marked with compare=False, so it is not taken
    # into account when comparing annotations ...
    assert LabeledSpan(0, 1, "test") == LabeledSpan(0, 1, "test", score=0.9)
    # ... but, again, the id is different.
    assert LabeledSpan(0, 1, "test")._id != LabeledSpan(0, 1, "test", score=0.9)._id

    # works also for nested annotations
    e1 = LabeledSpan(0, 1, "test")
    e2 = LabeledSpan(0, 1, "test", score=0.9)
    e3 = LabeledSpan(3, 4, "test", score=0.5)
    assert e1 == e2
    r1 = BinaryRelation(e1, e3, "test")
    r2 = BinaryRelation(e2, e3, "test")
    assert r1 == r2


@dataclasses.dataclass(frozen=True)
class Attribute(Annotation):
    ref: Annotation
    value: str


@pytest.fixture
def text_document():
    @dataclasses.dataclass
    class TextBasedDocumentWithEntitiesRelationsAndRelationAttributes(TextBasedDocument):
        entities1: AnnotationLayer[LabeledSpan] = annotation_field(target="text")
        entities2: AnnotationLayer[LabeledSpan] = annotation_field(target="text")
        relations: AnnotationLayer[BinaryRelation] = annotation_field(
            targets=["entities1", "entities2"]
        )
        labels: AnnotationLayer[Label] = annotation_field()
        relation_attributes: AnnotationLayer[Attribute] = annotation_field(target="relations")

    doc1 = TextBasedDocumentWithEntitiesRelationsAndRelationAttributes(text="Hello World!")
    e1 = LabeledSpan(0, 5, "word1")
    e2 = LabeledSpan(6, 11, "word2")
    doc1.entities1.append(e1)
    doc1.entities2.append(e2)
    r1 = BinaryRelation(e1, e2, "relation1")
    doc1.relations.append(r1)
    doc1.relation_attributes.append(Attribute(r1, "value1"))
    doc1.labels.append(Label("label3"))
    return doc1


def test_document_extend_from_other_full_copy(text_document):
    doc_new = type(text_document)(text=text_document.text)
    added_annotations = doc_new.add_all_annotations_from_other(text_document)

    assert text_document.asdict() == doc_new.asdict()
    assert set(added_annotations) == {
        "entities1",
        "entities2",
        "relations",
        "relation_attributes",
        "labels",
    }
    for layer_name, annotation_mapping in added_annotations.items():
        assert len(annotation_mapping) > 0
        available_annotations = text_document[layer_name]
        available_annotation_ids = [a._id for a in available_annotations]
        assert set(annotation_mapping) == set(available_annotation_ids)
        assert len(annotation_mapping) == 1
        # since we have only one annotation, we can construct the expected mapping
        assert annotation_mapping == {available_annotation_ids[0]: doc_new[layer_name][0]}


def test_document_extend_from_other_wrong_override_annotation_mapping(text_document):
    new_doc = type(text_document)(text="Hello World!")
    with pytest.raises(ValueError) as excinfo:
        new_doc.add_all_annotations_from_other(text_document, override_annotations={"text": {}})
    assert (
        str(excinfo.value)
        == 'Field "text" is not an annotation field of TextBasedDocumentWithEntitiesRelationsAndRelationAttributes, '
        "but keys in override_annotation_mapping must be annotation field names."
    )


def test_document_extend_from_other_override(text_document):
    @dataclasses.dataclass
    class TestDocument2(TokenBasedDocument):
        entities1: AnnotationLayer[LabeledSpan] = annotation_field(target="tokens")
        entities2: AnnotationLayer[LabeledSpan] = annotation_field(target="tokens")
        relations: AnnotationLayer[BinaryRelation] = annotation_field(
            targets=["entities1", "entities2"]
        )
        labels: AnnotationLayer[Label] = annotation_field()
        relation_attributes: AnnotationLayer[Attribute] = annotation_field(target="relations")

    token_document = TestDocument2(tokens=("Hello", "World", "!"))
    # create new entities
    e1_new = LabeledSpan(0, 1, "word1")
    e2_new = LabeledSpan(1, 2, "word2")
    # create annotation mapping
    e1 = text_document.entities1[0]
    e2 = text_document.entities2[0]
    annotation_mapping = {"entities1": {e1._id: e1_new}, "entities2": {e2._id: e2_new}}
    # add new entities ...
    token_document.entities1.append(e1_new)
    token_document.entities2.append(e2_new)
    # ... and the remaining annotations
    added_annotations = token_document.add_all_annotations_from_other(
        text_document, override_annotations=annotation_mapping
    )
    added_annotation_sets = {k: set(v) for k, v in added_annotations.items()}
    # check that the added annotations are as expected (the entity annotations are already there)
    assert added_annotation_sets == {
        "relations": {ann._id for ann in text_document.relations},
        "relation_attributes": {ann._id for ann in text_document.relation_attributes},
        "labels": {ann._id for ann in text_document.labels},
    }
    for layer_name, annotation_mapping in added_annotations.items():
        text_annotations = text_document[layer_name]
        token_annotations = token_document[layer_name]
        assert len(annotation_mapping) == len(text_annotations) == len(token_annotations) == 1
        # since we have only one annotation, we can construct the expected mapping
        assert annotation_mapping == {text_annotations[0]._id: token_annotations[0]}

    assert (
        len(token_document.entities1)
        == len(token_document.entities2)
        == len(token_document.relations)
        == len(token_document.labels)
        == len(token_document.relation_attributes)
        == 1
    )
    assert (
        str(token_document.entities1[0]) == str(token_document.relations[0].head) == "('Hello',)"
    )
    assert (
        str(token_document.entities2[0]) == str(token_document.relations[0].tail) == "('World',)"
    )
    assert token_document.labels[0] == text_document.labels[0]
    assert token_document.relation_attributes[0].ref == token_document.relations[0]
    assert (
        token_document.relation_attributes[0].value == text_document.relation_attributes[0].value
    )


def test_document_extend_from_other_remove(text_document):
    doc_new = type(text_document)(text=text_document.text)
    added_annotations = doc_new.add_all_annotations_from_other(
        text_document,
        removed_annotations={"entities1": {text_document.entities1[0]._id}},
        strict=False,
    )
    added_annotation_sets = {k: set(v) for k, v in added_annotations.items()}
    # the only entity in entities1 is removed and since the relation has it as head, the relation is removed as well
    assert added_annotation_sets == {
        "entities2": {ann._id for ann in text_document.entities2},
        "labels": {ann._id for ann in text_document.labels},
    }
    assert added_annotations == {
        "entities2": {text_document.entities2[0]._id: doc_new.entities2[0]},
        "labels": {text_document.labels[0]._id: doc_new.labels[0]},
    }

    assert len(doc_new.entities1) == 0
    assert len(doc_new.entities2) == 1
    assert len(doc_new.relations) == 0
    assert len(doc_new.labels) == 1
    assert len(doc_new.relation_attributes) == 0


def test_document_field_types():
    @dataclasses.dataclass
    class MyDocument(Document):
        text: str
        words: AnnotationLayer[Span] = annotation_field(target="text")

    annotation_fields = MyDocument.annotation_fields()
    annotation_field_names = {field.name for field in annotation_fields}
    assert annotation_field_names == {"words"}

    field_types = MyDocument.field_types()
    assert field_types == {"text": str, "words": AnnotationLayer[Span]}

    # this requires to resolve the field types with typing.get_type_hints() because field.type is
    # a string at this point (externally defined document class)
    # field_types = TextDocumentWithSpansBinaryRelationsAndLabeledPartitions.field_types()
    # assert field_types == {
    #    "_annotation_fields": Set[str],
    #    "_annotation_graph": Dict[str, List[str]],
    #    "binary_relations": AnnotationLayer[BinaryRelation],
    #    "id": Optional[str],
    #    "labeled_partitions": AnnotationLayer[LabeledSpan],
    #    "metadata": Dict[str, Any],
    #    "spans": AnnotationLayer[Span],
    #    "text": str,
    # }


def test_annotation_types():
    @dataclasses.dataclass
    class MyDocument(Document):
        text: str
        words: AnnotationLayer[Span] = annotation_field(target="text")

    annotation_types = MyDocument.annotation_types()
    assert annotation_types == {"words": Span}

    # annotation_types = TextDocumentWithSpansBinaryRelationsAndLabeledPartitions.annotation_types()
    # assert annotation_types == {
    #    "spans": Span,
    #    "labeled_partitions": LabeledSpan,
    #    "binary_relations": BinaryRelation,
    # }
