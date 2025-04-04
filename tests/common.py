import dataclasses
from typing import Any, Dict, Optional

from pie_core import Annotation, AnnotationLayer, Document, annotation_field


@dataclasses.dataclass(eq=True, frozen=True)
class BinaryRelation(Annotation):
    head: Annotation
    tail: Annotation
    label: str
    score: float = dataclasses.field(default=1.0, compare=False)

    def resolve(self) -> Any:
        return self.label, (self.head.resolve(), self.tail.resolve())


@dataclasses.dataclass(eq=True, frozen=True)
class Span(Annotation):
    start: int
    end: int

    def __str__(self) -> str:
        if not self.is_attached:
            return super().__str__()
        return str(self.target[self.start : self.end])

    def resolve(self) -> Any:
        if self.is_attached:
            return self.target[self.start : self.end]
        else:
            raise ValueError(f"{self} is not attached to a target.")


@dataclasses.dataclass(eq=True, frozen=True)
class LabeledSpan(Span):
    label: str
    score: float = dataclasses.field(default=1.0, compare=False)

    def resolve(self) -> Any:
        return self.label, super().resolve()


@dataclasses.dataclass(eq=True, frozen=True)
class Label(Annotation):
    label: str
    score: float = dataclasses.field(default=1.0, compare=False)

    def resolve(self) -> Any:
        return self.label


@dataclasses.dataclass
class TextBasedDocument(Document):
    text: str
    id: Optional[str] = None
    metadata: Dict[str, Any] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class TokenBasedDocument(Document):
    tokens: tuple[str, ...]
    id: Optional[str] = None
    metadata: Dict[str, Any] = dataclasses.field(default_factory=dict)

    def __post_init__(self) -> None:
        # When used in a dataset, the document gets serialized to json like structure which does not know tuples,
        # so they get converted to lists. This is a workaround to automatically convert the "tokens" back to tuples
        # when the document is created from a dataset.
        if isinstance(self.tokens, list):
            object.__setattr__(self, "tokens", tuple(self.tokens))
        elif not isinstance(self.tokens, tuple):
            raise ValueError("tokens must be a tuple.")

        # Call the default document construction code
        super().__post_init__()


@dataclasses.dataclass
class TestDocument(TextBasedDocument):
    sentences: AnnotationLayer[Span] = annotation_field(target="text")
    entities: AnnotationLayer[LabeledSpan] = annotation_field(target="text")
    relations: AnnotationLayer[BinaryRelation] = annotation_field(target="entities")


@dataclasses.dataclass
class TestDocumentWithEntities(TextBasedDocument):
    entities: AnnotationLayer[LabeledSpan] = annotation_field(target="text")


@dataclasses.dataclass
class TestDocumentWithSentences(TextBasedDocument):
    sentences: AnnotationLayer[Span] = annotation_field(target="text")
