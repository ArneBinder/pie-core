from collections.abc import Iterator, Sequence
from typing import (
    Any,
    Dict,
    Generic,
    Optional,
    TypeVar,
    Union,
    overload,
)

import torch.utils.data.dataset as torch_dataset

from pie_core.document import Document

DocumentType = TypeVar("DocumentType", bound=Document)
InputEncoding = TypeVar("InputEncoding")
TargetEncoding = TypeVar("TargetEncoding")
# TaskEncoding: defined below
TaskBatchEncoding = TypeVar("TaskBatchEncoding")
# ModelBatchEncoding: defined in models
ModelBatchOutput = TypeVar("ModelBatchOutput")
TaskOutput = TypeVar("TaskOutput")

Metadata = Dict[str, Any]


class TaskEncoding(Generic[DocumentType, InputEncoding, TargetEncoding]):
    def __init__(
        self,
        inputs: InputEncoding,
        targets: Optional[TargetEncoding] = None,
        document: Optional[DocumentType] = None,
        metadata: Optional[Metadata] = None,
    ) -> None:
        self._document = document
        self.inputs = inputs
        self._targets = targets
        self.metadata = metadata or {}

    @property
    def has_targets(self) -> bool:
        return self._targets is not None

    @property
    def targets(self) -> TargetEncoding:
        if self._targets is None:
            raise ValueError("task encoding has no targets.")
        return self._targets

    @targets.setter
    def targets(self, value) -> None:
        self._targets = value

    @property
    def has_document(self) -> bool:
        return self._document is not None

    @property
    def document(self) -> DocumentType:
        if self._document is None:
            raise ValueError("task encoding has no document.")
        return self._document


TaskEncodingType = TypeVar("TaskEncodingType", bound=TaskEncoding)


class TaskEncodingDataset(torch_dataset.Dataset[TaskEncodingType]):
    def __init__(self, encodings: Sequence[TaskEncodingType]):
        self._encodings = encodings

    @overload
    def __getitem__(self, index: int) -> TaskEncodingType: ...

    @overload
    def __getitem__(self, s: slice) -> Sequence[TaskEncodingType]: ...

    def __getitem__(
        self, index: Union[int, slice]
    ) -> Union[TaskEncodingType, Sequence[TaskEncodingType]]:
        return self._encodings[index]

    def __len__(self):
        return len(self._encodings)


class IterableTaskEncodingDataset(torch_dataset.IterableDataset[TaskEncodingType]):
    def __iter__(self) -> Iterator[TaskEncodingType]:
        yield from self._encodings

    def __init__(self, encodings: Iterator[TaskEncodingType]):
        self._encodings = encodings


class TaskEncodingSequence(Sequence[TaskEncodingType], Generic[TaskEncodingType, DocumentType]):
    def __init__(
        self,
        task_encodings: Sequence[TaskEncodingType],
        documents_in_order: Sequence[DocumentType],
    ):
        self.task_encodings = task_encodings
        self.documents_in_order = documents_in_order

    @overload
    def __getitem__(self, index: int) -> TaskEncodingType: ...

    @overload
    def __getitem__(self, s: slice) -> Sequence[TaskEncodingType]: ...

    def __getitem__(
        self, index: Union[int, slice]
    ) -> Union[TaskEncodingType, Sequence[TaskEncodingType]]:
        return self.task_encodings[index]

    def __len__(self) -> int:
        return len(self.task_encodings)
