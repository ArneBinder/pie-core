from typing import Any, Dict, List, Optional, Type, Union

import pytest
from pytorch_ie import DatasetDict
from transformers import AutoTokenizer, PreTrainedTokenizer

from pie_core import Document, DocumentStatistic


class TokenCountCollector(DocumentStatistic):

    def __init__(
        self,
        tokenizer: Union[str, PreTrainedTokenizer],
        text_field: str,
        tokenizer_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.tokenizer = (
            AutoTokenizer.from_pretrained(tokenizer) if isinstance(tokenizer, str) else tokenizer
        )
        self.tokenizer_kwargs = tokenizer_kwargs or {}
        self.text_field = text_field

    def _collect(self, doc: Document) -> int:
        text = getattr(doc, self.text_field)
        encodings = self.tokenizer(text, **self.tokenizer_kwargs)
        tokens = encodings.tokens()
        return len(tokens)


dataset = DatasetDict.load_dataset("pie/conll2003")
statistic = TokenCountCollector(
    text_field="text",
    tokenizer="bert-base-uncased",
    tokenizer_kwargs=dict(add_special_tokens=False),
)
values = statistic(dataset)
assert values == {
    "train": {"mean": 17.950502100989958, "std": 13.016237876955675, "min": 1, "max": 162},
    "validation": {"mean": 19.368307692307692, "std": 14.583363922289669, "min": 1, "max": 144},
    "test": {"mean": 16.774978279756734, "std": 13.176981022988947, "min": 1, "max": 138},
}
