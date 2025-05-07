from pathlib import Path
from typing import Dict, Optional, Type, Union

import pytest

from pie_core.hf_hub_mixin import PieBaseHFHubMixin, T


class HFHubObject(PieBaseHFHubMixin):
    config_name = "test_config.json"
    config_type_key = "test_type"

    def _save_pretrained(self, save_directory: Path) -> None:
        pass

    def _from_pretrained(
        cls: Type[T],
        *,
        model_id: str,
        revision: Optional[str],
        cache_dir: Optional[Union[str, Path]],
        force_download: bool,
        proxies: Optional[Dict],
        resume_download: bool,
        local_files_only: bool,
        token: Optional[Union[str, bool]],
        **model_kwargs,
    ) -> T:
        pass


@pytest.fixture(scope="module")
def hf_hub_object() -> HFHubObject:
    return HFHubObject()


def test_is_from_pretrained(hf_hub_object):
    assert not hf_hub_object.is_from_pretrained
