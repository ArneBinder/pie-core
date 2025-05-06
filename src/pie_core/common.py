from typing import TypeVar

from pie_core.hf_hub_mixin import PieBaseHFHubMixin
from pie_core.registrable import Registrable

T = TypeVar("T", bound=Registrable)


class RegistrableBaseHFHubMixin(Registrable[T], PieBaseHFHubMixin):
    pass
