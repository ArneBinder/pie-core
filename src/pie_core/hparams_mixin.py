# Copyright © 2019-2025 The Lightning AI team
# Modifications Copyright © 2025 Arne Binder
#
# The original work lives at:
#     https://github.com/Lightning-AI/pytorch-lightning/blob/2.5.1/src/lightning/pytorch/core/mixins/hparams_mixin.py
#     https://github.com/Lightning-AI/pytorch-lightning/blob/2.5.1/src/lightning/fabric/utilities/data.py
#
# NOTICE — the content has been modified from the original Lightning version:
#     HyperparametersMixin is renamed to PieHyperparametersMixin to avoid name clashes.
#     Unused methods and imports have been removed.
#     All subsequent changes are documented in the project’s Git history.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import copy
import inspect
import types
from argparse import Namespace
from collections.abc import Iterator, MutableMapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Optional, Union

from pie_core.utils.hparams import save_hyperparameters


class AttributeDict(dict):
    """A container to store state variables of your program.

    This is a drop-in replacement for a Python dictionary, with the additional functionality to access and modify keys
    through attribute lookup for convenience.

    Use this to define the state of your program, then pass it to
    :meth:`~lightning_fabric.fabric.Fabric.save` and :meth:`~lightning_fabric.fabric.Fabric.load`.

    Example:
        >>> import torch
        >>> model = torch.nn.Linear(2, 2)
        >>> state = AttributeDict(model=model, iter_num=0)
        >>> state.model
        Linear(in_features=2, out_features=2, bias=True)
        >>> state.iter_num += 1
        >>> state.iter_num
        1
        >>> state
        "iter_num": 1
        "model":    Linear(in_features=2, out_features=2, bias=True)
    """

    def __getattr__(self, key: str) -> Any:
        try:
            return self[key]
        except KeyError as e:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{key}'") from e

    def __setattr__(self, key: str, val: Any) -> None:
        self[key] = val

    def __delattr__(self, item: str) -> None:
        if item not in self:
            raise KeyError(item)
        del self[item]

    def __repr__(self) -> str:
        if not len(self):
            return ""
        max_key_length = max(len(str(k)) for k in self)
        tmp_name = "{:" + str(max_key_length + 3) + "s} {}"
        rows = [tmp_name.format(f'"{n}":', self[n]) for n in sorted(self.keys())]
        return "\n".join(rows)


_PRIMITIVE_TYPES = (bool, int, float, str)
_ALLOWED_CONFIG_TYPES = (AttributeDict, MutableMapping, Namespace)
_given_hyperparameters: ContextVar = ContextVar("_given_hyperparameters", default=None)


class PieHyperparametersMixin:
    __jit_unused_properties__: list[str] = ["hparams", "hparams_initial"]

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._log_hyperparams = False

    def save_hyperparameters(
        self,
        *args: Any,
        ignore: Optional[Union[Sequence[str], str]] = None,
        frame: Optional[types.FrameType] = None,
        logger: bool = True,
    ) -> None:
        """Save arguments to ``hparams`` attribute.

        Args:
            args: single object of `dict`, `NameSpace` or `OmegaConf`
                or string names or arguments from class ``__init__``
            ignore: an argument name or a list of argument names from
                class ``__init__`` to be ignored
            frame: a frame object. Default is None
            logger: Whether to send the hyperparameters to the logger. Default: True

        Example::
            >>> from pie_core.hparams_mixin import PieHyperparametersMixin
            >>> class ManuallyArgsModel(PieHyperparametersMixin):
            ...     def __init__(self, arg1, arg2, arg3):
            ...         super().__init__()
            ...         # manually assign arguments
            ...         self.save_hyperparameters('arg1', 'arg3')
            ...     def forward(self, *args, **kwargs):
            ...         ...
            >>> model = ManuallyArgsModel(1, 'abc', 3.14)
            >>> model.hparams
            "arg1": 1
            "arg3": 3.14

            >>> from pie_core.hparams_mixin import PieHyperparametersMixin
            >>> class AutomaticArgsModel(PieHyperparametersMixin):
            ...     def __init__(self, arg1, arg2, arg3):
            ...         super().__init__()
            ...         # equivalent automatic
            ...         self.save_hyperparameters()
            ...     def forward(self, *args, **kwargs):
            ...         ...
            >>> model = AutomaticArgsModel(1, 'abc', 3.14)
            >>> model.hparams
            "arg1": 1
            "arg2": abc
            "arg3": 3.14

            >>> from pie_core.hparams_mixin import PieHyperparametersMixin
            >>> class SingleArgModel(PieHyperparametersMixin):
            ...     def __init__(self, params):
            ...         super().__init__()
            ...         # manually assign single argument
            ...         self.save_hyperparameters(params)
            ...     def forward(self, *args, **kwargs):
            ...         ...
            >>> model = SingleArgModel(Namespace(p1=1, p2='abc', p3=3.14))
            >>> model.hparams
            "p1": 1
            "p2": abc
            "p3": 3.14

            >>> from pie_core.hparams_mixin import PieHyperparametersMixin
            >>> class ManuallyArgsModel(PieHyperparametersMixin):
            ...     def __init__(self, arg1, arg2, arg3):
            ...         super().__init__()
            ...         # pass argument(s) to ignore as a string or in a list
            ...         self.save_hyperparameters(ignore='arg2')
            ...     def forward(self, *args, **kwargs):
            ...         ...
            >>> model = ManuallyArgsModel(1, 'abc', 3.14)
            >>> model.hparams
            "arg1": 1
            "arg3": 3.14
        """
        self._log_hyperparams = logger
        given_hparams = _given_hyperparameters.get()
        # the frame needs to be created in this file.
        if given_hparams is None and not frame:
            current_frame = inspect.currentframe()
            if current_frame:
                frame = current_frame.f_back
        save_hyperparameters(self, *args, ignore=ignore, frame=frame, given_hparams=given_hparams)

    def _set_hparams(self, hp: Union[MutableMapping, Namespace, str]) -> None:
        hp = self._to_hparams_dict(hp)

        if isinstance(hp, dict) and isinstance(self.hparams, dict):
            self.hparams.update(hp)
        else:
            self._hparams = hp

    @staticmethod
    def _to_hparams_dict(
        hp: Union[MutableMapping, Namespace, str],
    ) -> Union[MutableMapping, AttributeDict]:
        if isinstance(hp, Namespace):
            hp = vars(hp)
        if isinstance(hp, dict):
            hp = AttributeDict(hp)
        elif isinstance(hp, _PRIMITIVE_TYPES):
            raise ValueError(f"Primitives {_PRIMITIVE_TYPES} are not allowed.")
        elif not isinstance(hp, _ALLOWED_CONFIG_TYPES):
            raise ValueError(f"Unsupported config type of {type(hp)}.")
        return hp

    @property
    def hparams(self) -> Union[AttributeDict, MutableMapping]:
        """The collection of hyperparameters saved with :meth:`save_hyperparameters`. It is mutable
        by the user. For the frozen set of initial hyperparameters, use :attr:`hparams_initial`.

        Returns:
            Mutable hyperparameters dictionary
        """
        if not hasattr(self, "_hparams"):
            self._hparams = AttributeDict()
        return self._hparams

    @property
    def hparams_initial(self) -> AttributeDict:
        """The collection of hyperparameters saved with :meth:`save_hyperparameters`. These
        contents are read-only. Manual updates to the saved hyperparameters can instead be
        performed through :attr:`hparams`.

        Returns:
            AttributeDict: immutable initial hyperparameters
        """
        if not hasattr(self, "_hparams_initial"):
            return AttributeDict()
        # prevent any change
        return copy.deepcopy(self._hparams_initial)
