# Copyright © 2019-2025 The Lightning AI team
# Modifications Copyright © 2025 Arne Binder
#
# The original work lives at:
#     https://github.com/Lightning-AI/pytorch-lightning/blob/2.5.1/src/lightning/pytorch/utilities/parsing.py
#
# NOTICE — the content has been modified from the original Lightning version:
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
"""Utilities used for parameter parsing."""

import copy
import inspect
import types
from collections.abc import Sequence
from dataclasses import fields, is_dataclass
from typing import Any, Literal, Optional, Union


def parse_class_init_keys(cls: type) -> tuple[str, Optional[str], Optional[str]]:
    """Parse key words for standard ``self``, ``*args`` and ``**kwargs``.

    Examples:

        >>> class Model:
        ...     def __init__(self, hparams, *my_args, anykw=42, **my_kwargs):
        ...         pass
        >>> parse_class_init_keys(Model)
        ('self', 'my_args', 'my_kwargs')
    """
    init_parameters = inspect.signature(cls.__init__).parameters  # type: ignore[misc]
    # docs claims the params are always ordered
    # https://docs.python.org/3/library/inspect.html#inspect.Signature.parameters
    init_params = list(init_parameters.values())
    # self is always first
    n_self = init_params[0].name

    def _get_first_if_any(
        params: list[inspect.Parameter],
        param_type: Literal[
            inspect._ParameterKind.VAR_POSITIONAL, inspect._ParameterKind.VAR_KEYWORD
        ],
    ) -> Optional[str]:
        for p in params:
            if p.kind == param_type:
                return p.name
        return None

    n_args = _get_first_if_any(init_params, inspect.Parameter.VAR_POSITIONAL)
    n_kwargs = _get_first_if_any(init_params, inspect.Parameter.VAR_KEYWORD)

    return n_self, n_args, n_kwargs


def get_init_args(frame: types.FrameType) -> dict[str, Any]:  # pragma: no-cover
    """For backwards compatibility: #16369."""
    _, local_args = _get_init_args(frame)
    return local_args


def _get_init_args(frame: types.FrameType) -> tuple[Optional[Any], dict[str, Any]]:
    _, _, _, local_vars = inspect.getargvalues(frame)
    if "__class__" not in local_vars:
        return None, {}
    cls = local_vars["__class__"]
    init_parameters = inspect.signature(cls.__init__).parameters
    self_var, args_var, kwargs_var = parse_class_init_keys(cls)
    filtered_vars = [n for n in (self_var, args_var, kwargs_var) if n]
    exclude_argnames = (*filtered_vars, "__class__", "frame", "frame_args")
    # only collect variables that appear in the signature
    local_args = {k: local_vars[k] for k in init_parameters}
    # kwargs_var might be None => raised an error by mypy
    if kwargs_var:
        local_args.update(local_args.get(kwargs_var, {}))
    local_args = {k: v for k, v in local_args.items() if k not in exclude_argnames}
    self_arg = local_vars.get(self_var, None)
    return self_arg, local_args


def collect_init_args(
    frame: types.FrameType,
    path_args: list[dict[str, Any]],
    inside: bool = False,
    classes: tuple[type, ...] = (),
) -> list[dict[str, Any]]:
    """Recursively collects the arguments passed to the child constructors in the inheritance tree.

    Args:
        frame: the current stack frame
        path_args: a list of dictionaries containing the constructor args in all parent classes
        inside: track if we are inside inheritance path, avoid terminating too soon
        classes: the classes in which to inspect the frames

    Return:
          A list of dictionaries where each dictionary contains the arguments passed to the
          constructor at that level. The last entry corresponds to the constructor call of the
          most specific class in the hierarchy.
    """
    _, _, _, local_vars = inspect.getargvalues(frame)
    # frame.f_back must be of a type types.FrameType for get_init_args/collect_init_args due to mypy
    if not isinstance(frame.f_back, types.FrameType):
        return path_args

    local_self, local_args = _get_init_args(frame)
    if "__class__" in local_vars and (not classes or isinstance(local_self, classes)):
        # recursive update
        path_args.append(local_args)
        return collect_init_args(frame.f_back, path_args, inside=True, classes=classes)
    if not inside:
        return collect_init_args(frame.f_back, path_args, inside=False, classes=classes)
    return path_args


def save_hyperparameters(
    obj: Any,
    *args: Any,
    ignore: Optional[Union[Sequence[str], str]] = None,
    frame: Optional[types.FrameType] = None,
    given_hparams: Optional[dict[str, Any]] = None,
) -> None:
    """See :meth:`~pytorch_lightning.LightningModule.save_hyperparameters`"""

    if len(args) == 1 and not isinstance(args, str) and not args[0]:
        # args[0] is an empty container
        return

    if not frame:
        current_frame = inspect.currentframe()
        # inspect.currentframe() return type is Optional[types.FrameType]: current_frame.f_back called only if available
        if current_frame:
            frame = current_frame.f_back
    if not isinstance(frame, types.FrameType):
        raise AttributeError("There is no `frame` available while being required.")

    if given_hparams is not None:
        init_args = given_hparams
    elif is_dataclass(obj):
        init_args = {f.name: getattr(obj, f.name) for f in fields(obj)}
    else:
        init_args = {}

        from pie_core.hparams_mixin import PieHyperparametersMixin

        for local_args in collect_init_args(frame, [], classes=(PieHyperparametersMixin,)):
            init_args.update(local_args)

    if ignore is None:
        ignore = []
    elif isinstance(ignore, str):
        ignore = [ignore]
    elif isinstance(ignore, (list, tuple)):
        ignore = [arg for arg in ignore if isinstance(arg, str)]

    ignore = list(set(ignore))
    init_args = {k: v for k, v in init_args.items() if k not in ignore}

    if not args:
        # take all arguments
        hp = init_args
        obj._hparams_name = "kwargs" if hp else None
    else:
        # take only listed arguments in `save_hparams`
        isx_non_str = [i for i, arg in enumerate(args) if not isinstance(arg, str)]
        if len(isx_non_str) == 1:
            hp = args[isx_non_str[0]]
            cand_names = [k for k, v in init_args.items() if v == hp]
            obj._hparams_name = cand_names[0] if cand_names else None
        else:
            hp = {arg: init_args[arg] for arg in args if isinstance(arg, str)}
            obj._hparams_name = "kwargs"

    # `hparams` are expected here
    obj._set_hparams(hp)

    # make a deep copy so there are no other runtime changes reflected
    obj._hparams_initial = copy.deepcopy(obj._hparams)
