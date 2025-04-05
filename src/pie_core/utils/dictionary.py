from typing import (
    Any,
    Dict,
    Hashable,
    Iterable,
    List,
    MutableMapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)


def list_of_dicts2dict_of_lists(
    list_of_dicts: List[Dict], keys: Optional[Sequence[Hashable]] = None
) -> dict[Hashable, List]:
    """Convert a list of dictionaries to a dictionary of lists.

    Example:
        list_of_dicts = [
            {"a": 1, "b": 2},
            {"a": 3, "b": 4},
            {"a": 5, "b": 6},
        ]
        list_of_dicts2dict_of_lists(list_of_dicts) == {
            "a": [1, 3, 5],
            "b": [2, 4, 6],
        }
    """

    if keys is None:
        keys = list(list_of_dicts[0].keys())
    return {k: [d[k] for d in list_of_dicts] for k in keys}


def dict_of_lists2list_of_dicts(dict_of_lists: Dict[Hashable, List]) -> List[Dict]:
    """Convert a dictionary of lists to a list of dictionaries.

    Example:
        dict_of_lists = {
            "a": [1, 3, 5],
            "b": [2, 4, 6],
        }
        dict_of_lists2list_of_dicts(dict_of_lists) == [
            {"a": 1, "b": 2},
            {"a": 3, "b": 4},
            {"a": 5, "b": 6},
        ]
    """
    return [dict(zip(dict_of_lists.keys(), t)) for t in zip(*dict_of_lists.values())]


def _flatten_dict_gen(
    d: Any,
    parent_key: Tuple[str, ...] = (),
) -> Iterable[Tuple[Tuple[str, ...], Any]]:
    for k, v in d.items():
        new_key = parent_key + (k,)
        if isinstance(v, MutableMapping):
            yield from _flatten_dict_gen(v, new_key)
        else:
            yield new_key, v


def flatten_dict(
    d: MutableMapping, parent_key: str = "", sep: Optional[str] = None
) -> Dict[Union[str, Tuple[str, ...]], Any]:
    """Flatten a nested dictionary.

    Example:
        d = {"a": {"b": 1, "c": 2}, "d": 3}
        flatten_nested_dict(d) == {"a/b": 1, "a/c": 2, "d": 3}
    """
    parent_key_tuple: Tuple[str, ...]
    if sep is None:
        parent_key_tuple = (parent_key,) if parent_key != "" else ()
        return dict(_flatten_dict_gen(d, parent_key=parent_key_tuple))
    else:
        if sep == "" or parent_key == "":
            parent_key_tuple = tuple(parent_key)
        else:
            parent_key_tuple = tuple(parent_key.split(sep))
        return {sep.join(k): v for k, v in _flatten_dict_gen(d, parent_key=parent_key_tuple)}


def _unflatten_dict(d: Dict[Tuple[str, ...], Any]) -> Union[Dict[str, Any], Any]:
    """Unflattens a dictionary with nested keys.

    Example:
        >>> d = {("a", "b", "c"): 1, ("a", "b", "d"): 2, ("a", "e"): 3}
        >>> unflatten_dict(d)
        {'a': {'b': {'c': 1, 'd': 2}, 'e': 3}}
    """
    result: Dict[str, Any] = {}
    for k, v in d.items():
        if len(k) == 0:
            if len(result) > 1:
                raise ValueError("Cannot unflatten dictionary with multiple root keys.")
            return v
        current = result
        for key in k[:-1]:
            current = current.setdefault(key, {})
        current[k[-1]] = v
    return result


def unflatten_dict(
    d: Dict[Union[str, Tuple[str, ...]], Any], sep: Optional[str] = None
) -> Union[Dict[str, Any], Any]:
    """Unflattens a dictionary with nested keys.

    Example:
        >>> d = {"a/b/c": 1, "a/b/d": 2, "a/e": 3}
        >>> unflatten_dict(d, sep="/")
        {'a': {'b': {'c': 1, 'd': 2}, 'e': 3}}
    """

    def _prepare_key(k: Union[str, Tuple[str, ...]]) -> Tuple[str, ...]:
        if isinstance(k, tuple):
            return k
        if sep is not None:
            if sep == "":
                return tuple(k)
            elif isinstance(k, str):
                return tuple(k.split(sep))
        else:
            return (k,)

    return _unflatten_dict({_prepare_key(k): v for k, v in d.items()})
