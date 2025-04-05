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
        >>> l_of_d = [
        >>>     {"a": 1, "b": 2},
        >>>     {"a": 3, "b": 4},
        >>>     {"a": 5, "b": 6},
        >>> ]
        >>> list_of_dicts2dict_of_lists(l_of_d) == {
        >>>     "a": [1, 3, 5],
        >>>     "b": [2, 4, 6],
        >>> }
    """

    if keys is None:
        keys = list(list_of_dicts[0].keys())
    return {k: [d[k] for d in list_of_dicts] for k in keys}


def dict_of_lists2list_of_dicts(dict_of_lists: Dict[Hashable, List]) -> List[Dict]:
    """Convert a dictionary of lists to a list of dictionaries.

    Example:
        >>> d_of_l = {
        >>>    "a": [1, 3, 5],
        >>>    "b": [2, 4, 6],
        >>> }
        >>> dict_of_lists2list_of_dicts(d_of_l) == [
        >>>    {"a": 1, "b": 2},
        >>>    {"a": 3, "b": 4},
        >>>    {"a": 5, "b": 6},
        >>>]
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


def flatten_dict(d: Any, parent_key: Tuple[str, ...] = ()) -> Dict[Tuple[str, ...], Any]:
    """Flatten a nested dictionary with tuple keys. If the input is not a dictionary, it returns a
    dictionary with an empty tuple as the key and the input value as the value.

    Example:
        >>> d = {"a": {"b": 1, "c": 2}, "d": 3}
        >>> flatten_dict(d) == {("a", "b"): 1, ("a", "c"): 2, ("d",): 3}
    """
    if not isinstance(d, MutableMapping):
        return {parent_key: d}
    return dict(_flatten_dict_gen(d, parent_key=parent_key))


def flatten_dict_s(
    d: Any, parent_key: str = "", sep: str = "/"
) -> Dict[Union[str, Tuple[str, ...]], Any]:
    """Flatten a nested dictionary with string keys.

    Example:
        >>> d = {"a": {"b": 1, "c": 2}, "d": 3}
        >>> flatten_dict_s(d) == {"a/b": 1, "a/c": 2, "d": 3}
    """
    parent_key_tuple: Tuple[str, ...]
    if sep == "" or parent_key == "":
        parent_key_tuple = tuple(parent_key)
    else:
        parent_key_tuple = tuple(parent_key.split(sep))
    return {sep.join(k): v for k, v in flatten_dict(d, parent_key=parent_key_tuple).items()}


def unflatten_dict(d: Dict[Tuple[str, ...], Any]) -> Union[Dict[str, Any], Any]:
    """Unflattens a dictionary with nested tuple keys. A dictionary with an empty tuple as the key
    is considered a root key, in which case the value is returned directly.

    Additionally, this version raises an error if there is a conflict such that one key tries
    to treat another key's value as a dictionary. For example:
        {("a",): 1, ("a", "b"): 2}
    should raise a ValueError.

    Example:
        >>> d = {("a", "b", "c"): 1, ("a", "b", "d"): 2, ("a", "e"): 3}
        >>> unflatten_dict(d)
        {'a': {'b': {'c': 1, 'd': 2}, 'e': 3}}
    """
    result: Dict[str, Any] = {}

    for path, value in d.items():
        # If the key path is empty, this is a direct value return:
        if not path:
            # If there's already something in result or if this isn't the only item, it's invalid
            if result or len(d) > 1:
                raise ValueError(
                    "Conflict at root level: trying to descend into a non-dict value."
                )
            return value

        # Walk through the path to create/find intermediate dictionaries
        current = result
        for key in path[:-1]:
            # If we've already stored a non-dict object at this key, it's a conflict
            if key in current and not isinstance(current[key], dict):
                raise ValueError(
                    f"Conflict at path {path}: trying to descend into a non-dict value {current[key]}."
                )
            # Use `setdefault` to ensure we have a dict here
            current = current.setdefault(key, {})

        # If we are about to assign to a key that already holds a dict, that's a conflict
        if path[-1] in current and isinstance(current[path[-1]], dict):
            raise ValueError(
                f"Conflict at path {path}: trying to overwrite existing dict with a non-dict value."
            )

        current[path[-1]] = value

    return result


def unflatten_dict_s(d: Dict[str, Any], sep: str = "/") -> Union[Dict[str, Any], Any]:
    """Unflattens a dictionary with nested string keys.

    Example:
        >>> d = {"a/b/c": 1, "a/b/d": 2, "a/e": 3}
        >>> unflatten_dict_s(d, sep="/")
        {'a': {'b': {'c': 1, 'd': 2}, 'e': 3}}
    """

    def _prepare_key(k: str) -> Tuple[str, ...]:
        return tuple(k) if sep == "" or k == "" else tuple(k.split(sep))

    return unflatten_dict({_prepare_key(k): v for k, v in d.items()})
