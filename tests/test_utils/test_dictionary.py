import pytest

from pie_core.utils.dictionary import (
    dict_of_lists2list_of_dicts,
    dict_update_nested,
    flatten_dict,
    flatten_dict_s,
    list_of_dicts2dict_of_lists,
    unflatten_dict,
    unflatten_dict_s,
)


def test_list_of_dicts2dict_of_lists():
    list_of_dicts = [
        {"a": 1, "b": 2},
        {"a": 3, "b": 4},
        {"a": 5, "b": 6},
    ]
    expected = {
        "a": [1, 3, 5],
        "b": [2, 4, 6],
    }
    assert list_of_dicts2dict_of_lists(list_of_dicts) == expected

    list_of_dicts = []
    expected = {"a": [], "b": []}
    assert list_of_dicts2dict_of_lists(list_of_dicts, keys=["a", "b"]) == expected


def test_dict_of_lists2list_of_dicts():
    dict_of_lists = {
        "a": [1, 3, 5],
        "b": [2, 4, 6],
    }
    expected = [
        {"a": 1, "b": 2},
        {"a": 3, "b": 4},
        {"a": 5, "b": 6},
    ]
    assert dict_of_lists2list_of_dicts(dict_of_lists) == expected

    dict_of_lists = {}
    expected = []
    assert dict_of_lists2list_of_dicts(dict_of_lists) == expected


def test_flatten_dict():
    d = {"a": {"b": 1, "c": 2}, "d": 3}
    assert flatten_dict(d) == {("a", "b"): 1, ("a", "c"): 2, ("d",): 3}

    assert flatten_dict({}) == {}
    assert flatten_dict(1) == {(): 1}


def test_flatten_dict_with_parent_key():
    d = {"a": {"b": 1, "c": 2}, "d": 3}
    assert flatten_dict(d, parent_key=("p",)) == {
        ("p", "a", "b"): 1,
        ("p", "a", "c"): 2,
        ("p", "d"): 3,
    }

    assert flatten_dict({}, parent_key=("p",)) == {}
    assert flatten_dict(1, parent_key=("p",)) == {("p",): 1}


def test_flatten_nested_dict_s():
    d = {"a": {"b": 1, "c": 2}, "d": 3}
    assert flatten_dict_s(d) == {"a/b": 1, "a/c": 2, "d": 3}
    assert flatten_dict_s(d, sep="/") == {"a/b": 1, "a/c": 2, "d": 3}
    assert flatten_dict_s(d, sep=".") == {"a.b": 1, "a.c": 2, "d": 3}
    assert flatten_dict_s(d, sep="") == {"ab": 1, "ac": 2, "d": 3}

    assert flatten_dict_s({}) == {}
    assert flatten_dict_s(1) == {"": 1}


def test_flatten_nested_dict_with_parent_key():
    d = {"a": {"b": 1, "c": 2}, "d": 3}
    assert flatten_dict(d, parent_key=("p",)) == {
        ("p", "a", "b"): 1,
        ("p", "a", "c"): 2,
        ("p", "d"): 3,
    }

    assert flatten_dict_s({}, parent_key="p") == {}
    assert flatten_dict_s(1, parent_key="p") == {"p": 1}


def test_flatten_nested_dict_s_with_parent_key():
    d = {"a": {"b": 1, "c": 2}, "d": 3}

    assert flatten_dict_s(d, parent_key="p/x", sep="/") == {"p/x/a/b": 1, "p/x/a/c": 2, "p/x/d": 3}
    assert flatten_dict_s(d, parent_key="p", sep=".") == {"p.a.b": 1, "p.a.c": 2, "p.d": 3}
    assert flatten_dict_s(d, parent_key="p", sep="") == {"pab": 1, "pac": 2, "pd": 3}

    assert flatten_dict_s({}, parent_key="p") == {}
    assert flatten_dict_s(1, parent_key="p") == {"p": 1}


def test_unflatten_dict():
    d_expected = {"a": {"b": 1, "c": 2}, "d": 3}
    assert unflatten_dict({("a", "b"): 1, ("a", "c"): 2, ("d",): 3}) == d_expected

    assert unflatten_dict({}) == {}

    assert unflatten_dict({(): 1}) == 1


def test_unflatten_dict_multiple_roots():
    with pytest.raises(ValueError) as excinfo:
        unflatten_dict({("a", "b"): 1, ("a",): 2})
    assert (
        str(excinfo.value)
        == "Conflict at path ('a',): trying to overwrite existing dict with a non-dict value."
    )

    with pytest.raises(ValueError) as excinfo:
        unflatten_dict({("a",): 1, (): 2})
    assert str(excinfo.value) == "Conflict at root level: trying to descend into a non-dict value."

    with pytest.raises(ValueError) as excinfo:
        unflatten_dict({(): 1, ("b",): 2})
    assert str(excinfo.value) == "Conflict at root level: trying to descend into a non-dict value."

    # check more complex case
    with pytest.raises(ValueError) as excinfo:
        unflatten_dict({("a", "b"): 1, ("a",): 2, ("a", "c"): 3})
    assert (
        str(excinfo.value)
        == "Conflict at path ('a',): trying to overwrite existing dict with a non-dict value."
    )

    # check more level of nesting
    with pytest.raises(ValueError) as excinfo:
        unflatten_dict({("a", "b", "c"): 1, ("a", "b"): 2})
    assert (
        str(excinfo.value)
        == "Conflict at path ('a', 'b'): trying to overwrite existing dict with a non-dict value."
    )

    with pytest.raises(ValueError) as excinfo:
        unflatten_dict({("a", "b"): 1, ("a", "b", "c"): 2})


def test_unflatten_dict_s():
    d_expected = {"a": {"b": 1, "c": 2}, "d": 3}
    assert unflatten_dict_s({"ab": 1, "ac": 2, "d": 3}) == {"ab": 1, "ac": 2, "d": 3}
    assert unflatten_dict_s({"a.b": 1, "a.c": 2, "d": 3}, sep=".") == d_expected
    assert unflatten_dict_s({"a/b": 1, "a/c": 2, "d": 3}, sep="/") == d_expected
    assert unflatten_dict_s({"ab": 1, "ac": 2, "d": 3}, sep="") == d_expected

    assert unflatten_dict_s({}) == {}

    assert unflatten_dict_s({"": 1}) == 1


def test_unflatten_dict_s_multiple_roots():

    with pytest.raises(ValueError) as excinfo:
        unflatten_dict_s({"a": 1, "": 2})
    assert str(excinfo.value) == "Conflict at root level: trying to descend into a non-dict value."

    with pytest.raises(ValueError) as excinfo:
        unflatten_dict_s({"": 1, "b": 2})
    assert str(excinfo.value) == "Conflict at root level: trying to descend into a non-dict value."

    with pytest.raises(ValueError) as excinfo:
        unflatten_dict_s({"a/b": 1, "a": 2})
    assert (
        str(excinfo.value)
        == "Conflict at path ('a',): trying to overwrite existing dict with a non-dict value."
    )


def test_dict_update_nested():

    # simple cases from docstring
    d = {"a": {"b": {"c": 1, "d": 2}, "e": 3}}
    u = {"a": {"b": {"c": 4, "d": 5}, "f": 6}}
    dict_update_nested(d, u, True)
    assert d == u == {"a": {"b": {"c": 4, "d": 5}, "f": 6}}

    d = {"a": {"b": {"c": 1, "d": 2}, "e": 3}}
    u = {"a": {"b": {"c": 4, "d": 5}, "f": 6}}
    dict_update_nested(d, u, False)
    assert d == {"a": {"b": {"c": 1, "d": 2}, "e": 3}}

    d = {"a": {"b": {"c": 1, "d": 2}, "e": 3}}
    u = {"a": {"b": {"c": 4, "d": 5}}}
    dict_update_nested(d, u)
    assert d == {"a": {"b": {"c": 4, "d": 5}, "e": 3}}

    d = {"a": {"b": {"c": 1}, "d": {"e": 2}}}
    u = {"a": {"b": {"c": 3}, "d": {"e": 4}}}
    override = {"a": {"b": True, "d": False}}
    dict_update_nested(d, u, override)
    assert d == {"a": {"b": {"c": 3}, "d": {"e": 2}}}

    # Override dicts
    # 'd' override ignored, override value is used only if merging values are both dicts.
    d = {"a": {"b": {"c": 1}}, "d": 2, "e": 3}
    u = {"a": {"b": {"c": 3}}, "d": 4, "f": 5}
    override = {"a": {"b": True}, "d": False}
    dict_update_nested(d, u, override)
    assert d == {"a": {"b": {"c": 3}}, "d": 4, "e": 3, "f": 5}

    # More nested override
    d = {"a": {"b": {"c": {"d": 1}}}}
    u = {"a": {"b": {"c": {"d": 2}}}}
    override = {"a": {"b": {"c": False}}}
    dict_update_nested(d, u, override)
    assert d == {"a": {"b": {"c": {"d": 1}}}}

    # Override for multiple targets
    d = {"a": {"b": {"c": {"d": 1}, "e": {"f": 2}}}}
    u = {"a": {"b": {"c": {"d": 3}, "e": {"f": 4}}}}
    override = {"a": {"b": {"c": True, "e": False}}}
    dict_update_nested(d, u, override)
    assert d == {"a": {"b": {"c": {"d": 3}, "e": {"f": 2}}}}

    # Override contains a target not contained in any of dicts (should not impact anything)
    d = {"a": {"b": {"c": {"d": 1}, "e": {"f": 2}}}}
    u = {"a": {"b": {"c": {"d": 3}, "e": {"f": 4}}}}
    override = {"g": True, "h": False}
    dict_update_nested(d, u, override)
    assert d == {"a": {"b": {"c": {"d": 3}, "e": {"f": 4}}}}

    # Update not-dict value with dict value
    with pytest.raises(ValueError) as excinfo:
        dict_update_nested({"a": 1}, {"a": {"b": 1}})
    assert str(excinfo.value) == "Cannot merge 1 and {'b': 1} because 1 is not a dict."

    # !Vice-versa is not checked and dict will be updated
    # with pytest.raises(ValueError) as excinfo:
    #     dict_update_nested({"a": {"b": 1}}, {"a": 1})
    # assert str(excinfo.value) == "Cannot merge {'b': 1} and 1 because 1 is not a dict."
