from pie_core.utils.dictionary import (
    dict_of_lists2list_of_dicts,
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


def test_flatten_nested_dict_s():
    d = {"a": {"b": 1, "c": 2}, "d": 3}
    assert flatten_dict_s(d) == {"a/b": 1, "a/c": 2, "d": 3}
    assert flatten_dict_s(d, sep="/") == {"a/b": 1, "a/c": 2, "d": 3}
    assert flatten_dict_s(d, sep=".") == {"a.b": 1, "a.c": 2, "d": 3}
    assert flatten_dict_s(d, sep="") == {"ab": 1, "ac": 2, "d": 3}

    assert flatten_dict_s({}) == {}


def test_flatten_nested_dict_with_parent_key():
    d = {"a": {"b": 1, "c": 2}, "d": 3}
    assert flatten_dict(d, parent_key=("p",)) == {
        ("p", "a", "b"): 1,
        ("p", "a", "c"): 2,
        ("p", "d"): 3,
    }

    assert flatten_dict_s({}, parent_key="p") == {}


def test_flatten_nested_dict_s_with_parent_key():
    d = {"a": {"b": 1, "c": 2}, "d": 3}

    assert flatten_dict_s(d, parent_key="p/x", sep="/") == {"p/x/a/b": 1, "p/x/a/c": 2, "p/x/d": 3}
    assert flatten_dict_s(d, parent_key="p", sep=".") == {"p.a.b": 1, "p.a.c": 2, "p.d": 3}
    assert flatten_dict_s(d, parent_key="p", sep="") == {"pab": 1, "pac": 2, "pd": 3}

    assert flatten_dict_s({}, parent_key="p") == {}


def test_unflatten_dict():
    d_expected = {"a": {"b": 1, "c": 2}, "d": 3}
    assert unflatten_dict({("a", "b"): 1, ("a", "c"): 2, ("d",): 3}) == d_expected

    assert unflatten_dict_s({}) == {}


def test_unflatten_dict_s():
    d_expected = {"a": {"b": 1, "c": 2}, "d": 3}
    assert unflatten_dict_s({"ab": 1, "ac": 2, "d": 3}) == {"ab": 1, "ac": 2, "d": 3}
    assert unflatten_dict_s({"a.b": 1, "a.c": 2, "d": 3}, sep=".") == d_expected
    assert unflatten_dict_s({"a/b": 1, "a/c": 2, "d": 3}, sep="/") == d_expected
    assert unflatten_dict_s({"ab": 1, "ac": 2, "d": 3}, sep="") == d_expected

    assert unflatten_dict_s({}) == {}
