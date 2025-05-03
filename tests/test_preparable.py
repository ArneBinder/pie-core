import logging

import pytest

from pie_core import PreparableMixin

logger = logging.getLogger(__name__)


class PreparableObject(PreparableMixin):
    PREPARED_ATTRIBUTES = ["attr"]

    def _prepare(self):
        self.attr = True

    def _post_prepare(self):
        if self.attr:
            logger.info("Post-prepared successfully!")


@pytest.fixture(scope="module")
def prepared_object():
    obj = PreparableObject()
    obj.prepare()
    return obj


@pytest.fixture(scope="module")
def unprepared_object():
    obj = PreparableObject()
    return obj


def test_is_prepared(unprepared_object, prepared_object):
    assert prepared_object.is_prepared
    assert not unprepared_object.is_prepared


def test_prepared_attributes(unprepared_object, prepared_object):
    assert not unprepared_object.is_prepared
    with pytest.raises(Exception) as excinfo:
        attrs = unprepared_object.prepared_attributes
    assert str(excinfo.value) == "The PreparableObject is not prepared."

    assert prepared_object.is_prepared
    assert prepared_object.prepared_attributes == {"attr": True}


def test_assert_is_prepared(unprepared_object):
    with pytest.raises(Exception) as excinfo:
        unprepared_object.assert_is_prepared()
    assert str(excinfo.value) == "Required attributes that are not set: ['attr']"


def test_post_prepare(prepared_object, unprepared_object, caplog):
    with caplog.at_level(logging.INFO):
        prepared_object.post_prepare()
    assert "Post-prepared successfully!" in caplog.text

    with pytest.raises(Exception) as excinfo:
        unprepared_object.post_prepare()
    assert str(excinfo.value) == "Required attributes that are not set: ['attr']"


def test_prepare():
    obj = PreparableObject()
    assert not obj.is_prepared
    obj.prepare()
    assert obj.is_prepared


def test_prepare_prepared_object(prepared_object, caplog):
    with caplog.at_level(logging.WARNING):
        prepared_object.prepare()
    assert (
        "The PreparableObject is already prepared, do not prepare again.\nattr = True"
        in caplog.text
    )


def test_prepare_with_bad_prepare_impl():
    class WrongPrepareObject(PreparableMixin):
        PREPARED_ATTRIBUTES = ["attr"]

        def _prepare(self):
            pass

    obj = WrongPrepareObject()

    with pytest.raises(Exception) as excinfo:
        obj.prepare()
    assert str(excinfo.value) == (
        "_prepare() was called, but the WrongPrepareObject is not prepared. "
        "Required attributes that are not set: ['attr']"
    )
