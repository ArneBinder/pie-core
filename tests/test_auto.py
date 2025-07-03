from typing import Optional

from pie_core import Auto, Registrable
from pie_core.hf_hub_mixin import HFHubMixin


class TestHFHubMixin(HFHubMixin):
    config_name: str = "auto_config.json"
    config_type_key: str = "auto_config_type"


class Test(Registrable, TestHFHubMixin):
    foo: str = None

    def __init__(self, *args, foo: Optional[str] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.foo = foo


@Test.register()
class Sub(Test):

    pass


@Test.register()
class Sub2(Test):

    pass


class AutoTest(TestHFHubMixin, Auto[Test]):
    BASE_CLASS = Test


def test_from_config():
    auto_test = AutoTest()
    sub = auto_test.from_config(config={"auto_config_type": "Sub", "foo": "Test"})
    assert isinstance(sub, Sub)
    assert sub.foo == "Test"


def test_from_config_with_kwargs_override():
    auto_test = AutoTest()
    sub = auto_test.from_config(
        config={"auto_config_type": "Sub", "foo": "Test"}, auto_config_type="Sub2", foo="Test2"
    )
    assert isinstance(sub, Sub2)
    assert sub.foo == "Test2"
