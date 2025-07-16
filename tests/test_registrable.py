import pytest

from pie_core.registrable import Registrable, RegistrationError


class A(Registrable):
    """A base class."""

    pass


@A.register()
class B(A):
    """A registered subclass."""

    pass


class C(A):
    """Unregistered subclass."""

    pass


class D(Registrable):
    """Unregistered base class with no children."""

    pass


def test_registrable():
    """A basic Registrable class usage."""
    # Check the registry entries are created
    assert A in Registrable._registry
    assert Registrable._registry[A] == {"B": B}

    # Instantiate an Object of type B by its name.
    clazz = A.by_name("B")()
    assert isinstance(clazz, B)


def test_base_class():

    assert A.base_class() is A
    assert B.base_class() is A
    assert C.base_class() is A

    with pytest.raises(RegistrationError) as e:
        D.base_class()
    assert str(e.value) == (
        f"{D.__name__} has no defined base class. "
        f"Please annotate {D.__name__} with @<SOME-PARENT-OF-{D.__name__}>.register() "
        f"to register it at <SOME-PARENT-OF-{D.__name__}>. "
        "In case you don't want to add this class to registry, you can instead manually set "
        f"the {D.__name__}.BASE_CLASS attribute."
    )


def test_has_base_class():
    assert A.has_base_class()
    assert B.has_base_class()
    assert C.has_base_class()
    assert not D.has_base_class()


def test_register():
    class Base(Registrable):
        pass

    class Test(Base):
        pass

    # Base.register() does two main things:
    # 1) Sets Base.BASE_CLASS to itself (if it already isn't)
    # 2) Returns a decorator function that adds argument Class to Base's registry.

    # We call Base.register() not as a decorator as usual, since we want to test the above parts separately.

    # Test register() functionality
    assert Base.BASE_CLASS is None
    assert Test.BASE_CLASS is None

    wrapper = Base.register()
    assert Base.BASE_CLASS is Base
    assert Test.BASE_CLASS is Base

    # Test decorator functionality
    assert Registrable._registry[Base] == {}
    wrapper(Test)
    assert Registrable._registry[Base] == {"Test": Test}


def test_register_from_a_subclass():
    class Base(Registrable):
        pass

    @Base.register()
    class Sub(Base):
        pass

    with pytest.raises(RegistrationError) as e:
        # Here we do not decorate any class, since error happens in register() itself,
        # and not in a decorator function it returns.
        Sub.register()
    assert str(e.value) == "Cannot register Sub; it is already registered as a subclass of Base"


def test_register_wrapper_target_not_a_subclass():
    class Base(Registrable):
        pass

    with pytest.raises(RegistrationError) as e:

        @Base.register()
        class Sub:
            pass

    assert str(e.value) == "Cannot register Sub as Sub; Sub must be a subclass of Base"


def test_register_wrapper_target_name_already_registered():
    class Base(Registrable):
        pass

    @Base.register()
    class Sub(Base):
        pass

    with pytest.raises(RegistrationError) as e:

        @Base.register("Sub")
        class Bus(Base):
            pass

    assert str(e.value) == "Cannot register Bus as Sub; name already in use for Sub"


def test_register_wrapper_wrong_base_class(caplog):
    class Base(Registrable):
        pass

    @Base.register()
    class Sub(Base):
        pass

    with pytest.raises(RegistrationError) as e:

        @Base.register()
        class Sub2(Base):
            BASE_CLASS = Sub

    assert str(e.value) == (
        "Sub2 already has a BASE_CLASS set to Sub. "
        "Please remove the BASE_CLASS attribute or use a different name for the registration."
    )


def test_by_name():
    assert A.by_name("B") == B

    with pytest.raises(RegistrationError) as e:
        A.by_name("C")
    assert str(e.value) == "C is not a registered name for A."

    with pytest.raises(RegistrationError) as e:
        A.by_name("D")
    assert str(e.value) == "D is not a registered name for A."

    # You should call by_name() from a base class, you can't just
    with pytest.raises(RegistrationError) as e:
        B.by_name("B")
    assert str(e.value) == "B is not a registered name for B."
    # but you can:
    assert B.base_class().by_name("B") == B
    # or from any other subclass:
    assert C.base_class().by_name("B") == B


def test_registered_name_for_class():
    assert A.registered_name_for_class(B) == "B"
    assert A.registered_name_for_class(C) is None
    assert A.registered_name_for_class(D) is None


def test_name_for_object_class():
    b = B()
    c = C()

    class Test(Registrable):
        pass

    test = Test()

    assert A.name_for_object_class(b) == "B"
    # Function returns class name if no entry in registry was found
    assert A.name_for_object_class(c) == "C"
    assert A.name_for_object_class(test) == "Test"


def test_auto_classes():
    class Base(Registrable):
        pass

    class NotSub(Registrable[Base]):
        """Some class not directly subclassed from Base, it can not be registered itself, but it
        can be used to retrieve other subclasses of Base."""

        BASE_CLASS = Base

    @Base.register()
    class Sub(Base):
        pass

    assert NotSub.base_class().by_name("Sub") == Sub

    with pytest.raises(RegistrationError) as e:
        NotSub.base_class().by_name("NotSub")
    assert e.value.args[0] == "NotSub is not a registered name for Base."
