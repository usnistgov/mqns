from typing import override

from mqns.simulator import Event, Time, func_to_event


class NopEventA(Event):
    @override
    def invoke(self) -> None:
        pass


class NopEventB(Event):
    @override
    def invoke(self) -> None:
        pass


def test_event_compare():
    t1 = Time.from_sec(1.0)
    t2 = Time.from_sec(2.0)

    e1a = NopEventA(t1)
    e1b = NopEventB(t1)
    e2a = NopEventA(t2)
    e2b = NopEventB(t2)

    assert e1a == e1b
    assert e1a <= e2a
    assert e1a < e2a
    assert e2a > e1a
    assert e2a >= e2b

    assert e1a != 1
    assert e1a != "A"


class PrintEvent(Event):
    @override
    def invoke(self) -> None:
        print("event happened")


def test_event_normal():
    te = PrintEvent(t=Time.from_sec(1), name="test event")
    print(te)

    te.invoke()
    assert not te.is_canceled
    te.cancel()
    assert te.is_canceled


def Print():
    print("event happened")


def test_event_simple():
    te = func_to_event(Time.from_sec(1), Print, name="test event")
    print(te)

    te.invoke()
