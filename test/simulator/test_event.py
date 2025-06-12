from qns.simulator.event import Event, func_to_event
from qns.simulator.ts import Time


def test_event_compare():
    t1 = Time(sec=1.0)
    t2 = Time(sec=2.0)

    e0a = Event()
    e0b = Event()
    e1a = Event(t1)
    e1b = Event(t1)
    e2a = Event(t2)
    e2b = Event(t2)

    assert e0a == e0b
    assert e1a == e1b
    assert e0a <= e0b
    assert e0a < e1a
    assert e1a > e0a
    assert e1a <= e2a
    assert e1a < e2a
    assert e2a > e1a
    assert e2a >= e2b

    assert e0a != 1
    assert e0a != "A"


class PrintEvent(Event):
    def invoke(self) -> None:
        print("event happened")


def test_event_normal():
    te = PrintEvent(t=Time(sec=1), name="test event")
    print(te)

    te.invoke()
    assert not te.is_canceled
    te.cancel()
    assert te.is_canceled


def Print():
    print("event happened")


def test_event_simple():
    te = func_to_event(t=Time(sec=1), name="test event", fn=Print)
    print(te)

    te.invoke()
