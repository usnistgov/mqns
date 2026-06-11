from typing import final, override

from mqns.simulator import Event, EventDispatcherMixin, Simulator, Time, event_handler


class OwnedEvent(Event):
    def __init__(self, t: Time, owner: "OwnerBase"):
        super().__init__(t)
        self.owner = owner

    @override
    def invoke(self):
        self.owner.handle(self)


@final
class EventA(OwnedEvent):
    pass


@final
class EventB(OwnedEvent):
    pass


@final
class EventC(OwnedEvent):
    pass


@final
class EventD(OwnedEvent):
    pass


class OwnerBase(EventDispatcherMixin):
    def __init__(self):
        self.invoked = set[str]()

    @event_handler
    def handle_a(self, event: "EventA"):
        _ = event
        self.invoked.add("B.a")

    @event_handler
    def handle_b(self, event: EventB):
        _ = event
        self.invoked.add("B.b")

    @event_handler
    def handle_c(self, event: EventC):
        _ = event
        self.invoked.add("B.c")


class OwnerSub(OwnerBase):
    @override
    def handle_b(self, event: EventB):
        _ = event
        self.invoked.add("S.b")

    @override
    @event_handler
    def handle_c(self, event: EventC):
        _ = event
        self.invoked.add("S.c")

    @event_handler
    def handle_d(self, event: EventD):
        _ = event
        self.invoked.add("S.d")


def test_dispatcher():
    s = Simulator(0, 1, accuracy=1000)
    owner = OwnerSub()
    s.add_event(EventA(s.ts, owner))
    s.add_event(EventB(s.ts, owner))
    s.add_event(EventC(s.ts, owner))
    s.add_event(EventD(s.ts, owner))

    s.run()
    assert owner.invoked == {"B.a", "S.b", "S.c", "S.d"}
