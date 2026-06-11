import inspect
from collections.abc import Callable
from typing import Any, get_type_hints

from mqns.simulator.event import Event


def _extract_event_type(f: Callable) -> type[Event]:
    sig = inspect.signature(f)
    params = list(sig.parameters.values())
    if len(params) != 2:
        raise TypeError("register_handler: handler must accept two parameters")

    hints = get_type_hints(f)
    typ = hints.get(params[1].name)
    if not (isinstance(typ, type) and issubclass(typ, Event)):
        raise TypeError("register_handler: handler must accept Event subclass")
    if not getattr(typ, "__final__", False):
        raise TypeError(f"register_handler: {typ} must be marked @final")
    return typ


def event_handler[E: Event](f: Callable[[Any, E], bool | None]):
    """
    Method decorator to register an event handler.

    Args:
        f: Handler function, ``def handle_event_a(self, event: EventA) -> bool|None``.

    This decorator is effective only if ``EventDispatcher.handle`` is not overridden.
    See ``EventDispatcher.handle`` for the semantics of ``f``'s return value.

    The event handler registry is per class and supports inheritance.
    Handlers may be registered both from the base class and the subclass.
    If a subclass overrides an event handler in the base class, it must keep the same function signature.
    """
    typ = _extract_event_type(f)
    setattr(f, "_event_handler", typ)
    return f


def _populate_handlers(cls: type):
    # Identify handler method names.
    handler_names = set[str]()
    for base in reversed(cls.mro()):
        for name, attr in base.__dict__.items():
            if getattr(attr, "_event_handler", None):
                handler_names.add(name)

    # Map commands to the most specific implementation in this class.
    handler_map: dict[type[Event], Callable] = {}
    for name in handler_names:
        handler = getattr(cls, name)
        typ = getattr(handler, "_event_handler", None) or _extract_event_type(handler)
        handler_map[typ] = handler
    cls._event_handlers = handler_map


class EventDispatcherMixin:
    """
    Mixin class for event dispatching functionality on event target (e.g. ``Application``).
    """

    def handle(self, event: Event, /) -> bool | None:
        """
        Dispatch an event.

        Args:
            event: Event instance.

        Returns:
        * If True, the event is fully handled and not passed to the next event target.
        * Otherwise, the event is passed to the next event target.
        """
        cls: type = type(self)

        if "_event_handlers" not in cls.__dict__:
            _populate_handlers(cls)

        handler = cls._event_handlers.get(type(event))
        if handler:
            return handler(self, event)

        return False
