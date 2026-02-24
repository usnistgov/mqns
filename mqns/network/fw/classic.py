import functools
from collections.abc import Callable, Mapping
from typing import Any, cast

from mqns.entity.cchannel import ClassicCommandDispatcherMixin, ClassicPacket, RecvClassicPacket, classic_cmd_handler
from mqns.entity.node import Application, Node, QNode
from mqns.network.fw.fib import Fib, FibEntry
from mqns.network.network import QuantumNetwork
from mqns.utils import log


def fw_control_cmd_handler(cmd: str):
    """
    Method decorator for a control message handler in Forwarder.

    ``handle_message(self, msg: dict) -> Any``
    """

    def decorator(f: Callable[[Any, Any], Any]):
        @functools.wraps(f)
        def wrapper(self: "ForwarderClassicMixin", pkt: ClassicPacket, msg: dict):
            log.debug(f"{self.node}: received control message from {pkt.src} | {msg}")
            f(self, msg)
            return True

        return classic_cmd_handler(cmd)(wrapper)

    return decorator


def fw_signaling_cmd_handler(cmd: str):
    """
    Method decorator for a signaling message handler in Forwarder.

    ``handle_message(self, msg: dict, fib_entry: FibEntry) -> Any``
    """

    def decorator(f: Callable[[Any, Any, FibEntry], Any]):
        @functools.wraps(f)
        def wrapper(self: "ForwarderClassicMixin", pkt: ClassicPacket, msg: dict):
            path_id: int = msg["path_id"]
            try:
                fib_entry = self.fib.get(path_id)
            except IndexError:
                log.debug(f"{self.node}: dropping signaling message from {pkt.src}, reason=no-fib-entry | {msg}")
                return True

            if pkt.dest != self.node:
                self.send_msg(pkt.dest, msg, fib_entry, forward=True)
                return True

            log.debug(f"{self.node}: received signaling message from {pkt.src} | {msg}")
            f(self, msg, fib_entry)
            return True

        return classic_cmd_handler(cmd)(wrapper)

    return decorator


class ForwarderClassicMixin(ClassicCommandDispatcherMixin):
    """
    Part of ``Forwarder`` logic related to classical message handling.

    * Dispatch classical control and signaling messages.
    * Forward classical signaling messages according to path_id lookup in FIB.
    """

    node: QNode
    network: QuantumNetwork
    fib: Fib

    def _init_classic_mixin(self) -> None:
        """
        Initializer, must be called from ``Forwarder.__init__()``.
        """
        cast(Application, self).add_handler(self.handle_classic_command, RecvClassicPacket)

    def send_ctrl(self, msg: Mapping):
        ctrl = self.network.get_controller()
        log.debug(f"{self.node}: sending control message to controller | {msg}")
        self.node.get_cchannel(ctrl).send(ClassicPacket(msg, src=self.node, dest=ctrl), ctrl)

    def send_msg(self, dest: Node, msg: Mapping, fib_entry: FibEntry, *, forward=False):
        """
        Send/forward a signaling message along the path specified in FIB entry.
        """
        dest_idx = fib_entry.route.index(dest.name)
        nh = fib_entry.route[fib_entry.own_idx + 1] if dest_idx > fib_entry.own_idx else fib_entry.route[fib_entry.own_idx - 1]
        next_hop = self.network.get_node(nh)

        log.debug(
            f"{self.node}: {'forwarding' if forward else 'sending'} signaling message to {dest.name} via {next_hop.name}"
            f" | {msg}"
        )
        self.node.get_cchannel(next_hop).send(ClassicPacket(msg, src=self.node, dest=dest), next_hop)
