"""
ClassicBridge via NATS JetStream
================================

This module provides :py:class:`ClassicBridge` and :py:class:`ClassicConnector`.
They allow the logic of non-quantum node(s) to be implemented in an external program.
The node whose logic is taken over by the external program is called a *split node*.

A *split node* still exists in the simulated network topology and has :py:class:`ClassicChannels <mqns.entity.cchannel.ClassicChannel>` connected.
Its :py:class:`~mqns.entity.node.Node` (including :py:class:`~mqns.entity.node.Controller` that is a subclass) must have the :py:class:`ClassicBridge` application installed.
When the split node receives a classic packet from another node, it is intercepted by the :py:class:`ClassicBridge` application and passed to the external program for processing.
When the external program wants to send a classic packet to another node, the :py:class:`ClassicBridge` application injects the packet into the network on behalf of the external program.

A prominent use case is simulating only the data plane with MQNS, while the control plane is implemented externally.
In this case, the :py:class:`~mqns.entity.node.Controller` is the only *split node* in the topology, which has the :py:class:`ClassicBridge` application installed, but no longer contains the usual :py:class:`~mqns.network.fw.RoutingController` application.
Such topology can be constructed with :py:meth:`NetworkBuilder.external_controller() <mqns.network.builder.NetworkBuilder.external_controller>` method.
The **extctrl** example demonstrates this use case.

Communication Protocol
----------------------

Both MQNS :py:class:`ClassicConnector` and the external program connect to a `NATS <https://nats.io/>`__ message broker.
They communicate with the `JetStream <https://docs.nats.io/concepts/jetstream>`__ protocol.

Each classic packet received or sent by a *split node* is represented by a NATS message.

Receiving a Packet
^^^^^^^^^^^^^^^^^^

When a *split node* receives a classic packet, it is published as a NATS JetStream message:

* Subject: ``<PREFIX>.O.<dst>.<src>``

  * ``<PREFIX>`` is a fixed prefix passed to ClassicConnector or ClassicBridge constructor.
  * ``<dst>`` is the name of the *split node* that received this packet.
  * ``<src>`` is the name of the simulated node that sent this packet.

* Payload: the classic packet payload
* Header ``t``: packet receipt time in simulation time slots
* Header ``fmt``: packet format, either ``"json"`` or ``"bytes"``

The external program should subscribe to ``<PREFIX>.O.*.*`` subjects to receive these NATS messages for processing.

Sending a Packet
^^^^^^^^^^^^^^^^

The external program can send a classic packet to another simulated node by publishing a NATS JetStream message:

* Subject: ``<PREFIX>.I.<dst>.<src>``

  * ``<dst>`` is the name of the simulated node that should receive this packet.
  * ``<src>`` is the name of the *split node* that should send this packet.

* Payload: the classic packet payload
* Header ``t``: packet transmission time in simulation time slots
* Header ``fmt``: packet format, either ``"json"`` or ``"bytes"``

MQNS :py:class:`ClassicConnector` subscribes to ``<PREFIX>.I.*.*`` subjects so that it can receive these NATS messages and inject them into the simulated topology.

Classic Packet Payload
^^^^^^^^^^^^^^^^^^^^^^

The classic packet payload is determined by the application logic.
For packets between the controller and forwarders, :py:mod:`mqns.network.fw.message` is a good start.

Apart from NATS messages that represent classic packets between a *split node* and other simulated nodes, :py:class:`ClassicConnector` also use NATS messages to ensure temporal consistency, described in the next subsection.
To avoid conflicts with these special messages, the token ``_`` cannot be used as a node name.

Clock Gate Protocol
-------------------

MQNS's clock may never advance past the external program's next possible input — otherwise causality breaks.
The *clock gate* is the mechanism that enforces this: MQNS runs only as far as the external program explicitly allows.

Release the Gate ("I-gate")
^^^^^^^^^^^^^^^^^^^^^^^^^^^

To release (move forward) the clock gate, the external program should publish a NATS JetStream message:

* Subject: ``<PREFIX>.I._.gate``
* Header ``t``: new upper-bound in simulation time slots

This allows MQNS to run until the specified timestamp, including any events scheduled within time slot ``t``.

Report Simulation Progress ("O-gate")
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

MQNS reports reaching the gate by publishing:

* Subject: ``<PREFIX>.O._.gate``
* Header ``t``: reached time in simulation time slots

MQNS (re-)publishes this message in these occasions:

* When the simulation starts, O-gate is published for the simulation start time (:py:attr:`~mqns.simulator.Simulator.ts`, usually zero).
  This is not a reply to anything, but only indicates simulator readiness.
* When the gate is released to a greater timestamp and the simulation has reached that time slot.
* When new packet(s) are injected at the current gate and these packet(s) have been sent into the topology.

Note that if the external program publishes I-gate with the same ``t`` as the current gate (i.e. a redundant I-gate at an already-reached timestamp), without injecting any packets, MQNS would not publish another O-gate.

Stop the Simulation
^^^^^^^^^^^^^^^^^^^

The external program may stop MQNS by publishing a NATS JetStream message:

* Subject: ``<PREFIX>.I._.stop``
* Header ``t``: stop time in simulation time slots

Publishing this message implies releasing the gate to the stop time.
However, MQNS will not publish an O-gate message before stopping.

Header ``t``
^^^^^^^^^^^^

Header ``t`` in every message is represented as *simulation time slots*.
See :py:meth:`Time.sec_to_slot <mqns.simulator.Time.sec_to_slot>` and :py:meth:`Time.sec <mqns.simulator.Time.sec>` on how it relates to *seconds* unit.

All packets and gate updates must be published in non-decreasing order.
In other words, the header ``t`` for every message must be greater than or equal to the timestamp on all previous messages published by the same party (i.e. either MQNS or the external program).
Once the clock moves forward, any incoming messages with a timestamp strictly smaller than a previous message will be considered causality violations and crash the simulation.

Gate Rules, Summarized
^^^^^^^^^^^^^^^^^^^^^^

The rules, all of which exist so the two timelines never disagree:

1. **The gate starts at the simulation start time** (:py:attr:`~mqns.simulator.Simulator.ts`, usually zero) and only the external program can release it, by publishing I-gate with a larger timestamp.
   MQNS never releases the gate by itself, and moving it backward raises ``AssertionError``.
2. **The gate blocks progress, not input.**
   Injected packets are queued regardless of the gate and processed once the gate opens past their timestamp.
   Nothing is dropped or reordered by the gate itself.
3. **Every message carries a non-decreasing timestamp.**
   All NATS messages from the external program, including packets and I-gate, share one ``t`` stream: stamping anything earlier than a previous message is a causality violation, which would cause the MQNS worker thread to raise ``RuntimeError`` and terminate the simulation.
   (Same-slot events may also run in any order, so never send two messages in one slot and rely on their order.)
4. **Release the gate only when the release moves something.**
   MQNS publishes O-gate only when the simulation's reached time advances past what was last reported.
   Releasing I-gate to an already-reached timestamp, with no new packets injected since the last report, advances nothing — no new O-gate is produced, and a controller blocking on that confirmation will hang forever.
5. **Expect an opening O-gate.**
   Before the first release, MQNS may report reaching :py:attr:`~mqns.simulator.Simulator.ts` once.
   This is not a reply to anything; a controller should just absorb it.
6. **Once the gate is released to ``G``, stamp new events at or after ``G``** — never before.
   Events stamped before MQNS's current time are dropped.

This is what makes the external program a *peer* of MQNS with its own timeline, rather than a subroutine of it: it can park a reaction on its own timer, or act with no data-plane event at all, and MQNS waits.
"""  # noqa: E501

import asyncio
import os
import queue
import threading
from dataclasses import dataclass
from typing import Final, Protocol, override

from mqns.entity.cchannel import ClassicPacket, RecvClassicPacket
from mqns.entity.node import Application, Node
from mqns.simulator import EventHandleSlot, Simulator, event_handler, func_to_event
from mqns.utils import log

try:
    import nats
except ImportError:
    nats = None


class _TransmittableItem(Protocol):
    def encode(self) -> tuple[str, bytes, dict[str, str]]: ...


@dataclass
class _GateReached:
    t: int

    def encode(self) -> tuple[str, bytes, dict[str, str]]:
        return "_.gate", b"", {"t": f"{self.t}"}


@dataclass
class _ClassicBridgePacket:
    t: int
    src: str
    dst: str
    payload: bytes
    is_json: bool

    def encode(self) -> tuple[str, bytes, dict[str, str]]:
        headers = {"t": f"{self.t}", "fmt": "json" if self.is_json else "bytes"}
        return f"{self.dst}.{self.src}", self.payload, headers


class ClassicConnector:
    """
    ClassicConnector is a singleton (one instance per :py:class:`~mqns.simulator.Simulator`) that
    facilitates synchronized communication via NATS JetStream, between one or more *split nodes*
    (nodes with :py:class:`ClassicBridge`) in the simulation and an external program that
    implements the application logic of these nodes.
    It acts as a PDES (Parallel Discrete Event Simulation) bridge, implementing
    a conservative synchronization protocol using a *Clock Gate* mechanism.
    """

    def __init__(self, simulator: Simulator, nats_prefix: str):
        """
        Args:
            simulator: A simulator with thread-safe event pool.
            nats_prefix: The prefix of NATS subjects.

        The NATS server URL is specified via ``NATS_URL`` environment variable, defaults to ``nats://127.0.0.1:4222``.
        Multiple servers may be specified with ``,`` delimiter.
        """
        if not nats:
            raise TypeError(
                "ClassicBridge requires nats-py optional dependency. Please install with: pip install 'nats-py>=2.16.0,<3'"
            )

        self.simulator: Final = simulator
        self.bridges: Final[dict[str, ClassicBridge]] = {}
        self.nats_servers: Final = os.getenv("NATS_URL", "nats://127.0.0.1:4222").split(",")
        self.nats_prefix: Final = nats_prefix

        self._gate_subject = f"{nats_prefix}.O._.gate"
        simulator.set_gate_reached_handler(lambda t: self.queue.put(_GateReached(t)))

        self._last_t = simulator.ts.slot
        self._last_gate_event: Final = EventHandleSlot()

        self.queue: Final = queue.Queue[_TransmittableItem](maxsize=4096)
        self._worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker_thread.start()

    def _worker_loop(self):
        asyncio.run(self._worker_run())

    async def _worker_run(self):
        assert nats
        try:
            nc = await nats.connect(self.nats_servers, connect_timeout=10, allow_reconnect=True)
            log.info("ClassicConnector connected to NATS server %s as client_id=%s", nc.connected_url, nc.client_id)
            async with nc:
                self._js = nc.jetstream()
                self._sub = await self._js.pull_subscribe(f"{self.nats_prefix}.I.*.*")

                done, _ = await asyncio.wait(
                    [
                        asyncio.create_task(self._tx_loop()),
                        asyncio.create_task(self._rx_loop()),
                    ],
                    return_when=asyncio.FIRST_EXCEPTION,
                )

                for task in done:
                    if exception := task.exception():
                        raise exception

        except Exception as e:
            log.error("NATS Worker Thread failed: %s", e)
            self.simulator.stop()

    async def _tx_loop(self):
        while self.simulator.running:
            try:
                cbp = await asyncio.to_thread(self.queue.get, timeout=0.5)
            except queue.Empty:
                continue

            await self._tx(cbp)

            await asyncio.sleep(0)

    async def _tx(self, item: _TransmittableItem):
        from nats.js.errors import BadRequestError  # noqa: PLC0415

        subject_suffix, payload, headers = item.encode()
        subject = f"{self.nats_prefix}.O.{subject_suffix}"

        try:
            ack = await self._js.publish(subject, payload, headers=headers)
            _ = ack.seq
        except BadRequestError:
            log.error("JetStream Error: Subject %s does not match a stream.", subject)
        except Exception as e:
            log.error("Failed to publish to JetStream: %s", e)
        finally:
            self.queue.task_done()

    async def _rx_loop(self):
        from nats.errors import TimeoutError  # noqa: PLC0415

        while self.simulator.running:
            try:
                msgs = await self._sub.fetch(batch=10, timeout=0.5)
            except TimeoutError:
                continue

            for msg in msgs:
                subject = msg.subject.split(".")
                if headers := msg.headers:
                    self._rx(subject[-2], subject[-1], headers, msg.data)
                await msg.ack()

            await asyncio.sleep(0)

    def _rx(self, dst: str, src: str, headers: dict[str, str], payload: bytes):
        t = int(headers.get("t", 0))
        if t < self._last_t:
            raise RuntimeError(f"ClassicConnector received timestamp {t} earlier than previous timestamp {self._last_t}")
        self._last_t = t

        if dst == "_":
            t = self.simulator.time(slot=t)
            match src:
                case "gate":
                    self._last_gate_event.set(self.simulator.update_gate(t))
                case "stop":
                    event = func_to_event(t, self.simulator.stop)
                    event.name = "ClassicConnector.stop"
                    event.priority = 0xFFFFFFFF
                    self.simulator.sched(event)
                    self._last_gate_event.set(self.simulator.update_gate(t))
                case _:
                    log.error("ClassicConnector received unexpected special subject ._.%s", src)
            return

        if not (bridge := self.bridges.get(src)):
            return

        cbp = _ClassicBridgePacket(
            t=t,
            src=src,
            dst=dst,
            payload=payload,
            is_json=headers.get("fmt") == "json",
        )
        self.simulator.sched(func_to_event(self.simulator.time(slot=cbp.t), bridge.inject, cbp))


class ClassicBridge(Application[Node]):
    """
    ClassicBridge delivers classic packets received on a simulated node to an external program
    and enables the external program to inject classic packets back into the simulated network.

    When a simulated node has a ClassicBridge installed, it is called a *split node*.
    In a *split node*, the external program takes over the application logic of that node,
    while the ClassicBridge acts as the network interface in the simulation world.
    """

    DEFAULT_NATS_PREFIX = "mqns.classicbridge"

    def __init__(self, *, nats_prefix=DEFAULT_NATS_PREFIX):
        """
        Args:
            nats_prefix: The prefix of NATS subjects.
                         If multiple ClassicBridges exist in the same simulator, they must have the same ``nats_prefix``.
        """
        super().__init__()
        self.nats_prefix: Final = nats_prefix

    @override
    def install(self, node) -> None:
        """
        Install ClassicBridge and turn a non-quantum node into a *split node*.
        """
        self._application_install(node, Node)
        try:
            self.conn: ClassicConnector = getattr(self.simulator, "_classic_connector")
        except AttributeError:
            self.conn = ClassicConnector(self.simulator, self.nats_prefix)
            setattr(self.simulator, "_classic_connector", self.conn)
        if self.conn.nats_prefix != self.nats_prefix:
            raise ValueError("every ClassicBridge in a scenario must have the same nats_prefix setting")
        self.conn.bridges[self.node.name] = self

    @event_handler
    def handle_packet(self, event: RecvClassicPacket) -> None:
        pkt = event.packet
        if pkt.dest is not self.node:
            return
        event.cancel()

        cbp = _ClassicBridgePacket(
            t=event.t.slot,
            src=pkt.src.name,
            dst=self.node.name,
            payload=pkt.encode(),
            is_json=pkt.is_json,
        )

        try:
            self.conn.queue.put_nowait(cbp)
        except queue.Full:
            self.log_error("drop packet from %s | %s", cbp.src, pkt.msg)

    def inject(self, cbp: _ClassicBridgePacket) -> None:
        dst = self.node.network.get_node(cbp.dst)
        pkt = ClassicPacket(cbp.payload, src=self.node, dest=dst)
        pkt.is_json = cbp.is_json
        self.log_debug("inject packet to %s | %s", cbp.dst, pkt.msg)
        self.node.send_cpacket(dst, pkt)
