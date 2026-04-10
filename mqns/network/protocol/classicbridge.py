import asyncio
import os
import queue
import threading
from dataclasses import dataclass
from typing import Protocol, override

from mqns.entity.cchannel import ClassicPacket, RecvClassicPacket
from mqns.entity.node import Application, Node
from mqns.simulator import Event, Simulator, func_to_event
from mqns.utils import log

try:
    import nats
except ImportError:
    nats = None


class TransmittableItem(Protocol):
    def encode(self) -> tuple[str, bytes, dict[str, str]]: ...


@dataclass
class GateReached(TransmittableItem):
    t: int

    def encode(self) -> tuple[str, bytes, dict[str, str]]:
        return "_.gate", b"", {"t": f"{self.t}"}


@dataclass
class ClassicBridgePacket(TransmittableItem):
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
    ClassicConnector is a singleton (one instance per ``Simulator``) that facilitates synchronized communication,
    via NATS JetStream, between one or more *split nodes* (nodes with ``ClassicBridge``) in the simulation and
    an external program that implements the application logic of these nodes.
    It acts as a PDES (Parallel Discrete Event Simulation) bridge, implementing a conservative synchronization protocol
    using a *Clock Gate* mechanism.

    This connector is configured via:

    * ``NATS_URL`` environment variable specifies one or more NATS servers, defaults to ``nats://localhost:4222``.
    * ``nats_prefix`` constructor argument specifies the prefix of NATS subjects.

    When a *split node* receives a classic packet, it is published as:

    * Subject: ``<PREFIX>.O.<dst>.<src>``
    * Payload: the classic packet payload
    * Header ``t``: packet receipt time in simulation time slots
    * Header ``fmt``: packet format, either ``"json"`` or ``"bytes"``

    If the external program wants to transmit a packet out of a node *src*, it should publish:

    * Subject: ``<PREFIX>.I.<dst>.<src>``
    * Payload: the classic packet payload
    * Header ``t``: packet transmission time in simulation time slots
    * Header ``fmt``: packet format, either ``"json"`` or ``"bytes"``

    To maintain temporal consistency, this connector must be used on a ``Simulator`` with thread-safe event pool,
    which includes a clock gate feature that prevents the simulation from going beyond a clock gate time slot.
    The simulator will not advance its internal clock beyond the current Clock Gate (initially zero), effectively
    pausing the simulation at the first time slot until an external gate update is received.
    The external program is responsible for advancing this gate by publishing a heartbeat:

    * Subject: ``<PREFIX>.I._.gate`` (note: ``_`` is a keyword for ClassicConnector and cannot be used as a node name)
    * Header ``t``: new upper-bound in simulation time slots

    When the simulator reaches the gate, it would notify the external program by publishing:

    * Subject: ``<PREFIX>.O._.gate``
    * Header ``t``: stop time in simulation time slots

    The external program may stop the simulator by publishing:

    * Subject: ``<PREFIX>.I._.stop``
    * Header ``t``: stop time in simulation time slots

    Publishing the stop message implies releasing the gate to the stop time.
    However, the simulator will not publish ``<PREFIX>.O._.gate`` before stopping.

    The external program must publish all packets and gate updates in non-decreasing order. In other words,
    the header ``t`` for every message must be greater than or equal to the timestamp on all previous messages.
    Once the clock moves forward, any incoming messages with a timestamp strictly smaller than a previous message
    will be considered causality violations, and could either crash the simulation or cause a deadlock.
    """

    def __init__(self, simulator: Simulator, nats_prefix: str):
        if not nats:
            raise TypeError(
                "ClassicBridge requires nats-py optional dependency. Please install with: pip install 'nats-py>=2.14.0,<3'"
            )

        self.simulator = simulator
        self.bridges: dict[str, "ClassicBridge"] = {}
        self.nats_servers = os.getenv("NATS_URL", "nats://127.0.0.1:4222").split(",")
        self.nats_prefix = nats_prefix

        self._gate_subject = f"{nats_prefix}.O._.gate"
        simulator.set_gate_reached_handler(lambda t: self.queue.put(GateReached(t)))

        self._last_t = 0
        self._last_gate_event: Event | None = None

        self.queue = queue.Queue[TransmittableItem](maxsize=4096)
        self._worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker_thread.start()

    def _worker_loop(self):
        asyncio.run(self._worker_run())

    async def _worker_run(self):
        assert nats
        try:
            nc = await nats.connect(self.nats_servers, connect_timeout=10, allow_reconnect=True)
            log.info(f"ClassicConnector connected to NATS server {nc.connected_url} as client_id={nc.client_id}")
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
            log.error(f"NATS Worker Thread failed: {e}")
            self.simulator.stop()

    async def _tx_loop(self):
        while self.simulator.running:
            try:
                cbp = await asyncio.to_thread(self.queue.get, timeout=0.5)
            except queue.Empty:
                continue

            await self._tx(cbp)

            await asyncio.sleep(0)

    async def _tx(self, item: TransmittableItem):
        from nats.js.errors import BadRequestError  # noqa: PLC0415

        subject_suffix, payload, headers = item.encode()
        subject = f"{self.nats_prefix}.O.{subject_suffix}"

        try:
            ack = await self._js.publish(subject, payload, headers=headers)
            _ = ack.seq
        except BadRequestError:
            log.error(f"JetStream Error: Subject {subject} does not match a stream.")
        except Exception as e:
            log.error(f"Failed to publish to JetStream: {e}")
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
            t = self.simulator.time(time_slot=t)
            match src:
                case "gate":
                    if self._last_gate_event:
                        self._last_gate_event.cancel()
                    self._last_gate_event = self.simulator.update_gate(t)
                case "stop":
                    event = func_to_event(t, self.simulator.stop)
                    event.name = "ClassicConnector.stop"
                    event.priority = 0xFFFFFFFF
                    self.simulator.add_event(event)
                    self.simulator.update_gate(t)
                case _:
                    log.error(f"ClassicConnector received unexpected special subject ._.{src}")
            return

        if not (bridge := self.bridges.get(src)):
            return

        cbp = ClassicBridgePacket(
            t=t,
            src=src,
            dst=dst,
            payload=payload,
            is_json=headers.get("fmt") == "json",
        )
        self.simulator.add_event(func_to_event(self.simulator.time(time_slot=cbp.t), bridge.inject, cbp))


class ClassicBridge(Application[Node]):
    """
    ClassicBridge delivers classic packets received on a simulated node to an external program
    and enables the external program to inject classic packets back into the simulated network.

    When a simulated node (typically a ``Controller`` but it could be any non-quantum node) has
    a ClassicBridge installed, it is called a *split node*.
    In a *split node*, the external program takes over the application logic of that node,
    while the ClassicBridge acts as the network interface in the simulation world.
    """

    DEFAULT_NATS_PREFIX = "mqns.classicbridge"

    def __init__(self, *, nats_prefix=DEFAULT_NATS_PREFIX):
        super().__init__()
        self.nats_prefix = nats_prefix
        self.add_handler(self.handle_packet, RecvClassicPacket)

    @override
    def install(self, node):
        self._application_install(node, Node)
        try:
            self.conn: ClassicConnector = getattr(self.simulator, "_classic_connector")
        except AttributeError:
            self.conn = ClassicConnector(self.simulator, self.nats_prefix)
            setattr(self.simulator, "_classic_connector", self.conn)
        if self.conn.nats_prefix != self.nats_prefix:
            raise ValueError("every ClassicBridge in a scenario must have the same nats_prefix setting")
        self.conn.bridges[self.node.name] = self

    def handle_packet(self, event: RecvClassicPacket) -> bool:
        pkt = event.packet
        if pkt.dest is not self.node:
            return False

        cbp = ClassicBridgePacket(
            t=event.t.time_slot,
            src=pkt.src.name,
            dst=self.node.name,
            payload=pkt.encode(),
            is_json=pkt.is_json,
        )

        try:
            self.conn.queue.put_nowait(cbp)
        except queue.Full:
            log.error(f"{self.node}: ClassicBridge drops packet from {cbp.src} | {pkt.msg}")

        return True

    def inject(self, cbp: ClassicBridgePacket) -> None:
        dst = self.node.network.get_node(cbp.dst)
        pkt = ClassicPacket(cbp.payload, src=self.node, dest=dst)
        pkt.is_json = cbp.is_json
        log.debug(f"{self.node}: ClassicBridge injects packet to {cbp.dst} | {pkt.msg}")
        self.node.send_cpacket(dst, pkt)
