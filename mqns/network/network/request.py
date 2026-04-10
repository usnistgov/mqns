from typing import TypedDict, Unpack

from mqns.entity.node import QNode


class RequestAttr(TypedDict, total=False):
    """
    Request attributes.

    This is currently empty.
    In the future, it may contain properties such as minimum fidelity and desired throughput.
    """


class Request:
    """Requests entanglement pairs between a source and a destination."""

    def __init__(self, src: QNode, dst: QNode, **attr: Unpack[RequestAttr]):
        """
        Args:
            src: Left node to receive one of the entangled qubits.
            dst: Right node to receive one of the entangled qubits.
        """
        self.src = src
        self.dst = dst
        _ = attr

    def __repr__(self) -> str:
        return f"<Request {self.src}-{self.dst}>"
