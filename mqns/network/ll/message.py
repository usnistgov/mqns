from typing import Literal, NamedTuple, TypedDict

from mqns.entity.cchannel import ClassicPacket


class ReserveMsg(TypedDict):
    """
    Message between LinkLayers to request and accept qubit reservation for entanglement generation.
    """

    cmd: Literal["RESERVE_REQ", "RESERVE_ABORT", "RESERVE_RES"]
    """
    * RESERVE_REQ -- pri-to-2nd, request entanglement generation.
    * RESERVE_ABORT -- pri-to-2nd, cancel the reservation and abort entanglement generation.
    * RESERVE_RES -- 2nd-to-pri, accept entanglement generation.
    """
    path_id: int | None
    """
    Path identifier assigned to the generated entanglement.
    """
    key: str
    """
    Reservation key that uniquely identifies this reservation and the resulting entanglement.
    """


class ReserveParsedMsg(NamedTuple):
    cmd: str
    src: str
    path_id: int | None
    key: str

    @staticmethod
    def new(pkt: ClassicPacket, msg: ReserveMsg) -> "ReserveParsedMsg":
        return ReserveParsedMsg(cmd=msg["cmd"], src=pkt.src.name, path_id=msg["path_id"], key=msg["key"])

    def __repr__(self) -> str:
        return f"{self.cmd}({self.src}:{self.path_id}, {self.key})"
