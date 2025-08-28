from typing import Literal, TypedDict

try:
    from typing import NotRequired
except ImportError:
    from typing_extensions import NotRequired


MultiplexingVector = list[tuple[int, int]]


class PathInstructions(TypedDict):
    req_id: int
    """
    Request identifier: nonnegative integer to uniquely identify the src-dst pair within the network.
    """

    route: list[str]
    """
    Path vector: a list of node names, in the order they appear in the path.

    There must a qchannel and a cchannel between adjacent nodes.
    """

    swap: list[int]
    """
    Swap sequence: nonnegative integers to control swapping order.

    This list shall have the same length as `route`.
    Each element represents swapping rank of the corresponding node.
    A node with smaller rank shall perform swapping before a node with larger rank.
    """

    m_v: NotRequired[MultiplexingVector]
    """
    Multiplexing vector, used in buffer-space multiplexing scheme only.

    This list shall have one element per qchannel, i.e. one less than `route`.
    Each element is a pair of nonnegative integers, corresponding to left and right qchannels.
    Each integer indicates how many memory qubits shall be allocated on the left/right qchannel for this path.
    If an integer is zero, it means allocating all qubits assigned to that qchannel for this path.
    """

    purif: dict[str, int]
    """
    Purification scheme.

    Each key is a segment name consists of two node names concatenated with a hyphen ("-"),
    where the nodes appear in the same order as in the route but do not have to be adjacent.
    Each value is an integer of the required rounds of purification at this segment.
    The default for every segment is zero i.e. no purification is performed.
    """


def make_path_instructions(
    req_id: int,
    route: list[str],
    swap: list[int],
    m_v: MultiplexingVector | None,
    purif: dict[str, int],
) -> PathInstructions:
    instructions: PathInstructions = {
        "req_id": req_id,
        "route": route,
        "swap": swap,
        "purif": purif,
    }
    if m_v is not None:
        instructions["m_v"] = m_v

    validate_path_instructions(instructions)
    return instructions


def validate_path_instructions(instructions: PathInstructions) -> None:
    def check_purif_segment(segment_name: str) -> bool:
        try:
            idx0, idx1 = [route.index(node_name) for node_name in segment_name.split("-")]
            return idx0 < idx1
        except ValueError:
            return False

    route = instructions["route"]
    if len(route) != len(instructions["swap"]) or len(route) == 0:
        raise ValueError("swapping order does not match route length")

    if "m_v" in instructions and len(instructions["m_v"]) != len(route) - 1:
        raise ValueError("multiplexing vector does not match route length")

    for segment_name in instructions["purif"].keys():
        if not check_purif_segment(segment_name):
            raise ValueError(f"purif segment {segment_name} does not exist in route")


class InstallPathMsg(TypedDict):
    cmd: Literal["install_path"]
    path_id: int
    instructions: PathInstructions


class PurifMsgBase(TypedDict):
    path_id: int
    purif_node: str
    partner: str
    epr: str
    measure_epr: str
    round: int


class PurifSolicitMsg(PurifMsgBase):
    cmd: Literal["PURIF_SOLICIT"]


class PurifResponseMsg(PurifMsgBase):
    cmd: Literal["PURIF_RESPONSE"]
    result: bool


class SwapUpdateMsg(TypedDict):
    cmd: Literal["SWAP_UPDATE"]
    path_id: int
    swapping_node: str
    partner: str
    epr: str
    new_epr: str | None  # None means swapping failed
