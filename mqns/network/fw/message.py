from typing import Literal, NotRequired, TypedDict

from mqns.simulator import Time

type SwapSequence = list[int]
type MultiplexingVector = list[tuple[int, int]]


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

    swap: SwapSequence
    """
    Swap sequence: nonnegative integers to control swapping order.

    This list shall have the same length as `route`.
    Each element represents swapping rank of the corresponding node.
    A node with smaller rank shall perform swapping before a node with larger rank.

    To disable swapping, set this to a list of zeros.
    When swapping is disabled, the forwarder will consume entanglement upon completing purification,
    without attempting entanglement swapping.
    """

    swap_cutoff: list[int]
    """
    Swap cutoff time: maximum age at each swapping step.

    This list shall have the same length as `swap`.
    The i-th element corresponds to the i-th node in the `route` list.
    Each element is a duration in time_slot unit (see `Time` class); `-1` means no restriction.

    The semantics of "age" depend on the CutoffScheme passed to ProactiveForwarder.
    Since the first and last nodes in `route` do not perform swapping, the first and last elements
    in this list have no effect. Likewise, if swapping has been disabled, this list has no effect.
    """

    m_v: NotRequired[MultiplexingVector]
    """
    Multiplexing vector, used in buffer-space multiplexing scheme only.

    This list shall have one element per qchannel, i.e. one less than `route`.
    Each element is a pair of nonnegative integers, corresponding to left and right qchannels.
    Each integer indicates how many memory qubits shall be allocated on the left/right qchannel for this path.
    If an integer is zero, it means allocating all qubits assigned to that qchannel for this path.

    Example:
        route = [S,    R,     D]
        m_v   = [ (4,2), (3,0) ]

        S should allocate 4 qubits on S-R channel.
        R should allocate 2 qubits on S-R channel and 3 qubits on R-D channel.
        D should allocate all qubits assigned to R-D channel.
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
    swap: SwapSequence,
    swap_cutoff: list[Time | None] | None,
    m_v: MultiplexingVector | None,
    purif: dict[str, int],
) -> PathInstructions:
    instructions: PathInstructions = {
        "req_id": req_id,
        "route": route,
        "swap": swap,
        "swap_cutoff": [-1] * len(swap),
        "purif": purif,
    }
    if swap_cutoff is not None:
        instructions["swap_cutoff"] = [-1 if t is None else t.time_slot for t in swap_cutoff]
    if m_v is not None:
        instructions["m_v"] = m_v

    validate_path_instructions(instructions)
    return instructions


def validate_path_instructions(instructions: PathInstructions) -> None:
    def check_purif_segment(segment_name: str) -> bool:
        try:
            idx0, idx1 = (route.index(node_name) for node_name in segment_name.split("-"))
            return idx0 < idx1
        except ValueError:
            return False

    route = instructions["route"]
    if len(route) == 0:
        raise ValueError("route is empty")

    if len(instructions["swap"]) != len(route):
        raise ValueError("swapping order does not match route length")

    if "swap_cutoff" in instructions and len(instructions["swap_cutoff"]) != len(route):
        raise ValueError("swap_cutoff does not match swapping order length")

    if "m_v" in instructions and len(instructions["m_v"]) != len(route) - 1:
        raise ValueError("multiplexing vector does not match route length")

    for segment_name in instructions["purif"].keys():
        if not check_purif_segment(segment_name):
            raise ValueError(f"purif segment {segment_name} does not exist in route")


class InstallPathMsg(TypedDict):
    cmd: Literal["install_path"]
    path_id: int
    instructions: PathInstructions


class UninstallPathMsg(TypedDict):
    cmd: Literal["uninstall_path"]
    path_id: int


class CutoffDiscardMsg(TypedDict):
    cmd: Literal["CUTOFF_DISCARD"]
    path_id: int
    epr: str
    round: int


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
