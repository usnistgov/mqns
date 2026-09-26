from mqns.network.builder import NetworkBuilder

END_NODES = ("A", "B", "C", "D", "G", "H", "I", "K", "L", "M")
"""Nodes with degree=1."""


def define_topo(
    b: NetworkBuilder,
    *,
    ch_capacity: tuple[int, int],
) -> None:
    """
    Build a 13-node tree/star topology characterized by two highly contested central backbone routing links.

    .. figure:: /_static/examples/multiplexing.svg
        :alt: topology diagram
        :align: center
        :width: 100%

    Args:
        b: NetworkBuilder instance.
        ch_capacity: How many qubits per channel.
                     First integer applies to the side with a red dot in the diagram.
                     Second integer applies to the other side.
    """
    c0, c1 = ch_capacity
    b.topo(
        channels=[
            # left spokes -> E
            (("A", "E"), 30, (c0, c1)),
            (("B", "E"), 30, (c0, c1)),
            (("C", "E"), 30, (c0, c1)),
            (("D", "E"), 30, (c0, c1)),
            # middle trunks
            (("E", "F"), 30, (c1, c0)),
            (("F", "J"), 30, (c0, c1)),
            # right spokes from J
            (("J", "K"), 30, (c0, c1)),
            (("J", "L"), 30, (c0, c1)),
            (("J", "M"), 30, (c0, c1)),
            # bottom spokes from F
            (("G", "F"), 30, (c1, c0)),
            (("F", "H"), 30, (c0, c1)),
            (("F", "I"), 30, (c0, c1)),
        ],
        fiber_alpha=0.17,
        eta_s=0.8,
        eta_d=0.5,
        t_cohere=0.1,
    )
