from mqns.network.network import QuantumNetwork
from mqns.network.protocol import LinkLayer


def gather_etg_decoh(net: QuantumNetwork) -> tuple[int, int, float]:
    """
    Gather LinkLayer entanglement and decoherence counters.

    Returns:
      [0]: total entanglement counts.
      [1]: total decoherence counts.
      [2]: decoherence ratio.
    """
    total_etg = 0
    total_decohered = 0
    for node in net.nodes:
        for ll_app in node.get_apps(LinkLayer):
            total_etg += ll_app.cnt.n_etg
            total_decohered += ll_app.cnt.n_decoh
    decoh_ratio = total_decohered / total_etg if total_etg > 0 else 0
    return total_etg, total_decohered, decoh_ratio
