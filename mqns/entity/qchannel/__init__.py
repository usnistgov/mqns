from mqns.entity.qchannel.link_arch import ChannelParameters, LinkArch, LinkArchAlways, LinkArchParameters
from mqns.entity.qchannel.link_arch_dim import LinkArchDimBk, LinkArchDimBkSeq, LinkArchDimDual
from mqns.entity.qchannel.link_arch_input import LINK_ARCH_MAP, LinkArchInput, LinkArchLiteral, parse_link_arch
from mqns.entity.qchannel.link_arch_sim import LinkArchSim
from mqns.entity.qchannel.link_arch_sr import LinkArchSr
from mqns.entity.qchannel.qchannel import QuantumChannel, QuantumChannelInitKwargs, RecvQubitPacket

__all__ = [
    "ChannelParameters",
    "LINK_ARCH_MAP",
    "LinkArch",
    "LinkArchAlways",
    "LinkArchDimBk",
    "LinkArchDimBkSeq",
    "LinkArchDimDual",
    "LinkArchInput",
    "LinkArchLiteral",
    "LinkArchParameters",
    "LinkArchSim",
    "LinkArchSr",
    "parse_link_arch",
    "QuantumChannel",
    "QuantumChannelInitKwargs",
    "RecvQubitPacket",
]

for name in __all__:
    if name in (
        "LINK_ARCH_MAP",
        "LinkArchInput",
        "LinkArchLiteral",
    ):
        continue
    globals()[name].__module__ = __name__
