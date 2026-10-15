from tap import Tap

from mqns.entity.qchannel import LINK_ARCH_MAP, LinkArchLiteral
from mqns.models.epr import EPR_TYPE_MAP, EprTypeLiteral
from mqns.models.error import TimeDecayInput
from mqns.models.error.input import ErrorModelInputBasic, ErrorModelInputLength, ErrorModelInputTime
from mqns.network.proactive import MUX_SCHEME_MAP, MuxSchemeLiteral


def tap_configure(tap: Tap) -> None:
    """
    When called from ``Tap.configure()`` function, define command line arguments for supported literal types.

    Recognized keys:

    * ``mode``
    * ``sync_timing``

    Recognized types:

    * ``EprTypeLiteral``
    * ``ErrorModelInput*``
    * ``LinkArchLiteral``
    * ``MuxSchemeLiteral``
    * ``TimeDecayInput``
    """
    for key, typ in tap._annotations.items():
        if key == "mode":
            tap.add_argument(
                f"--{key}",
                help=f"(default={getattr(tap, key)}) choose mode: [P]roactive/[R]eactive forwarding, "
                "[C]entralized/[D]istributed control, [A]sync/[S]ync timing",
            )
        elif key == "sync_timing":
            dflt = getattr(tap, key, [])
            dflt_desc = f"default={dflt}" if dflt else "default: derive from t_cohere"
            tap.add_argument(
                f"--{key}",
                type=float,
                nargs=3,
                default=dflt,
                metavar=("t_ext", "t_rtg", "t_int"),
                help=f"(3*float, {dflt_desc}) SYNC timing mode phase durations in seconds",
            )
        elif typ is EprTypeLiteral:
            tap.add_argument(f"--{key}", type=str, default="W", choices=EPR_TYPE_MAP.keys())
        elif typ is ErrorModelInputBasic:
            tap.add_argument(f"--{key}", type=str, metavar="ErrorModelType:p_error")
        elif typ in (ErrorModelInputTime, ErrorModelInputLength, TimeDecayInput):
            tap.add_argument(f"--{key}", type=str, metavar="ErrorModelType:rate")
        elif typ is MuxSchemeLiteral:
            tap.add_argument(f"--{key}", type=str, choices=MUX_SCHEME_MAP.keys())
        elif typ is LinkArchLiteral:
            tap.add_argument(f"--{key}", type=str, default="DIM-BK-SeQUeNCe", choices=LINK_ARCH_MAP.keys())
