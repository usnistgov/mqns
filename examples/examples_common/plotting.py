import os
from collections.abc import Mapping, Sequence
from typing import cast, overload

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure, SubFigure

want_transparent = os.getenv("MQNS_PLTTRANSPARENT", "1") != "0"
"""
Whether to save figures as transparent image.
Default is yes.
Change to opaque image with MQNS_PLTTRANSPARENT=0 environment variable.
"""

want_show = os.getenv("MQNS_PLTSHOW", "1") != "0"
"""
Whether to display the plot on GUI systems.
Default is yes.
Disable the display window with MQNS_PLTSHOW=0 environment variable.
"""

type Axes1D = Sequence[Axes]
"""
1-dimensional array of Axes.
"""

type Axes2D = Mapping[tuple[int, int], Axes]
"""
2-dimensional array of Axes.
"""

type SubFigure1D = Sequence[SubFigure]
"""
1-dimensional array of SubFigure.
"""

type SubFigure2D = Mapping[tuple[int, int], SubFigure]
"""
2-dimensional array of SubFigure.
"""


@overload
def plt_save(save_to: str, /, **kwargs) -> None:
    """
    Save current plot to file if requested.

    Args:
        save_to: Output filename; empty string skips saving.
    """
    pass


@overload
def plt_save(*save_to: tuple[Figure, str], **kwargs) -> None:
    """
    Save figures to files if requested.

    Args:
        save_to: Pairs of figure and output filename; empty string skips saving.
    """
    pass


def plt_save(*save_to: str | tuple[Figure, str], **kwargs) -> None:
    for item in save_to:
        fig, filename = (plt, item) if isinstance(item, str) else item
        if filename:
            cast(Figure, fig).savefig(filename, dpi=300, transparent=want_transparent, **kwargs)

    if want_show:
        plt.show()


__all__ = [
    "Axes",
    "Axes1D",
    "Axes2D",
    "Figure",
    "mpl",
    "plt_save",
    "plt",
    "SubFigure",
    "SubFigure1D",
    "SubFigure2D",
]
