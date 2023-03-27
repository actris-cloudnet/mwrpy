"""Module for metadata of statistic plotting module."""
from typing import NamedTuple, Sequence


class StatMeta(NamedTuple):
    """Class for statistic plotting module."""

    name: str | None = None
    cbar: str | Sequence[str] | None = None
    clabel: str | Sequence[tuple[str, str]] | None = None
    ylabel: str | None = None
    plot_range: tuple[float, float] | None = None
    plot_type: str | None = None
    add_txt: bool | None = False


_COLORS = {
    "green": "#3cb371",
    "darkgreen": "#253A24",
    "lightgreen": "#70EB5D",
    "yellowgreen": "#C7FA3A",
    "yellow": "#FFE744",
    "orange": "#ffa500",
    "pink": "#B43757",
    "red": "#F57150",
    "shockred": "#E64A23",
    "seaweed": "#646F5E",
    "seaweed_roll": "#748269",
    "white": "#ffffff",
    "lightblue": "#6CFFEC",
    "blue": "#209FF3",
    "skyblue": "#CDF5F6",
    "darksky": "#76A9AB",
    "darkpurple": "#464AB9",
    "lightpurple": "#6A5ACD",
    "purple": "#BF9AFF",
    "darkgray": "#2f4f4f",
    "lightgray": "#ECECEC",
    "gray": "#d3d3d3",
    "lightbrown": "#CEBC89",
    "lightsteel": "#a0b0bb",
    "steelblue": "#4682b4",
    "mask": "#C8C8C8",
}


_CMAP_DA = [
    _COLORS["darksky"],
    _COLORS["purple"],
]

_CMAP_QF = [
    _COLORS["darkgray"],
    _COLORS["darkpurple"],
    _COLORS["orange"],
    _COLORS["shockred"],
    _COLORS["seaweed_roll"],
    _COLORS["blue"],
    _COLORS["yellow"],
    _COLORS["yellowgreen"],
]


ATTRIBUTES = {
    "data_availability": StatMeta(
        name="Data Availability (% of not flagged, # of scans)",
        cbar=_CMAP_DA,
        ylabel="Hours",
        plot_range=(0, 24 * 31),
        plot_type="bar",
        add_txt=True,
    ),
    "quality_flag": StatMeta(
        name="Quality Flags",
        cbar=_CMAP_QF,
        ylabel="Fraction",
        plot_range=(0, 1.0),
        plot_type="bar",
    ),
    "spectral_consistency": StatMeta(
        name="Spectral Consistency (% of simultaneous rain_detected)",
        ylabel="Frequency (GHz)",
        plot_type="heat",
    ),
    "receiver_temperature": StatMeta(
        name="Receiver Temperature",
        ylabel="Daily mean and range (K)",
        plot_type="lines_rec",
    ),
    "receiver_stability": StatMeta(
        name="Receiver 1 Stability (% above range)",
        ylabel="Mean absolute difference (K)",
        plot_type="box",
    ),
    "ambient_target": StatMeta(
        name="Ambient Target Temperature",
        ylabel="Daily mean and range of sensor mean (K)",
        plot_type="lines_amb",
    ),
}
