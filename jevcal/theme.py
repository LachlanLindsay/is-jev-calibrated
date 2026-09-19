"""Chart theme.

Two selected modes, not one flipped into the other: the dark steps are their own
values chosen against the dark surface. Swap the hexes here for a brand palette
and every chart follows.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Theme:
    name: str
    surface: str
    page: str
    ink: str
    ink_secondary: str
    ink_muted: str
    grid: str
    axis: str
    series_1: str
    series_2: str
    critical: str
    good: str

    @property
    def is_dark(self) -> bool:
        return self.name == "dark"


LIGHT = Theme(
    name="light",
    surface="#fcfcfb",
    page="#f9f9f7",
    ink="#0b0b0b",
    ink_secondary="#52514e",
    ink_muted="#898781",
    grid="#e1e0d9",
    axis="#c3c2b7",
    series_1="#2a78d6",
    series_2="#eb6834",
    critical="#d03b3b",
    good="#0ca30c",
)

DARK = Theme(
    name="dark",
    surface="#1a1a19",
    page="#0d0d0d",
    ink="#ffffff",
    ink_secondary="#c3c2b7",
    ink_muted="#898781",
    grid="#2c2c2a",
    axis="#383835",
    series_1="#3987e5",
    series_2="#d95926",
    critical="#d03b3b",
    good="#0ca30c",
)

THEMES = {"light": LIGHT, "dark": DARK}


def get(name: str) -> Theme:
    try:
        return THEMES[name]
    except KeyError as exc:
        raise ValueError(f"unknown theme {name!r}; expected one of {sorted(THEMES)}") from exc
