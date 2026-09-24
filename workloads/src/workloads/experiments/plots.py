"""Number formatting, palette and axis styling shared by experiment plots."""

SURFACE = "#fcfcfb"
BAR = "#2a78d6"
# Categorical slots 1-4 (blue, orange, aqua, yellow), assigned in this order.
# Aqua and yellow are 2.7:1 and 2.1:1 against SURFACE, so lines in these colors
# also carry direct labels.
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")
INK = "#0b0b0b"
SECONDARY_INK = "#52514e"
MUTED_INK = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"


def readable(value: float) -> str:
    for unit, scale in (("G", 1e9), ("M", 1e6), ("k", 1e3)):
        if value >= scale:
            scaled = value / scale
            # Not :.3g, which switches to exponent notation from 1000 up.
            if scaled >= 100:
                return f"{scaled:.0f} {unit}"
            digits = 1 if scaled >= 10 else 2
            return f"{scaled:.{digits}f}".rstrip("0").rstrip(".") + f" {unit}"
    return f"{value:.0f}"


def percent(share: float) -> str:
    if 0 < share < 0.001:
        return "<0.1%"
    return f"{share:.1%}"
