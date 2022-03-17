from __future__ import annotations

from . import control

CONTROL_ARM = "control"

treatments = {
    CONTROL_ARM: control.arm,
    "x": control.arm,
    "y": control.arm,
    "z": control.arm,
}
