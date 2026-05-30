"""Static Viser scene for a thesis overview of the six supported tools."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Dict, Sequence

import tyro

from dextoolbench.llm_supported_objects import SUPPORTED_LLM_OBJECT_FAMILIES
from dextoolbench.objects import NAME_TO_OBJECT
from laptop.utils import log_info

TOOL_SET_Z = 0.56
TOOL_SET_ROWS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("hammer", SUPPORTED_LLM_OBJECT_FAMILIES["hammer"]),
    ("screwdriver", SUPPORTED_LLM_OBJECT_FAMILIES["screwdriver"]),
)
TOOL_SET_MESH_COLOR_OVERRIDES = {
    "cuboid_hammer_v014": (125, 135, 155, 1.0),
    "cylinder_screwdriver_v3009": (70, 95, 130, 1.0),
}

DEFAULT_TOOL_SET_CAMERA_POSITION = (0.0, -0.08, 1.08)
DEFAULT_TOOL_SET_CAMERA_LOOK_AT = (0.0, 0.0, TOOL_SET_Z)


@dataclass(frozen=True)
class ToolPlacement:
    """One deterministic tabletop placement for a displayed tool."""

    object_name: str
    family: str
    position: tuple[float, float, float]
    yaw_rad: float

    @property
    def wxyz(self) -> tuple[float, float, float, float]:
        """Return the Viser WXYZ quaternion for this placement's yaw."""
        half_yaw = 0.5 * self.yaw_rad
        return (math.cos(half_yaw), 0.0, 0.0, math.sin(half_yaw))


@dataclass
class ToolSetVisArgs:
    """CLI args for the six-tool thesis overview viewer."""

    port: int = 8086
    """Viser server port for the tool-set overview browser."""

    startup_only: bool = False
    """Initialize the scene and exit immediately; useful for smoke tests."""

    show_labels: bool = False
    """Show small object-name labels below each tool."""


# Return the stable 3-by-2 family-grid tool placements used by the overview scene.
def tool_set_placements(
    *,
    columns_x: Sequence[float] = (-0.205, 0.0, 0.205),
    rows_y: Sequence[float] = (0.18, -0.17),
    z: float = TOOL_SET_Z,
) -> tuple[ToolPlacement, ...]:
    """Return deterministic tabletop placements for the supported hammer/screwdriver tools."""
    if len(columns_x) != 3:
        raise ValueError("Tool-set visualization requires exactly three column x positions.")
    if len(rows_y) != 2:
        raise ValueError("Tool-set visualization requires exactly two row y positions.")

    placements: list[ToolPlacement] = []
    for row_index, (family, object_names) in enumerate(TOOL_SET_ROWS):
        for column_index, object_name in enumerate(object_names):
            yaw_rad = math.pi / 2.0
            row_y = float(rows_y[row_index])
            if object_name == "claw_hammer":
                row_y += 0.025
            placements.append(
                ToolPlacement(
                    object_name=object_name,
                    family=family,
                    position=(
                        float(columns_x[column_index]),
                        row_y,
                        float(z),
                    ),
                    yaw_rad=yaw_rad,
                )
            )
    return tuple(placements)


class ToolSetVisApp:
    """Static six-tool tabletop overview for thesis screenshots."""

    # Initialize the six-tool static Viser scene.
    def __init__(self, args: ToolSetVisArgs) -> None:
        """Create hovering supported tool meshes and a screenshot-friendly camera."""
        try:
            import viser  # type: ignore
            from viser.extras import ViserUrdf  # type: ignore
        except Exception as exc:  # pragma: no cover - depends on optional package/runtime
            raise RuntimeError(
                "Tool-set visualization requested but the viser package is unavailable."
            ) from exc

        self._args = args
        self._viser_urdf = ViserUrdf
        self.server = viser.ViserServer(port=args.port)
        self.placements = tool_set_placements()
        self._tool_frames: Dict[str, object] = {}

        @self.server.on_client_connect
        def _(client) -> None:
            client.camera.position = DEFAULT_TOOL_SET_CAMERA_POSITION
            client.camera.look_at = DEFAULT_TOOL_SET_CAMERA_LOOK_AT

        self._setup_scene()

    # Add all supported tool URDFs to the scene.
    def _setup_scene(self) -> None:
        """Populate the static tool-set overview scene."""
        self.server.scene.add_frame("/world", show_axes=False)
        self.server.scene.add_grid(
            "/grid",
            width=0.6,
            height=0.5,
            cell_size=0.05,
            visible=False,
        )

        for placement in self.placements:
            root_node_name = f"/tools/{placement.family}/{placement.object_name}"
            frame = self.server.scene.add_frame(
                root_node_name,
                position=placement.position,
                wxyz=placement.wxyz,
                show_axes=False,
            )
            self._tool_frames[placement.object_name] = frame
            mesh_color_override = TOOL_SET_MESH_COLOR_OVERRIDES.get(placement.object_name)
            if mesh_color_override is None:
                self._viser_urdf(
                    self.server,
                    NAME_TO_OBJECT[placement.object_name].urdf_path,
                    root_node_name=root_node_name,
                )
            else:
                self._viser_urdf(
                    self.server,
                    NAME_TO_OBJECT[placement.object_name].urdf_path,
                    root_node_name=root_node_name,
                    mesh_color_override=mesh_color_override,
                )
            if self._args.show_labels:
                self.server.scene.add_label(
                    f"{root_node_name}/label",
                    text=placement.object_name.replace("_", " "),
                    position=(0.0, -0.09, 0.025),
                )

        self.server.gui.add_markdown("# Supported Tool Set")
        self.server.gui.add_markdown(
            "Top row: hammers. Bottom row: screwdrivers. Tools hover for a clean screenshot."
        )

    # Run the static viewer indefinitely.
    def run_forever(self) -> None:
        """Serve the tool-set overview scene until interrupted."""
        log_info(f"Tool-set visualizer running at {getattr(self.server, 'url', '')}")
        while True:  # pragma: no cover - interactive runtime
            time.sleep(1.0)


# Parse CLI args and run the six-tool thesis overview viewer.
def main() -> None:
    """Entry point for the six-tool thesis overview viewer."""
    app = ToolSetVisApp(tyro.cli(ToolSetVisArgs))
    if app._args.startup_only:
        log_info("Tool-set visualizer startup completed.")
        return
    app.run_forever()


if __name__ == "__main__":
    main()
