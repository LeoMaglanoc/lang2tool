"""CLI demo: wire MockLLMParamClient + GeometricPoseConverter end-to-end."""

from __future__ import annotations

import argparse
import json
import sys

from llm_runtime.goals.converter import GeometricPoseConverter
from llm_runtime.llm.generator import LLMParametricGoalGenerator
from llm_runtime.llm.mock_client import MockLLMParamClient


# Parse CLI arguments for task label and optional output path.
def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Return parsed CLI arguments."""
    parser = argparse.ArgumentParser(
        description="LLM goal-setting demo: instruction → GeometricGoalV1 → SE(3) sequence."
    )
    parser.add_argument(
        "--task",
        default="swing_down",
        help="Task label (or substring) to use as the user instruction.",
    )
    parser.add_argument(
        "--output-json",
        metavar="PATH",
        default=None,
        help="Optional path to write goals JSON compatible with fixedGoalStatesJsonPath.",
    )
    return parser.parse_args(argv)


# Run the full mock pipeline and print results.
def main(argv: list[str] | None = None) -> None:
    """Execute the end-to-end demo pipeline and optionally write goals JSON."""
    args = _parse_args(argv)

    # Wire components.
    client = MockLLMParamClient()
    converter = GeometricPoseConverter()
    generator = LLMParametricGoalGenerator(param_client=client, converter=converter)

    instruction = args.task
    print(f"\n[demo] Instruction: '{instruction}'")

    # Generate and display validated parameters.
    params = generator.generate_params(instruction)
    print("\n[demo] GeometricGoalV1 params:")
    print(f"  task_label              = {params.task_label}")
    print(f"  object_frame            = {params.object_frame}")
    print(f"  contact_point_object    = {params.contact_point_object}")
    print(f"  approach_direction_object = {params.approach_direction_object}")
    print(f"  tool_axis_object        = {params.tool_axis_object}")
    print(f"  pregrasp_offset_m       = {params.pregrasp_offset_m}")
    print(f"  grasp_depth_m           = {params.grasp_depth_m}")
    print(f"  lift_height_m           = {params.lift_height_m}")
    print(f"  timing_s                = {params.timing_s}")

    # Generate and display SE(3) sequence.
    sequence = generator.generate_pose_sequence(instruction)
    print(f"\n[demo] SE(3) pose sequence ({len(sequence)} waypoints):")
    for i, pose in enumerate(sequence):
        x, y, z, qx, qy, qz, qw = pose
        print(
            f"  wp{i + 1}: pos=({x:.4f}, {y:.4f}, {z:.4f})  "
            f"quat=({qx:.4f}, {qy:.4f}, {qz:.4f}, {qw:.4f})"
        )

    # Build JSON payload compatible with env fixedGoalStatesJsonPath.
    goals_payload = {"goals": [list(pose) for pose in sequence]}

    if args.output_json:
        with open(args.output_json, "w") as fh:
            json.dump(goals_payload, fh, indent=2)
        print(f"\n[demo] Goals written to: {args.output_json}")
    else:
        print("\n[demo] Goals JSON (pass --output-json PATH to save):")
        print(json.dumps(goals_payload, indent=2))


if __name__ == "__main__":
    main(sys.argv[1:])
