"""Unit tests for semantic pose ontology math helpers."""

from __future__ import annotations

import pytest

from llm_runtime.semantic_pose import (
    compute_semantic_quat_delta,
    get_object_pose_semantics_payload,
    quat_mul_xyzw,
    quat_rotate_xyzw,
)


# Verify upright rotates primary local X axis to world +Z for identity orientation.
def test_compute_semantic_quat_delta_upright_aligns_primary_axis_to_world_up() -> None:
    """Ensure upright delta aligns tool primary axis with world up."""
    q_identity = (0.0, 0.0, 0.0, 1.0)
    dq = compute_semantic_quat_delta("claw_hammer", q_identity, "upright")
    q_new = quat_mul_xyzw(dq, q_identity)
    aligned_axis = quat_rotate_xyzw(q_new, (1.0, 0.0, 0.0))

    assert pytest.approx(aligned_axis[0], abs=1e-5) == 0.0
    assert pytest.approx(aligned_axis[1], abs=1e-5) == 0.0
    assert pytest.approx(aligned_axis[2], abs=1e-5) == 1.0


# Verify face_table aligns the configured face normal toward world -Z.
def test_compute_semantic_quat_delta_face_table_aligns_face_normal_down() -> None:
    """Ensure face_table delta aligns local face-normal with world down."""
    q_identity = (0.0, 0.0, 0.0, 1.0)
    dq = compute_semantic_quat_delta("flat_spatula", q_identity, "face_table")
    q_new = quat_mul_xyzw(dq, q_identity)
    face_normal_world = quat_rotate_xyzw(q_new, (0.0, 0.0, 1.0))

    assert pytest.approx(face_normal_world[0], abs=1e-5) == 0.0
    assert pytest.approx(face_normal_world[1], abs=1e-5) == 0.0
    assert pytest.approx(face_normal_world[2], abs=1e-5) == -1.0


# Verify metadata payload exposes ontology labels expected by tool-calling.
def test_get_object_pose_semantics_payload_includes_core_ontology_targets() -> None:
    """Ensure payload lists all core ontology target labels."""
    payload = get_object_pose_semantics_payload("claw_hammer")
    assert payload["quaternion_convention"] == "xyzw"
    assert payload["semantic_targets"] == [
        "upright",
        "flat",
        "head_down",
        "tip_forward",
        "face_table",
    ]
    assert payload["axes_local"]["head_axis"] == [1.0, 0.0, 0.0]
    assert payload["axes_local"]["tip_axis"] == [-1.0, 0.0, 0.0]
    assert payload["axes_local"]["strike_face_normal"] == [0.0, -1.0, 0.0]
    assert payload["points_local"]["head_center"] == pytest.approx([0.123122, 0.0, 0.001475])
    assert payload["points_local"]["strike_point"] == pytest.approx([0.123122, -0.038279, 0.001475])
    assert payload["points_local"]["swing_support_point"] == pytest.approx(
        [-0.0523742, 0.00921726, -0.00364191]
    )


# Verify long screwdriver exposes explicit non-default semantic axes and points.
def test_get_object_pose_semantics_payload_exposes_long_screwdriver_axes() -> None:
    """Long screwdriver payload should expose explicit semantic axes instead of the default stub."""
    payload = get_object_pose_semantics_payload("long_screwdriver")

    assert payload["axes_local"]["primary_axis"] == [1.0, 0.0, 0.0]
    assert payload["axes_local"]["head_axis"] == [-1.0, 0.0, 0.0]
    assert payload["axes_local"]["tip_axis"] == [1.0, 0.0, 0.0]
    assert payload["axes_local"]["strike_face_normal"] == [1.0, 0.0, 0.0]
    assert payload["points_local"]["strike_point"] == pytest.approx(
        [0.19718152, -0.00140855, -0.0027986]
    )
    assert payload["points_local"]["swing_support_point"] == pytest.approx(
        [0.19718152, -0.00140855, -0.0027986]
    )


# Verify the newly supported hammer/screwdriver instances expose explicit family-specific semantics.
def test_get_object_pose_semantics_payload_exposes_new_supported_object_axes() -> None:
    """Mallet and short screwdriver should expose non-default family-specific semantic axes."""
    mallet_payload = get_object_pose_semantics_payload("mallet_hammer")
    short_payload = get_object_pose_semantics_payload("short_screwdriver")

    assert mallet_payload["axes_local"]["tip_axis"] == [-1.0, 0.0, 0.0]
    assert mallet_payload["axes_local"]["strike_face_normal"] == [0.0, -1.0, 0.0]
    assert mallet_payload["points_local"]["strike_point"] == pytest.approx(
        [0.19395672, -0.0462501, -0.02061698]
    )
    assert mallet_payload["points_local"]["swing_support_point"] == pytest.approx(
        [-0.06529985, 0.00319191, -0.0122853]
    )
    assert short_payload["axes_local"]["head_axis"] == [-1.0, 0.0, 0.0]
    assert short_payload["axes_local"]["strike_face_normal"] == [1.0, 0.0, 0.0]
    assert short_payload["points_local"]["strike_point"] == pytest.approx(
        [0.13108414, 0.00201369, -0.00028784]
    )
    assert short_payload["points_local"]["swing_support_point"] == pytest.approx(
        [0.13108414, 0.00201369, -0.00028784]
    )


# Verify the primitive training additions expose explicit family-specific semantics.
def test_get_object_pose_semantics_payload_exposes_primitive_training_object_axes() -> None:
    """Primitive hammer and screwdriver additions should expose explicit non-default semantics."""
    primitive_hammer_payload = get_object_pose_semantics_payload("cuboid_hammer_v014")
    primitive_screwdriver_payload = get_object_pose_semantics_payload("cylinder_screwdriver_v3009")

    assert primitive_hammer_payload["axes_local"]["tip_axis"] == [-1.0, 0.0, 0.0]
    assert primitive_hammer_payload["axes_local"]["strike_face_normal"] == [0.0, -1.0, 0.0]
    assert primitive_hammer_payload["points_local"]["strike_point"] == pytest.approx(
        [0.09594095009006728, -0.030644512266176062, 0.0]
    )
    assert primitive_hammer_payload["points_local"]["swing_support_point"] == pytest.approx(
        [-0.07670341, 0.0, 0.0]
    )
    assert primitive_screwdriver_payload["axes_local"]["head_axis"] == [-1.0, 0.0, 0.0]
    assert primitive_screwdriver_payload["axes_local"]["strike_face_normal"] == [1.0, 0.0, 0.0]
    assert primitive_screwdriver_payload["points_local"]["strike_point"] == pytest.approx(
        [0.11591023, 0.0, 0.0]
    )
    assert primitive_screwdriver_payload["points_local"]["swing_support_point"] == pytest.approx(
        [0.11591023, 0.0, 0.0]
    )
