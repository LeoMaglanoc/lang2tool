"""Canonical per-object start poses for LLM taskless evaluation bootstrap."""

from __future__ import annotations

from typing import Dict, List

OBJECT_DEFAULT_START_POSES: Dict[str, List[float]] = {
    "claw_hammer": [
        0.09076502773615436,
        0.08070982635558854,
        0.5443193171844398,
        0.07781270573992773,
        -0.9964358547253277,
        -0.025800699064297956,
        0.019876975902544947,
    ],
    "mallet_hammer": [
        0.10964212432374565,
        0.05731781420714077,
        0.5498402402360318,
        0.022423567732318832,
        -0.9991131768598976,
        0.025734957847727325,
        0.024652695180282735,
    ],
    "cuboid_hammer_v014": [
        0.09076502773615436,
        0.08070982635558854,
        0.5443193171844398,
        0.07781270573992773,
        -0.9964358547253277,
        -0.025800699064297956,
        0.019876975902544947,
    ],
    "sharpie_marker": [
        -0.059410100881627466,
        0.025261806520960994,
        0.5505245236143564,
        -0.3209584591801845,
        0.0009497833629880104,
        -0.013341742563683699,
        0.9469988190581914,
    ],
    "staples_marker": [
        0.0007053400880538364,
        0.03615152907360053,
        0.5501651905249406,
        0.44262162671107125,
        0.009997628064351227,
        0.021857134216425694,
        0.8963863054981248,
    ],
    "flat_eraser": [
        0.017891907420138905,
        0.01433750098519826,
        0.5594944113702583,
        0.7265684189136249,
        0.008695016088282936,
        0.021493013816529252,
        0.686702832154974,
    ],
    "handle_eraser": [
        -0.007975596300357938,
        0.007902676303352507,
        0.6192360284957011,
        -0.0020572368992152596,
        0.00031836295671427983,
        -0.999384909694403,
        0.03500669502363274,
    ],
    "blue_brush": [
        0.08628694939852355,
        0.06359268497014203,
        0.553529241099744,
        0.014606509946195487,
        -0.9989579627547618,
        -0.018517840881811038,
        0.03907336797774226,
    ],
    "red_brush": [
        0.10717264858508924,
        0.0309506937228321,
        0.5506258883605348,
        0.021312152321177176,
        -0.7035186228215238,
        -0.7101620973271443,
        0.01664737296564496,
    ],
    "flat_spatula": [
        0.1639509022680955,
        0.052446195016069064,
        0.5617782639543012,
        0.015356679360908298,
        0.0011567242615518891,
        -0.9976511059907643,
        0.06674657372025378,
    ],
    "spoon_spatula": [
        0.12791466976289834,
        0.0638481537610911,
        0.5658170794548423,
        -0.0491773739927425,
        -0.07300454430589098,
        -0.9956496238098624,
        0.030557306902712245,
    ],
    "long_screwdriver": [
        -0.08121582760511159,
        0.04770862987737767,
        0.5519587969799857,
        0.9926745259136501,
        0.0055147057933565835,
        -0.019325801689775535,
        0.11913600216211073,
    ],
    "short_screwdriver": [
        -0.08124288503508437,
        0.06992222301753948,
        0.5534889486522074,
        0.10394449734798628,
        0.011913907911660804,
        -0.050428675390105704,
        0.9932323741037372,
    ],
    "cylinder_screwdriver_v3009": [
        -0.08121582760511159,
        0.04770862987737767,
        0.5519587969799857,
        0.9926745259136501,
        0.0055147057933565835,
        -0.019325801689775535,
        0.11913600216211073,
    ],
}


# Return canonical start pose for an object in taskless LLM mode.
def get_default_start_pose(object_name: str) -> List[float]:
    """Return canonical 7D start pose for an object or raise a clear error."""
    if object_name not in OBJECT_DEFAULT_START_POSES:
        known = ", ".join(sorted(OBJECT_DEFAULT_START_POSES.keys()))
        raise KeyError(
            f"Missing default start pose for object '{object_name}'. Known objects: {known}"
        )
    return list(OBJECT_DEFAULT_START_POSES[object_name])


# Build one canonical object start pose with the runtime Z safety offset applied.
def build_default_start_pose(object_name: str, *, z_offset: float, table_z: float) -> List[float]:
    """Return the object's default start pose after applying the runtime Z clamp."""
    start_pose = get_default_start_pose(object_name)
    min_pose_z = float(table_z) + float(z_offset)
    start_pose[2] = max(float(start_pose[2]) + float(z_offset), min_pose_z)
    return start_pose
