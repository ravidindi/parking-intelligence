"""Official traffic offence catalogue and rule-based parking offence assignment."""

from __future__ import annotations

from typing import Any


# Official/public legal categories used for detector output.
# Source: Bengaluru Traffic Police spot-fines page:
# https://btp.karnataka.gov.in/117/spot-fines/en
OFFENCE_CATALOG: dict[str, dict[str, Any]] = {
    "MV_ACT_177_NO_PARKING": {
        "code": "MV_ACT_177",
        "legal_section": "Section 177 of M.V. Act",
        "label": "No Parking",
        "fine_amount": 1000,
        "source_url": "https://btp.karnataka.gov.in/117/spot-fines/en",
        "contexts": {"no_parking", "restricted_zone", "commercial_corridor"},
    },
    "190_CLAUSE_117_WRONG_PARKING": {
        "code": "190_CLAUSE_117",
        "legal_section": "190 Clause 117",
        "label": "Wrong Parking",
        "fine_amount": 1000,
        "source_url": "https://btp.karnataka.gov.in/117/spot-fines/en",
        "contexts": {
            "wrong_parking",
            "main_road",
            "footpath",
            "double_parking",
            "road_crossing",
            "bus_stop_school_hospital",
            "traffic_light_zebra",
            "opposite_parked_vehicle",
            "bus_stop",
        },
    },
    "SEC_15_2_RW_SEC_177_WRONG_PARKING_TOW_2W": {
        "code": "SEC_15_2_RW_SEC_177_TOWING_2W",
        "legal_section": "Sec 15(2), read with Sec 177 M.V. Act",
        "label": "Wrong Parking + Towing charges (Two-wheeler)",
        "fine_amount": 1650,
        "source_url": "https://btp.karnataka.gov.in/117/spot-fines/en",
        "contexts": {"towing_two_wheeler"},
    },
    "SEC_15_2_RW_SEC_177_WRONG_PARKING_TOW_CAR_3W": {
        "code": "SEC_15_2_RW_SEC_177_TOWING_CAR_3W",
        "legal_section": "Sec 15(2), read with Sec 177 M.V. Act",
        "label": "Wrong Parking + Towing charges (Car / Three-wheeler)",
        "fine_amount": 2000,
        "source_url": "https://btp.karnataka.gov.in/117/spot-fines/en",
        "contexts": {"towing_car", "towing_three_wheeler"},
    },
    "SEC_15_2_RW_SEC_177_WRONG_PARKING_TOW_MTV": {
        "code": "SEC_15_2_RW_SEC_177_TOWING_MTV",
        "legal_section": "Sec 15(2), read with Sec 177 M.V. Act",
        "label": "Medium Transport Vehicle towing charges (MTV)",
        "fine_amount": 2250,
        "source_url": "https://btp.karnataka.gov.in/117/spot-fines/en",
        "contexts": {"towing_mtv"},
    },
    "SEC_15_2_RW_SEC_177_WRONG_PARKING_TOW_HTV": {
        "code": "SEC_15_2_RW_SEC_177_TOWING_HGV",
        "legal_section": "Sec 15(2), read with Sec 177 M.V. Act",
        "label": "Wrong Parking + Towing charges (HGV)",
        "fine_amount": 2500,
        "source_url": "https://btp.karnataka.gov.in/117/spot-fines/en",
        "contexts": {"towing_htv", "towing_hgv"},
    },
}

SUBTYPE_LABELS = {
    "main_road": "Parking in a main road",
    "footpath": "Footpath parking",
    "double_parking": "Double parking",
    "road_crossing": "Parking near road crossing",
    "bus_stop_school_hospital": "Parking near bus stop, school, or hospital",
    "traffic_light_zebra": "Parking near traffic light or zebra crossing",
    "opposite_parked_vehicle": "Parking opposite another parked vehicle",
    "bus_stop": "Parking other than bus stop",
    "no_parking": "No-parking zone violation",
    "wrong_parking": "Wrong parking",
    "restricted_zone": "Restricted-zone parking",
    "commercial_corridor": "Commercial-corridor parking obstruction",
    "towing_two_wheeler": "Wrong parking with two-wheeler towing",
    "towing_car": "Wrong parking with car towing",
    "towing_three_wheeler": "Wrong parking with three-wheeler towing",
    "towing_mtv": "Wrong parking with medium transport vehicle towing",
    "towing_htv": "Wrong parking with heavy transport vehicle towing",
    "towing_hgv": "Wrong parking with heavy goods vehicle towing",
}


def list_offences() -> list[dict[str, Any]]:
    rows = []
    for item in OFFENCE_CATALOG.values():
        payload = item.copy()
        payload["contexts"] = sorted(payload["contexts"])
        rows.append(payload)
    return sorted(rows, key=lambda item: item["code"])


def classify_parking_offence(
    *,
    alert: bool,
    restricted_zone: bool,
    offence_context: str | None = None,
    vehicle_class: str | None = None,
) -> dict[str, Any] | None:
    """Assign a legal traffic offence to an illegal-parking alert.

    YOLO supplies the vehicle class and stationary alert. Road-rule context comes
    from the camera/geofence configuration because a single image usually cannot
    reliably prove a no-parking board, bus stop, school zone, etc.
    """
    if not alert:
        return None

    normalized = _normalize_context(offence_context)
    if normalized.startswith("towing"):
        code = _towing_code(normalized, vehicle_class)
    elif normalized in OFFENCE_CATALOG["MV_ACT_177_NO_PARKING"]["contexts"]:
        code = "MV_ACT_177_NO_PARKING"
    elif restricted_zone and not offence_context:
        code = "MV_ACT_177_NO_PARKING"
    else:
        code = "190_CLAUSE_117_WRONG_PARKING"

    offence = OFFENCE_CATALOG[code].copy()
    offence["contexts"] = sorted(offence["contexts"])
    offence["detected_subtype"] = SUBTYPE_LABELS.get(normalized, SUBTYPE_LABELS["wrong_parking"])
    offence["assignment_basis"] = "YOLO stationary vehicle alert + camera/geofence road-rule context"
    return offence


def _normalize_context(value: str | None) -> str:
    text = (value or "").strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "restricted": "restricted_zone",
        "commercial": "commercial_corridor",
        "mainroad": "main_road",
        "main_road_parking": "main_road",
        "parking_in_a_main_road": "main_road",
        "crossing": "road_crossing",
        "road_crossing": "road_crossing",
        "zebra": "traffic_light_zebra",
        "traffic_light": "traffic_light_zebra",
        "bus_stop": "bus_stop_school_hospital",
        "school": "bus_stop_school_hospital",
        "hospital": "bus_stop_school_hospital",
        "tow": "towing_car",
        "towing": "towing_car",
        "towing_2w": "towing_two_wheeler",
        "towing_two_wheeler": "towing_two_wheeler",
        "towing_3w": "towing_three_wheeler",
        "towing_car_3w": "towing_car",
        "towing_mtv": "towing_mtv",
        "towing_htv": "towing_htv",
        "towing_hgv": "towing_hgv",
    }
    return aliases.get(text, text or "wrong_parking")


def _towing_code(context: str, vehicle_class: str | None) -> str:
    vehicle = (vehicle_class or "").lower()
    if "motorcycle" in vehicle or "two" in context:
        return "SEC_15_2_RW_SEC_177_WRONG_PARKING_TOW_2W"
    if "truck" in vehicle or "bus" in vehicle or "hgv" in context or "htv" in context:
        return "SEC_15_2_RW_SEC_177_WRONG_PARKING_TOW_HTV"
    if "mtv" in context:
        return "SEC_15_2_RW_SEC_177_WRONG_PARKING_TOW_MTV"
    return "SEC_15_2_RW_SEC_177_WRONG_PARKING_TOW_CAR_3W"
