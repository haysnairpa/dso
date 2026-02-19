from __future__ import annotations

import datetime as dt
import json
import os
import re
import textwrap
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI


TOLERANCE_MM = 1e-3

LEGAL_TERM_LOGO_OVERRIDES_RAW = {
    "Product with no As-Received small part": "small_part_warning",
    "Small Part Warning": "small_part_warning",
    "Small Part maybe generated warning": "small_part_warning",
    "Mr. Yucky Logo": "small_part_warning",
    "Mr. Yucky": "small_part_warning",
    "Company Name": "mattel_logo",
    "Mattel Logo": "mattel_logo",
    "CE Logo": "ce_mark",
    "CE Mark": "ce_mark",
    "Lion Mark": "lion_mark",
    "Malaysian Conformity (MC) Mark": "mc_mark",
    "Malaysiaon Conformity (MC) Mark": "mc_mark",
    "MC Mark": "mc_mark",
    "UKCA Mark": "ukca",
    "UKCA": "ukca",
    "Customer Information Block (CIB)": "cib_mark",
    "CIB": "cib_mark",
    "Argentina Certification of Conformity Seal": "ast_logo",
    "AST Logo": "ast_logo",
    "Brazilian Logo": "compulsorio_logo",
    "Brazilian Logo Option 1": "compulsorio_logo",
    "Brazilian Logo Option 2": "no_compulsorio_logo",
    "GCC Mark": "gcc_mark",
    "EAC Label": "eac_label",
    "Mobius Loop": "mobius_loop",
    "Mobius Loop symbol for Shipper Individual": "mobius_loop",
    "Ukrainian National Conformity Symbol": "ukraine_symbol",
    "Ukraine Symbol": "ukraine_symbol",
    "Triman Logo": "triman_logo",
    "Sorting Bin Logo": "sorting_bin_logo",
    "Sorting Label Logo": "sorting_bin_logo",
    "Floor Standing POP Display": "fspop",
    'Italian "IT" Logo': "it_logo",
    "IT Logo": "it_logo",
    "Paper or Cardboard Packaging Component(s)": "paper_cardboard_symbol",
}

SYMBOL_ALIAS_MAP = {
    "small_part_warning": ["small_part_warning_logo", "mr_yucky"],
    "mattel_logo": ["mattel_logo"],
    "ce_mark": ["ce_mark"],
    "lion_mark": ["lion_mark"],
    "mc_mark": ["mc", "malaysian_conformity_mark"],
    "ukca": ["ukca"],
    "cib_mark": ["cib_mark", "customer_information_block"],
    "ast_logo": ["ast_logo", "argentina_certification_of_conformity_seal"],
    "compulsorio_logo": ["compulsorio_logo", "brazilian_logo"],
    "no_compulsorio_logo": ["no_compulsorio_logo", "brazilian_logo_option_2"],
    "gcc_mark": ["gcc_mark"],
    "eac_label": ["eac_label"],
    "mobius_loop": ["mobius_loop"],
    "ukraine_symbol": ["ukraine_symbol", "ukrainian_national_conformity_symbol"],
    "triman_logo": ["triman_logo"],
    "sorting_bin_logo": ["sorting_bin_logo", "sorting_label_logo"],
    "fspop": ["fspop", "floor_standing_pop_display"],
    "it_logo": ["it_logo", "italian_it_logo"],
    "paper_cardboard_symbol": ["paper_cardboard_symbol", "paper_or_cardboard_packaging_components"],
}


_symbol_class_map_cache: Dict[str, str] = {}


def _canonicalize(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip().lower())


def _get_symbol_class_map() -> Dict[str, str]:
    global _symbol_class_map_cache
    if _symbol_class_map_cache:
        return _symbol_class_map_cache

    base_map: Dict[str, str] = {}
    for name, label in LEGAL_TERM_LOGO_OVERRIDES_RAW.items():
        base_map[_canonicalize(name)] = label

    detected_labels = set(base_map.values())
    for label in detected_labels:
        base_map[_canonicalize(label)] = label
        base_map[_canonicalize(label.replace("_", " "))] = label
        base_map[label] = label

    _symbol_class_map_cache = base_map
    return _symbol_class_map_cache


def normalize_symbol_key(value: str) -> str:
    raw = str(value or "").strip()
    lowered = _canonicalize(raw)
    symbol_map = _get_symbol_class_map()

    if lowered in symbol_map:
        return symbol_map[lowered]

    for key, aliases in SYMBOL_ALIAS_MAP.items():
        lowered_aliases = {_canonicalize(alias) for alias in aliases}
        if lowered in lowered_aliases or raw in aliases:
            return key

    sanitized = re.sub(r"[^a-z0-9+]+", "_", lowered).strip("_")
    if sanitized in symbol_map:
        return symbol_map[sanitized]
    spaced = sanitized.replace("_", " ") if sanitized else lowered
    if spaced in symbol_map:
        return symbol_map[spaced]

    return sanitized or lowered or raw


def build_detection_index(records: List[Dict]) -> Dict[str, Any]:
    all_detections: List[Dict] = []
    for record in records:
        panels_by_id = {panel.get("panel_id"): panel for panel in record.get("panels", [])}
        for det in record.get("detections", []):
            symbol_key = normalize_symbol_key(det.get("label"))
            panel_id = det.get("panel_id")
            panel_info = panels_by_id.get(panel_id)
            all_detections.append(
                {
                    "symbol_key": symbol_key,
                    "label": det.get("label"),
                    "confidence": det.get("confidence"),
                    "box_pixels": det.get("box_pixels"),
                    "box_mm": det.get("box_mm"),
                    "color_stats": det.get("color_stats"),
                    "patch_path": det.get("patch_path"),
                    "panel_id": panel_id,
                    "panel": panel_info,
                    "packaging_panels_mm2": record.get("packaging_panel_areas_mm2"),
                    "largest_panel_area_mm2": record.get("largest_panel_area_mm2"),
                    "source_image": record.get("source_image"),
                }
            )

    detections_by_symbol: Dict[str, List[Dict]] = defaultdict(list)
    for det in all_detections:
        detections_by_symbol[det["symbol_key"]].append(det)

    packaging_meta: Dict[str, Any] = {}
    for record in records:
        src = record.get("source_image")
        packaging_meta[src] = {
            "panel_areas_mm2": record.get("packaging_panel_areas_mm2"),
            "largest_panel_area_mm2": record.get("largest_panel_area_mm2"),
        }

    return {
        "records": records,
        "all": all_detections,
        "by_symbol": detections_by_symbol,
        "packaging_meta": packaging_meta,
    }


def extract_symbol_candidates(rule_texts: List[Dict], detection_index: Dict[str, Any]) -> List[Dict]:
    candidates = []
    for entry in rule_texts or []:
        text_value = entry.get("text") if isinstance(entry, dict) else entry
        if not text_value:
            continue
        raw = str(text_value).strip()
        normalized_key = normalize_symbol_key(raw)
        if normalized_key in detection_index["by_symbol"]:
            candidates.append({
                "raw_text": raw,
                "symbol_key": normalized_key,
            })
    return candidates


def parse_numeric_value(value_str: str) -> Tuple[Optional[float], Optional[str]]:
    match = re.match(r"(-?\d+(?:\.\d+)?)\s*([a-zA-Z0-9²^]*)", value_str.strip())
    if not match:
        return None, None
    number = float(match.group(1))
    unit = match.group(2).lower() if match.group(2) else None
    return number, unit


def convert_to_mm(value: float, unit: Optional[str]) -> float:
    if unit in (None, "mm", "mm."):
        return value
    if unit in {"cm", "cm."}:
        return value * 10.0
    if unit in {"in", "inch", "inches"}:
        return value * 25.4
    return value


def convert_area_to_mm2(value: float, unit: Optional[str]) -> float:
    if unit in (None, "mm2", "mm^2"):
        return value
    if unit in {"cm2", "cm^2"}:
        return value * 100.0
    if unit in {"in2", "in^2", "sqin"}:
        return value * (25.4**2)
    return value


def parse_requirements(requirements: List[str]) -> Dict[str, Any]:
    parsed: Dict[str, Any] = {
        "presence_required": False,
        "dimension_checks": [],
        "color_checks": [],
        "color_profile_checks": [],
        "panel_checks": [],
        "packaging_checks": [],
        "other": [],
    }

    for raw_req in requirements or []:
        requirement = raw_req.strip()
        lowered = requirement.lower()
        if lowered in {"text_available = true", "symbol_available = true", "available = true"}:
            parsed["presence_required"] = True
            continue

        comparison = re.match(r"([a-zA-Z_ ]+)\s*(>=|<=|=|!=|>|<)\s*(.+)", requirement)
        if not comparison:
            parsed["other"].append(requirement)
            continue

        field = _canonicalize(comparison.group(1)).replace(" ", "_")
        operator = comparison.group(2)
        value_token = comparison.group(3).strip()
        numeric_value, unit = parse_numeric_value(value_token)

        if field in {"text_height", "symbol_height", "height", "height_mm"}:
            if numeric_value is None:
                parsed["other"].append(requirement)
            else:
                parsed["dimension_checks"].append({
                    "metric": "height_mm",
                    "operator": operator,
                    "expected": convert_to_mm(numeric_value, unit),
                    "raw": requirement,
                })
        elif field in {"text_width", "symbol_width", "width", "width_mm"}:
            if numeric_value is None:
                parsed["other"].append(requirement)
            else:
                parsed["dimension_checks"].append({
                    "metric": "width_mm",
                    "operator": operator,
                    "expected": convert_to_mm(numeric_value, unit),
                    "raw": requirement,
                })
        elif field in {"diameter", "diameter_mm"}:
            if numeric_value is None:
                parsed["other"].append(requirement)
            else:
                expected_mm = convert_to_mm(numeric_value, unit)
                for metric in ("width_mm", "height_mm"):
                    parsed["dimension_checks"].append({
                        "metric": metric,
                        "operator": operator,
                        "expected": expected_mm,
                        "raw": requirement,
                    })
        elif field in {"height_px", "width_px"}:
            if numeric_value is None:
                parsed["other"].append(requirement)
            else:
                parsed["dimension_checks"].append({
                    "metric": field,
                    "operator": operator,
                    "expected": numeric_value,
                    "raw": requirement,
                })
        elif field in {"color_contrast", "contrast", "contrast_ratio"}:
            if numeric_value is None:
                parsed["other"].append(requirement)
            else:
                parsed["color_checks"].append({
                    "metric": "contrast_ratio",
                    "operator": operator,
                    "expected": numeric_value,
                    "raw": requirement,
                })
        elif "panel_area" in field:
            if numeric_value is None:
                parsed["other"].append(requirement)
            else:
                if "cm2" in field or "cm²" in field:
                    area_mm2 = numeric_value * 100
                else:
                    area_mm2 = numeric_value
                parsed["packaging_checks"].append({
                    "metric": "largest_panel_area_mm2",
                    "operator": operator,
                    "expected": area_mm2,
                    "raw": requirement,
                })
        elif field in {"position", "panel", "panel_position"}:
            parsed["panel_checks"].append({
                "field": "panel_id",
                "operator": operator,
                "expected": value_token.replace("'", "").strip(),
                "raw": requirement,
            })
        elif field == "panel_area_mm2":
            if numeric_value is None:
                parsed["other"].append(requirement)
            else:
                parsed["panel_checks"].append({
                    "field": "panel_area_mm2",
                    "operator": operator,
                    "expected": convert_area_to_mm2(numeric_value, unit),
                    "raw": requirement,
                })
        elif field in {"largest_panel_area_mm2", "largest_panel_area_cm2"}:
            if numeric_value is None:
                parsed["other"].append(requirement)
            else:
                expected_mm2 = convert_area_to_mm2(numeric_value, "cm2") if field.endswith("cm2") else convert_area_to_mm2(numeric_value, unit)
                parsed["packaging_checks"].append({
                    "metric": "largest_panel_area_mm2",
                    "operator": operator,
                    "expected": expected_mm2,
                    "raw": requirement,
                })
        elif field in {"color_scheme", "colour_scheme"}:
            normalized_value = _canonicalize(value_token)
            if "contrast" in normalized_value:
                parsed["color_checks"].append({
                    "metric": "contrast_ratio",
                    "operator": ">=" if operator == "=" else operator,
                    "expected": 4.5,
                    "raw": requirement,
                })
            else:
                options = [_canonicalize(opt.strip()) for opt in re.split(r"\s+or\s+", normalized_value)] if normalized_value else []
                parsed["color_profile_checks"].append({
                    "metric": "color_scheme",
                    "operator": operator,
                    "expected": options or ([normalized_value] if normalized_value else []),
                    "raw": requirement,
                })
        elif field in {"foreground_color", "background_color"}:
            normalized_value = _canonicalize(value_token)
            options = [_canonicalize(opt.strip()) for opt in re.split(r"\s+or\s+", normalized_value)] if normalized_value else []
            parsed["color_profile_checks"].append({
                "metric": field,
                "operator": operator,
                "expected": options or ([normalized_value] if normalized_value else []),
                "raw": requirement,
            })
        elif field in {"panel_id", "panel", "panel_position"}:
            parsed["panel_checks"].append({
                "field": "panel_id",
                "operator": operator,
                "expected": value_token.replace("'", "").strip(),
                "raw": requirement,
            })
        else:
            parsed["other"].append(requirement)

    return parsed


def compare_numeric(measured: Optional[float], operator: str, expected: float) -> Tuple[bool, str]:
    if measured is None:
        return False, "missing_measurement"
    if operator == ">=":
        return measured >= expected - TOLERANCE_MM, "OK" if measured >= expected - TOLERANCE_MM else "below_minimum"
    if operator == ">":
        return measured > expected, "OK" if measured > expected else "not_greater"
    if operator == "<=":
        return measured <= expected + TOLERANCE_MM, "OK" if measured <= expected + TOLERANCE_MM else "above_maximum"
    if operator == "<":
        return measured < expected, "OK" if measured < expected else "not_less"
    if operator == "=":
        return abs(measured - expected) <= TOLERANCE_MM, "OK" if abs(measured - expected) <= TOLERANCE_MM else "mismatch"
    if operator == "!=":
        return abs(measured - expected) > TOLERANCE_MM, "OK" if abs(measured - expected) > TOLERANCE_MM else "matches_disallowed_value"
    return False, "unsupported_operator"


def detection_evidence(det: Dict) -> Dict:
    panel = det.get("panel") or {}
    return {
        "source_image": det.get("source_image"),
        "label": det.get("label"),
        "panel_id": det.get("panel_id"),
        "panel_area_mm2": panel.get("area_mm2"),
        "box_pixels": det.get("box_pixels"),
        "box_mm": det.get("box_mm"),
        "color_stats": det.get("color_stats"),
        "patch_path": det.get("patch_path"),
        "confidence": float(det.get("confidence")) if det.get("confidence") is not None else None,
    }


def check_symbol_presence(symbol_key: str, detections: List[Dict], presence_required: bool) -> Dict:
    status = "PASS" if detections else ("FAIL" if presence_required else "NOT_EVALUATED")
    return {
        "tool": "check_symbol_presence",
        "status": status,
        "evidence": {
            "symbol_key": symbol_key,
            "detections_found": len(detections),
        },
    }


def check_dimensions(detections: List[Dict], dimension_checks: List[Dict]) -> Dict:
    if not dimension_checks:
        return {"tool": "check_dimensions", "status": "NOT_APPLICABLE", "evidence": []}

    if not detections:
        return {
            "tool": "check_dimensions",
            "status": "FAIL",
            "evidence": [],
            "reason": "No detections found to measure dimensions",
        }

    overall_pass = True
    evidence = []

    for det in detections:
        box_mm = det.get("box_mm") or {}
        box_px = det.get("box_pixels") or {}
        det_checks = []

        for check in dimension_checks:
            metric = check["metric"]
            measured = None
            if metric.endswith("_mm"):
                if metric == "width_mm":
                    measured = box_mm.get("width")
                elif metric == "height_mm":
                    measured = box_mm.get("height")
            elif metric.endswith("_px"):
                if metric == "width_px":
                    measured = box_px.get("width")
                elif metric == "height_px":
                    measured = box_px.get("height")

            passed, reason = compare_numeric(measured, check["operator"], check["expected"])
            if not passed:
                overall_pass = False
            det_checks.append({
                "metric": metric,
                "operator": check["operator"],
                "expected": check["expected"],
                "measured": measured,
                "result": "PASS" if passed else "FAIL",
                "reason": reason,
                "raw": check["raw"],
            })

        evidence.append({
            "detection": detection_evidence(det),
            "checks": det_checks,
        })

    return {
        "tool": "check_dimensions",
        "status": "PASS" if overall_pass else "FAIL",
        "evidence": evidence,
    }


def _relative_luminance(rgb):
    if not rgb:
        return None
    r, g, b = [channel / 255.0 for channel in rgb]
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def _classify_tone(rgb):
    if not rgb:
        return None
    lum = _relative_luminance(rgb)
    if lum is None:
        return None
    if lum < 0.25:
        return "black"
    if lum > 0.75:
        return "white"
    return "mid"


def _classify_color_scheme(fore_rgb, back_rgb):
    def luminance(rgb):
        r, g, b = [channel / 255.0 for channel in rgb]
        return 0.299 * r + 0.587 * g + 0.114 * b

    if not fore_rgb or not back_rgb:
        return None, None

    fore_lum = luminance(fore_rgb)
    back_lum = luminance(back_rgb)

    fore_class = "black" if fore_lum < 0.25 else ("white" if fore_lum > 0.75 else "mid")
    back_class = "black" if back_lum < 0.25 else ("white" if back_lum > 0.75 else "mid")
    return fore_class, back_class


def check_color_contrast(detections, color_checks, color_profile_checks):
    have_numeric = bool(color_checks)
    have_profile = bool(color_profile_checks)
    if not have_numeric and not have_profile:
        return {"tool": "check_color_contrast", "status": "NOT_APPLICABLE", "evidence": []}
    if not detections:
        return {
            "tool": "check_color_contrast",
            "status": "FAIL",
            "reason": "symbol_not_detected",
            "evidence": [],
        }

    overall_pass = True
    evidence = []

    for det in detections:
        color_stats = det.get("color_stats") or {}
        fore_rgb = color_stats.get("foreground_rgb")
        bg_rgb = color_stats.get("background_rgb")
        contrast_ratio = color_stats.get("contrast_ratio")

        _ = _classify_color_scheme(fore_rgb, bg_rgb)
        det_checks = []

        for check in color_checks:
            metric = check["metric"]
            operator = check["operator"]
            expected = check["expected"]

            if metric == "contrast_ratio":
                measured = contrast_ratio
                passed, reason = compare_numeric(measured, operator, expected) if measured is not None else (False, "missing_measurement")
                if not passed:
                    overall_pass = False
                det_checks.append({
                    "metric": metric,
                    "operator": operator,
                    "expected": expected,
                    "measured": measured,
                    "result": "PASS" if passed else "FAIL",
                    "reason": reason,
                    "raw": check["raw"],
                })
            else:
                det_checks.append({
                    "metric": metric,
                    "operator": operator,
                    "expected": expected,
                    "measured": None,
                    "result": "NOT_SUPPORTED",
                    "reason": "unsupported_metric",
                    "raw": check["raw"],
                })

        fore_tone = _classify_tone(fore_rgb)
        back_tone = _classify_tone(bg_rgb)

        for check in color_profile_checks:
            metric = check["metric"]
            operator = check["operator"]
            expected_values = check.get("expected", [])
            if not isinstance(expected_values, (list, tuple)):
                expected_values = [expected_values]
            expected_norm = [_canonicalize(val) for val in expected_values if val is not None]

            measured = None
            if metric == "color_scheme":
                if fore_tone and back_tone:
                    measured = f"{fore_tone}_{back_tone}"
                elif contrast_ratio is not None:
                    measured = "contrasting" if contrast_ratio >= 4.5 else "non_contrasting"
            elif metric == "foreground_color":
                measured = fore_tone
            elif metric == "background_color":
                measured = back_tone

            measured_norm = _canonicalize(measured) if measured else None
            passed = False
            if measured_norm is not None:
                if operator == "=":
                    passed = measured_norm in expected_norm
                elif operator == "!=":
                    passed = all(measured_norm != val for val in expected_norm)

            reason = "insufficient_color_stats" if measured_norm is None else ("match" if passed else "mismatch")
            if not passed:
                overall_pass = False

            det_checks.append({
                "metric": metric,
                "operator": operator,
                "expected": expected_values,
                "measured": measured,
                "result": "PASS" if passed else "FAIL",
                "reason": reason,
                "raw": check["raw"],
            })

        evidence.append({
            "detection": detection_evidence(det),
            "checks": det_checks,
        })

    return {
        "tool": "check_color_contrast",
        "status": "PASS" if overall_pass else "FAIL",
        "evidence": evidence,
    }


def check_packaging_panels(packaging_meta: Optional[Dict[str, Any]], packaging_checks: List[Dict]) -> Dict:
    if not packaging_checks:
        return {"tool": "check_packaging_panels", "status": "NOT_APPLICABLE", "evidence": []}
    if not packaging_meta:
        return {"tool": "check_packaging_panels", "status": "FAIL", "reason": "no_packaging_metadata", "evidence": []}

    evidence = []
    overall_pass = True

    for check in packaging_checks:
        metric = check["metric"]
        measured = packaging_meta.get(metric)
        expected = check["expected"]
        operator = check["operator"]
        passed, reason = compare_numeric(measured, operator, expected) if measured is not None else (False, "missing_measurement")
        if not passed:
            overall_pass = False
        evidence.append({
            "metric": metric,
            "operator": operator,
            "expected": expected,
            "measured": measured,
            "result": "PASS" if passed else "FAIL",
            "reason": reason,
            "raw": check["raw"],
        })

    return {
        "tool": "check_packaging_panels",
        "status": "PASS" if overall_pass else "FAIL",
        "evidence": evidence,
    }


def check_panel_condition(detections: List[Dict], panel_checks: List[Dict]) -> Dict:
    if not panel_checks:
        return {"tool": "check_panel_condition", "status": "NOT_APPLICABLE", "evidence": []}
    if not detections:
        return {
            "tool": "check_panel_condition",
            "status": "FAIL",
            "reason": "symbol_not_detected",
            "evidence": [],
        }

    unsupported = []
    overall_pass = True
    evidence = []

    for det in detections:
        panel = det.get("panel") or {}
        det_checks = []

        for check in panel_checks:
            field = check["field"]
            expected_raw = check["expected"]
            normalized_expected = _canonicalize(expected_raw)
            operator = check["operator"]

            if field == "panel_id":
                measured = _canonicalize(det.get("panel_id")) if det.get("panel_id") else ""
                if normalized_expected in {"any_panel", "any"} and operator == "=":
                    passed = True
                    reason = "any_panel"
                elif normalized_expected.startswith("same_panel_as") or normalized_expected.startswith("close_proximity"):
                    unsupported.append(check["raw"])
                    passed = True
                    reason = "unsupported_condition"
                else:
                    if operator == "=":
                        passed = measured == normalized_expected
                        reason = "match" if passed else "panel_mismatch"
                    elif operator == "!=":
                        passed = measured != normalized_expected
                        reason = "different_panel" if passed else "panel_match_disallowed"
                    else:
                        unsupported.append(check["raw"])
                        passed = True
                        reason = "unsupported_operator"
            elif field == "panel_area_mm2":
                measured = panel.get("area_mm2")
                passed, reason = compare_numeric(measured, operator, check["expected"])
                if reason == "unsupported_operator":
                    unsupported.append(check["raw"])
            else:
                unsupported.append(check["raw"])
                passed = True
                reason = "unsupported_field"

            if not passed:
                overall_pass = False

            det_checks.append({
                "field": field,
                "operator": operator,
                "expected": expected_raw,
                "measured": det.get("panel_id") if field == "panel_id" else panel.get("area_mm2"),
                "result": "PASS" if passed else "FAIL",
                "reason": reason,
                "raw": check["raw"],
            })

        evidence.append({
            "detection": detection_evidence(det),
            "checks": det_checks,
        })

    status = "NOT_EVALUATED" if unsupported and overall_pass else ("PASS" if overall_pass else "FAIL")
    return {
        "tool": "check_panel_condition",
        "status": status,
        "unsupported_conditions": [str(cond) for cond in unsupported],
        "evidence": evidence,
    }


TOOL_REGISTRY = {
    "check_symbol_presence": {
        "description": "Verifies at least one detection exists for the symbol. Requires symbol_available requirement.",
        "requires_detections": True,
        "callable": check_symbol_presence,
    },
    "check_dimensions": {
        "description": "Validates width_mm / height_mm comparisons on detections.",
        "requires_detections": True,
        "callable": check_dimensions,
    },
    "check_color_contrast": {
        "description": "Confirms color_scheme / foreground/background requirements using color stats.",
        "requires_detections": True,
        "callable": check_color_contrast,
    },
    "check_panel_condition": {
        "description": "Tests panel_id or panel_area_mm2 constraints considering detection panel assignment.",
        "requires_detections": True,
        "callable": check_panel_condition,
    },
    "check_packaging_panels": {
        "description": "Applies global packaging metadata rules such as largest_panel_area_cm2.",
        "requires_detections": False,
        "callable": check_packaging_panels,
    },
}

LLM_PLANNER_PROMPT = textwrap.dedent(
    """
You are a packaging compliance planning assistant. Given rule requirements and detection context, decide which deterministic tools must run. Use only the registry. Output strict JSON:
{
  "tool_plan": [ { "tool": str, "reason": str } ],
  "notes": optional str
}

Registry:
{json.dumps({k: {vk: vv for vk, vv in v.items() if vk != 'callable'} for k, v in TOOL_REGISTRY.items()}, ensure_ascii=False, indent=2)}

Guidelines:
- Include check_symbol_presence when symbol availability is required.
- Include check_dimensions for width_mm / height_mm comparisons.
- Include check_color_contrast when color_scheme or foreground/background constraints appear.
- Include check_panel_condition for panel_id or panel_area requirements tied to detections.
- Include check_packaging_panels for largest_panel_area_* checks using packaging metadata.
- Skip tools that cannot succeed (e.g., no detections).
- If nothing applies, return empty plan with reason.

Examples:

Example 1
Input:
{
  "requirements": ["symbol_available = true", "height_mm >= 5.0"],
  "detections_found": 1,
  "has_packaging_meta": false,
  "available_tools": ["check_symbol_presence", "check_dimensions", "check_color_contrast"]
}
Output:
{
  "tool_plan": [
    {"tool": "check_symbol_presence", "reason": "Requirement enforces symbol availability."},
    {"tool": "check_dimensions", "reason": "Rule specifies height_mm constraint."}
  ]
}

Example 2
Input:
{
  "requirements": ["largest_panel_area_cm2 < 278.75", "symbol_available = true"],
  "detections_found": 0,
  "has_packaging_meta": true,
  "available_tools": ["check_symbol_presence", "check_packaging_panels"]
}
Output:
{
  "tool_plan": [
    {"tool": "check_symbol_presence", "reason": "Need to confirm symbol presence."},
    {"tool": "check_packaging_panels", "reason": "Rule compares largest panel area against threshold."}
  ]
}

Example 3
Input:
{
  "requirements": ["color_scheme = black_white"],
  "detections_found": 0,
  "has_packaging_meta": false,
  "available_tools": ["check_color_contrast"]
}
Output:
{
  "tool_plan": [],
  "notes": "No detections available; color validation not possible."
}
"""
).strip()


def build_planner_payload(rule_requirements, detections, packaging_meta, parsed_requirements):
    return {
        "requirements": rule_requirements,
        "detections_found": len(detections),
        "has_packaging_meta": bool(packaging_meta),
        "presence_required": parsed_requirements["presence_required"],
        "dimension_checks": parsed_requirements["dimension_checks"],
        "color_checks": parsed_requirements["color_checks"],
        "color_profile_checks": parsed_requirements["color_profile_checks"],
        "panel_checks": parsed_requirements["panel_checks"],
        "packaging_checks": parsed_requirements["packaging_checks"],
        "available_tools": list(TOOL_REGISTRY.keys()),
    }


LLM_MIN_INTERVAL = 4.0
LLM_MAX_RETRIES = 3
LLM_RATE_LIMIT_BUFFER = 3


def _call_llm(*, client: OpenAI, model: str, messages: List[Dict[str, str]]) -> Dict[str, Any]:
    _last_llm_call_ts = getattr(_call_llm, "_last_call_ts", 0.0)

    for attempt in range(LLM_MAX_RETRIES):
        wait = LLM_MIN_INTERVAL - (time.time() - _last_llm_call_ts)
        if wait > 0:
            time.sleep(wait)

        try:
            start = time.time()
            response = client.chat.completions.create(
                model=model,
                temperature=0.0,
                response_format={"type": "json_object"},
                messages=messages,
            )
            _call_llm._last_call_ts = time.time()
            payload = json.loads(response.choices[0].message.content)
            payload.setdefault("latency_seconds", time.time() - start)
            payload.setdefault("model", model)
            return payload
        except Exception as exc:
            if "rate limit" in str(exc).lower() and attempt < LLM_MAX_RETRIES - 1:
                backoff = min(5 * (2**attempt), 60) + LLM_RATE_LIMIT_BUFFER
                time.sleep(backoff)
                continue
            return {"status": "ERROR", "rationale": f"LLM call failed: {exc}", "raw_response": repr(exc)}


def parse_notes_conditions(notes: Dict[str, Any]) -> List[str]:
    conditions = []
    for key, value in (notes or {}).items():
        if "panel_area" in key.lower():
            if isinstance(value, str):
                conditions.append(f"{key} {value}")
            elif isinstance(value, (int, float)):
                conditions.append(f"{key} = {value}")
    return conditions


def plan_tools(*, client: OpenAI, model: str, rule_requirements, detections, packaging_meta, parsed_requirements) -> Dict[str, Any]:
    payload = build_planner_payload(rule_requirements, detections, packaging_meta, parsed_requirements)
    result = _call_llm(
        client=client,
        model=model,
        messages=[
            {"role": "system", "content": LLM_PLANNER_PROMPT},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False, indent=2)},
        ],
    )
    if "tool_plan" not in result:
        result["tool_plan"] = []
    return result


class SymbolRuleEngine:
    def __init__(
        self,
        *,
        detection_json_path: Path,
        legal_rules_path: Path,
        client: OpenAI,
        model: str,
        dry_run: bool = False,
    ) -> None:
        self.detection_json_path = Path(detection_json_path)
        self.legal_rules_path = Path(legal_rules_path)
        self.client = client
        self.model = model
        self.dry_run = dry_run
        self.detection_records = self._load_json(self.detection_json_path)
        self.legal_rules = self._load_json(self.legal_rules_path)
        self.detection_index = build_detection_index(self.detection_records)

    @staticmethod
    def _load_json(path: Path) -> Any:
        if not path.exists():
            raise FileNotFoundError(f"Required JSON file not found: {path}")
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def run(self) -> Dict[str, Any]:
        results = []
        for section in self.legal_rules:
            section_type = section.get("type")
            for item in section.get("item", []):
                item_no = item.get("item_no") or item.get("no")
                item_name = item.get("item_name") or item.get("name")
                for rule_idx, rule in enumerate(item.get("rules", []), start=1):
                    rule_result = self.evaluate_rule(
                        section_type=section_type,
                        item_no=item_no,
                        item_name=item_name,
                        rule_index=rule_idx,
                        rule=rule,
                    )
                    if rule_result:
                        results.append(rule_result)

        timestamp = dt.datetime.utcnow().isoformat() + "Z"
        return {
            "generated_at": timestamp,
            "model": self.model,
            "rules_evaluated": len(results),
            "results": results,
        }

    def evaluate_rule(
        self,
        *,
        section_type: Optional[str],
        item_no: Optional[str],
        item_name: Optional[str],
        rule_index: int,
        rule: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        rule_texts = rule.get("text") or []
        requirements = list(rule.get("requirement") or [])

        notes = rule.get("notes") or {}
        notes_conditions = parse_notes_conditions(notes)
        requirements.extend(notes_conditions)
        if not (rule_texts or requirements):
            return None

        parsed_requirements = parse_requirements(requirements)
        symbol_candidates = extract_symbol_candidates(rule_texts, self.detection_index)

        rule_results = []
        packaging_meta_by_image = self.detection_index.get("packaging_meta", {})

        for candidate in (
            symbol_candidates
            or [{"symbol_key": normalize_symbol_key(item_name or ""), "raw_text": item_name}]
        ):
            symbol_key = candidate["symbol_key"]
            detections = self.detection_index["by_symbol"].get(symbol_key, [])

            if detections:
                packaging_meta = packaging_meta_by_image.get(detections[0]["source_image"]) if detections else None
                plan = plan_tools(
                    client=self.client,
                    model=self.model,
                    rule_requirements=requirements,
                    detections=detections,
                    packaging_meta=packaging_meta,
                    parsed_requirements=parsed_requirements,
                )
                tool_plan = [step["tool"] for step in plan.get("tool_plan", [])]

                if parsed_requirements["dimension_checks"] and "check_dimensions" not in tool_plan:
                    tool_plan.append("check_dimensions")
                if parsed_requirements["panel_checks"] and "check_panel_condition" not in tool_plan:
                    tool_plan.append("check_panel_condition")
                if parsed_requirements["color_checks"] and "check_color_contrast" not in tool_plan:
                    tool_plan.append("check_color_contrast")
            else:
                packaging_meta = next(iter(packaging_meta_by_image.values()), None)
                tool_plan = []

                if parsed_requirements["presence_required"]:
                    tool_plan.append("check_symbol_presence")
                if parsed_requirements["dimension_checks"]:
                    tool_plan.append("check_dimensions")
                if parsed_requirements["color_checks"] or parsed_requirements["color_profile_checks"]:
                    tool_plan.append("check_color_contrast")
                if parsed_requirements["panel_checks"]:
                    tool_plan.append("check_panel_condition")
                if packaging_meta and parsed_requirements.get("packaging_checks"):
                    tool_plan.append("check_packaging_panels")

            tool_results = []
            for tool_name in tool_plan:
                if tool_name == "check_symbol_presence":
                    tool_results.append(check_symbol_presence(symbol_key, detections, parsed_requirements["presence_required"]))
                elif tool_name == "check_dimensions":
                    tool_results.append(check_dimensions(detections, parsed_requirements["dimension_checks"]))
                elif tool_name == "check_color_contrast":
                    tool_results.append(check_color_contrast(detections, parsed_requirements["color_checks"], parsed_requirements["color_profile_checks"]))
                elif tool_name == "check_panel_condition":
                    tool_results.append(check_panel_condition(detections, parsed_requirements["panel_checks"]))
                elif tool_name == "check_packaging_panels":
                    tool_results.append(check_packaging_panels(packaging_meta, parsed_requirements["packaging_checks"]))

            llm_payload = self._invoke_llm(
                section_type=section_type,
                item_no=item_no,
                item_name=item_name,
                rule_texts=rule_texts,
                requirements=requirements,
                symbol_candidate=candidate,
                detections=detections,
                tool_results=tool_results,
            )

            rule_results.append(
                {
                    "section_type": section_type,
                    "item_no": item_no,
                    "item_name": item_name,
                    "rule_index": rule_index,
                    "rule_texts": rule_texts,
                    "requirements": requirements,
                    "symbol_candidate": candidate,
                    "detections": [detection_evidence(det) for det in detections],
                    "tool_results": tool_results,
                    "llm_summary": llm_payload,
                }
            )

        return {
            "section_type": section_type,
            "item_no": item_no,
            "item_name": item_name,
            "rule_index": rule_index,
            "rule_results": rule_results,
        }

    def _invoke_llm(
        self,
        *,
        section_type: Optional[str],
        item_no: Optional[str],
        item_name: Optional[str],
        rule_texts: List[Dict[str, str]],
        requirements: List[str],
        symbol_candidate: Dict[str, str],
        detections: List[Dict[str, Any]],
        tool_results: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        summary_payload = {
            "section_type": section_type,
            "item_no": item_no,
            "item_name": item_name,
            "rule_texts": rule_texts,
            "requirements": requirements,
            "symbol_candidate": symbol_candidate,
            "detections": [detection_evidence(det) for det in detections],
        }
        prompt = self._build_prompt(summary_payload, tool_results)
        if self.dry_run:
            return {
                "status": "DRY_RUN",
                "rationale": "LLM not called (dry_run=True)",
                "raw_response": None,
                "prompt": prompt,
            }

        result = _call_llm(
            client=self.client,
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": textwrap.dedent(
                        """
                        You are a compliance validator. Given tool outputs, determine if requirements are met.
                        CRITICAL RULES:
                        1. You MUST mention ALL requirements in your evidence_notes, not just failed ones
                        2. For each requirement, state the tool result: PASS, FAIL, or NOT_APPLICABLE
                        3. Status = PASS only if ALL requirements pass
                        4. Status = FAIL if ANY requirement fails
                        5. Status = ERROR only if tools didn't run or crashed
                        6. Status = NOT_APPLICABLE if no requirements apply to this detection
                        Evidence format (MANDATORY):
                        - "check_symbol_presence: PASS/FAIL (reason)"
                        - "check_dimensions: PASS/FAIL (width_mm: X, height_mm: Y)"
                        - "check_color_contrast: PASS/FAIL (contrast_ratio: X)"
                        - "check_panel_condition: PASS/FAIL (panel_id: X)"
                        - "check_packaging_panels: PASS/FAIL (largest_panel_area_mm2: X)"
                        Output strict JSON:
                        {
                          "status": "PASS" | "FAIL" | "ERROR" | "NOT_APPLICABLE",
                          "rationale": "Brief summary mentioning ALL checks",
                          "evidence_notes": ["List ALL tool results with metrics"],
                          "actions": ["Corrective actions if FAIL, empty if PASS"]
                        }
                        """
                    ).strip(),
                },
                {"role": "user", "content": prompt},
            ],
        )
        result.setdefault("model", self.model)
        return result

    @staticmethod
    def _build_prompt(summary_payload: Dict[str, Any], tool_results: List[Dict[str, Any]]) -> str:
        instructions = (
            "Review the legal rule texts, requirements, and deterministic tool outputs. "
            "Decide compliance strictly from the tool results. Output a JSON object with keys: "
            "status (PASS, FAIL, NOT_APPLICABLE, or ERROR), rationale (string), "
            "evidence_notes (list of strings extracted from tool evidence), and "
            "actions (optional list of follow-up actions)."
        )
        return (
            f"{instructions}\n\n"
            f"Rule context: {json.dumps(summary_payload, ensure_ascii=False, indent=2)}\n\n"
            f"Tool outputs: {json.dumps(tool_results, ensure_ascii=False, indent=2)}\n\n"
            "Respond with JSON only."
        )
