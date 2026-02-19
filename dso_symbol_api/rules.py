from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Dict

from dso_common.config import env_str
from dso_common.llm_client import build_llm_client
from dso_symbol_api.rules_engine import SymbolRuleEngine


def validate_symbols_with_llm_engine(*, detections: list, legal_rules_logo: Any, country: str, product_metadata: Dict[str, Any]) -> Dict[str, Any]:
    if legal_rules_logo is None:
        raise ValueError("Legal rules not loaded")

    detection_records = [
        {
            "source_image": "packaging_image",
            "detections": detections,
            "panels": [],
            "packaging_panel_areas_mm2": product_metadata.get("panel_areas_mm2", {}),
            "largest_panel_area_mm2": product_metadata.get("largest_panel_area_mm2", 0),
        }
    ]

    with tempfile.NamedTemporaryFile(mode="w", suffix="_detections.json", delete=False, encoding="utf-8") as tmp_det:
        import json

        json.dump(detection_records, tmp_det, ensure_ascii=False, indent=2)
        detections_path = tmp_det.name

    with tempfile.NamedTemporaryFile(mode="w", suffix="_legal_rules_logo.json", delete=False, encoding="utf-8") as tmp_rules:
        import json

        json.dump(legal_rules_logo, tmp_rules, ensure_ascii=False, indent=2)
        rules_path = tmp_rules.name

    try:
        provider = env_str("SYMBOL_VALIDATOR_PROVIDER", env_str("VALIDATOR_PROVIDER", "fireworks") or "fireworks") or "fireworks"
        model_name = env_str(
            "SYMBOL_VALIDATOR_MODEL",
            env_str("VALIDATOR_MODEL", "accounts/fireworks/models/llama-v3p1-70b-instruct")
            or "accounts/fireworks/models/llama-v3p1-70b-instruct",
        )

        llm_client = build_llm_client(provider=provider)

        engine = SymbolRuleEngine(
            detection_json_path=Path(detections_path),
            legal_rules_path=Path(rules_path),
            client=llm_client,
            model=model_name,
            dry_run=False,
        )
        validation_report = engine.run()

        validation_results = validation_report.get("results", [])
        passed = 0
        total_rules = 0
        for r in validation_results:
            for rule_result in r.get("rule_results", []):
                total_rules += 1
                llm_summary = rule_result.get("llm_summary", {})
                if llm_summary.get("status") == "PASS":
                    passed += 1
        failed = total_rules - passed

        return {
            "validation_results": validation_results,
            "summary": {
                "total_rules": total_rules,
                "passed": passed,
                "failed": failed,
                "compliance_rate": passed / total_rules if total_rules else 0,
            },
            "full_report": validation_report,
        }
    finally:
        try:
            os.unlink(detections_path)
        except OSError:
            pass
        try:
            os.unlink(rules_path)
        except OSError:
            pass
