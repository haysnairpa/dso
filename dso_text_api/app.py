from __future__ import annotations

import os
import tempfile
import traceback
from typing import Any, Dict, Optional

from flask import Flask, jsonify, request

import pandas as pd

from dso_common.config import env_bool, env_int, env_str
from dso_common.llm_client import build_llm_client
from dso_legal_tools.rule_extraction import process_legal_terms
from dso_text_api.service import LoadedTextModels, detect_text_from_pdf_file, load_text_models


app_text = Flask(__name__)

_loaded: Optional[LoadedTextModels] = None


def _excel_text_sheet_to_legal_terms(*, excel_path: str, sheet_name: str) -> list[dict]:
    df = pd.read_excel(excel_path, sheet_name=sheet_name, header=0)
    df.columns = [str(c).strip() for c in df.columns]
    df = df.fillna("")

    def find_col_like(candidates: list[str]) -> str | None:
        for cand in candidates:
            for c in df.columns:
                if c.lower() == cand.lower():
                    return c
        for cand in candidates:
            for c in df.columns:
                if cand.lower() in c.lower():
                    return c
        return None

    type_col = find_col_like(["Type of Requirement", "Type of requirements"])
    no_col = find_col_like(["No", "Number", "No."])
    item_col = find_col_like(["Item", "Items"])
    requirement_col = find_col_like(["Requirement", "Requirements"])
    reference_col = find_col_like(["Reference"])
    remarks_col = find_col_like(["Remarks"])

    exclude = {type_col, no_col, item_col, requirement_col, reference_col, remarks_col}
    lang_cols = [c for c in df.columns if c not in exclude and c and not c.lower().startswith("unnamed")]

    if type_col:
        df[type_col] = df[type_col].replace("", pd.NA).ffill().fillna("")
    if item_col:
        df[item_col] = df[item_col].replace("", pd.NA).ffill().fillna("")

    def row_has_any_info(r) -> bool:
        if no_col and str(r.get(no_col, "")).strip():
            return True
        if requirement_col and str(r.get(requirement_col, "")).strip():
            return True
        for lc in lang_cols:
            if str(r.get(lc, "")).strip():
                return True
        if reference_col and str(r.get(reference_col, "")).strip():
            return True
        if remarks_col and str(r.get(remarks_col, "")).strip():
            return True
        return False

    df = df[df.apply(row_has_any_info, axis=1)].reset_index(drop=True)

    def sort_key_no(no: str):
        import re

        if not no:
            return (9999,)
        parts = re.findall(r"\d+", no)
        return tuple(int(x) for x in parts) if parts else (9999,)

    groups: dict[str, list[dict]] = {}
    order: list[str] = []

    for _, row in df.iterrows():
        t = str(row.get(type_col, "")).strip() if type_col else ""
        no = str(row.get(no_col, "")).strip() if no_col else ""

        if t not in groups:
            groups[t] = []
            order.append(t)

        if not no:
            continue

        item_text = str(row.get(item_col, "")).strip() if item_col else ""
        requirement_text = str(row.get(requirement_col, "")).strip() if requirement_col else ""
        ref_val = str(row.get(reference_col, "")).strip() if reference_col else ""
        remarks_val = str(row.get(remarks_col, "")).strip() if remarks_col else ""

        lang_list = []
        for lc in lang_cols:
            v = str(row.get(lc, "")).strip()
            if v == "":
                v = "-"
            lang_list.append({lc: v})

        groups[t].append(
            {
                "no": no,
                "item": item_text,
                "requirement": requirement_text,
                "reference": [ref_val] if ref_val else [],
                "remarks": [remarks_val] if remarks_val else [],
                "language reference": lang_list,
            }
        )

    result: list[dict] = []
    for t in order:
        items = groups.get(t) or []
        if not items:
            continue
        items_sorted = sorted(items, key=lambda x: sort_key_no(x.get("no", "")))
        result.append({"type": t, "item": items_sorted})

    return result


@app_text.route('/health', methods=['GET'])
def health_check_text():
    return jsonify({
        "status": "healthy",
        "service": "text_detection_api",
        "version": "1.0",
    })


@app_text.route('/load_models', methods=['POST'])
def load_models_text():
    global _loaded

    try:
        hi_sam_checkpoint = env_str("HI_SAM_CHECKPOINT", os.path.join(os.getcwd(), "pretrained_checkpoint", "hi_sam_h.pth"))
        legal_rules_text_path = env_str("LEGAL_RULES_TEXT_PATH", os.path.join(os.getcwd(), "legal_rules.json"))
        enable_panel_detection = env_bool("ENABLE_PANEL_DETECTION", True)
        panel_model = env_str("PANEL_DETECTOR_MODEL", "gpt-4o") or "gpt-4o"
        openai_api_key = env_str("OPENAI_API_KEY")

        _loaded = load_text_models(
            hi_sam_checkpoint=hi_sam_checkpoint,
            legal_rules_text_path=legal_rules_text_path,
            enable_panel_detection=enable_panel_detection,
            openai_api_key=openai_api_key,
            panel_model=panel_model,
        )

        rules_count = 0
        try:
            rules_count = len(_loaded.legal_rules_text) if _loaded.legal_rules_text is not None else 0
        except Exception:
            rules_count = 0

        return jsonify({
            "status": "success",
            "message": "Models loaded successfully",
            "hi_sam_checkpoint": hi_sam_checkpoint,
            "rules_count": rules_count,
            "panel_detection": enable_panel_detection,
        })

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e),
            "traceback": traceback.format_exc(),
        }), 500


@app_text.route('/convert_excel_text', methods=['POST'])
def convert_excel_text():
    global _loaded

    try:
        if 'excel' not in request.files:
            return jsonify({"status": "error", "message": "No Excel file provided"}), 400

        excel_file = request.files['excel']
        sheet_name = request.form.get('sheet_name', 'Text Availability - 21A')

        with tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx') as tmp_excel:
            excel_file.save(tmp_excel.name)
            excel_path = tmp_excel.name

        try:
            terms = _excel_text_sheet_to_legal_terms(excel_path=excel_path, sheet_name=sheet_name)

            with tempfile.NamedTemporaryFile(mode='w', suffix='_legal_terms_text.json', delete=False, encoding='utf-8') as tmp_terms:
                import json

                json.dump(terms, tmp_terms, ensure_ascii=False, indent=2)
                terms_path = tmp_terms.name

            rules_path = tempfile.mktemp(suffix='_legal_rules_text.json')

            provider = env_str("PARSER_PROVIDER", env_str("VALIDATOR_PROVIDER", "fireworks") or "fireworks") or "fireworks"
            api_key = env_str("PARSER_API_KEY")
            if not api_key:
                api_key = env_str("OPENAI_API_KEY") if provider == "openai" else env_str("FIREWORKS_API_KEY")
            if not api_key:
                raise ValueError("Missing PARSER_API_KEY (or OPENAI_API_KEY/FIREWORKS_API_KEY)")

            model = env_str(
                "PARSER_MODEL",
                env_str("VALIDATOR_MODEL", "accounts/fireworks/models/llama-v3p1-70b-instruct")
                or "accounts/fireworks/models/llama-v3p1-70b-instruct",
            )
            delay = float(env_str("PARSER_DELAY", "0.5") or "0.5")

            process_legal_terms(
                input_path=terms_path,
                output_path=rules_path,
                delay_between_calls=delay,
                parser_api_key=api_key,
                parser_provider=provider,
                parser_model=model,
            )

            with open(rules_path, 'r', encoding='utf-8') as f:
                import json

                rules = json.load(f)

            if _loaded is not None:
                _loaded.legal_rules_text = rules

            total_items = 0
            try:
                for sec in terms:
                    total_items += len(sec.get('item', []))
            except Exception:
                total_items = 0

            return jsonify({
                "status": "success",
                "message": "Excel converted successfully",
                "legal_terms_path": terms_path,
                "legal_rules_path": rules_path,
                "sections_count": len(terms),
                "total_items": total_items,
            })
        finally:
            try:
                os.unlink(excel_path)
            except OSError:
                pass

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e),
            "traceback": traceback.format_exc(),
        }), 500


@app_text.route('/detect_text', methods=['POST'])
def detect_text():
    global _loaded

    if _loaded is None:
        return jsonify({
            "status": "error",
            "message": "Models not loaded. Call /load_models first.",
        }), 400

    try:
        if 'pdf' not in request.files:
            return jsonify({"status": "error", "message": "No PDF file provided"}), 400

        pdf_file = request.files['pdf']
        config = request.form.to_dict()
        dpi = int(config.get('dpi', 300))
        detect_panels = config.get('detect_panels', 'true').lower() == 'true'

        result = detect_text_from_pdf_file(
            models=_loaded,
            pdf_file_storage=pdf_file,
            dpi=dpi,
            detect_panels=detect_panels,
        )

        return jsonify({
            "status": "success",
            **result,
        })

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e),
            "traceback": traceback.format_exc(),
        }), 500


@app_text.route('/validate_text', methods=['POST'])
def validate_text():
    global _loaded

    if _loaded is None or _loaded.legal_rules_text is None:
        return jsonify({
            "status": "error",
            "message": "Legal rules not loaded. Call /load_models first.",
        }), 400

    try:
        data = request.get_json(force=True) or {}
        ocr_results = data.get('ocr_results', [])
        country = data.get('country', 'USA')
        product_metadata = data.get('product_metadata', {})

        from dso_text_api.rules import RuleEngine

        provider = env_str("VALIDATOR_PROVIDER", "fireworks") or "fireworks"
        model_name = env_str("VALIDATOR_MODEL", "accounts/fireworks/models/llama-v3p1-70b-instruct") or "accounts/fireworks/models/llama-v3p1-70b-instruct"
        delay = float(env_str("VALIDATOR_DELAY", "0.5") or "0.5")

        llm_client = build_llm_client(provider=provider)

        image_metadata = product_metadata.get('image_metadata', {
            'width': 1000,
            'height': 1000,
            'dpi': 300,
        })

        import json
        import tempfile

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8') as tmp_rules:
            json.dump(_loaded.legal_rules_text, tmp_rules, ensure_ascii=False, indent=2)
            rules_path = tmp_rules.name

        with tempfile.NamedTemporaryFile(mode='w', suffix='_plan_cache.json', delete=False, encoding='utf-8') as tmp_cache:
            tmp_cache.write('{"_prompt_version": "1"}')
            plan_cache_path = tmp_cache.name

        try:
            engine = RuleEngine(
                rules_path=rules_path,
                ocr_data=ocr_results,
                image_metadata=image_metadata,
                product_metadata={**product_metadata, "country": country},
                api_client=llm_client,
                plan_cache_path=plan_cache_path,
            )
            final_report = engine.validate(model_name=model_name, delay_between_calls=delay)
        finally:
            try:
                os.unlink(rules_path)
            except OSError:
                pass
            try:
                os.unlink(plan_cache_path)
            except OSError:
                pass

        return jsonify({
            "status": "success",
            "validation_results": final_report,
        })

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e),
            "traceback": traceback.format_exc(),
        }), 500


def run_api_text(port: int = 5001):
    app_text.run(host='0.0.0.0', port=port, debug=False)


if __name__ == '__main__':
    port = env_int("TEXT_API_PORT", 5001)
    run_api_text(port=port)
