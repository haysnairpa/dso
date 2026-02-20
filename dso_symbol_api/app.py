from __future__ import annotations

import os
import tempfile
import traceback
from typing import Optional

from flask import Flask, jsonify, request

import pandas as pd

from dso_common.config import env_str
from dso_legal_tools.rule_extraction import process_legal_terms
from dso_symbol_api.service import (
    LoadedSymbolModels,
    clear_gpu_memory,
    detect_symbols_from_pdf_file,
    load_symbol_models,
    maybe_reload_logo_rules,
)
from dso_symbol_api.rules import validate_symbols_with_llm_engine


app = Flask(__name__)

_loaded: Optional[LoadedSymbolModels] = None


def _excel_logo_sheet_to_legal_terms(*, excel_path: str, sheet_name: str) -> list[dict]:
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
    dim_req_col = find_col_like(["Dimension Requirement"])

    exclude = {type_col, no_col, item_col, requirement_col, dim_req_col}
    other_cols = [c for c in df.columns if c not in exclude and c and not c.lower().startswith("unnamed")]

    if type_col:
        df[type_col] = df[type_col].replace("", pd.NA).ffill().fillna("")
    if item_col:
        df[item_col] = df[item_col].replace("", pd.NA).ffill().fillna("")

    def row_has_any_info(r) -> bool:
        if no_col and str(r.get(no_col, "")).strip():
            return True
        if requirement_col and str(r.get(requirement_col, "")).strip():
            return True
        if dim_req_col and str(r.get(dim_req_col, "")).strip():
            return True
        for oc in other_cols:
            if str(r.get(oc, "")).strip():
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
        dim_req_text = str(row.get(dim_req_col, "")).strip() if dim_req_col else ""

        entry = {
            "no": no,
            "item": item_text,
            "requirement": requirement_text,
        }
        if dim_req_col:
            entry["dimension_requirement"] = dim_req_text

        groups[t].append(entry)

    result: list[dict] = []
    for t in order:
        items = groups.get(t) or []
        if not items:
            continue
        items_sorted = sorted(items, key=lambda x: sort_key_no(x.get("no", "")))
        result.append({"type": t, "item": items_sorted})

    return result


@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "healthy",
        "service": "symbol_detection_api",
        "version": "1.0",
    })


@app.route('/clear_gpu', methods=['POST'])
def clear_gpu():
    global _loaded

    try:
        if _loaded is not None:
            clear_gpu_memory(unload=_loaded)
            _loaded = None
        else:
            clear_gpu_memory(unload=None)

        return jsonify({
            "status": "success",
            "message": "GPU memory cleared successfully",
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e),
        }), 500


@app.route('/load_models', methods=['POST'])
def load_models():
    global _loaded

    try:
        symbol_model_path = env_str("SYMBOL_MODEL_PATH", os.path.join(os.getcwd(), "models", "symbol", "best.pt"))
        legal_rules_logo_path = env_str(
            "LEGAL_RULES_LOGO_PATH",
            os.path.join(os.getcwd(), "data", "rules", "logo_rules.json"),
        )

        _loaded = load_symbol_models(symbol_model_path=symbol_model_path, legal_rules_logo_path=legal_rules_logo_path)

        return jsonify({
            "status": "success",
            "message": "YOLO model loaded successfully",
            "model_path": symbol_model_path,
            "rules_path": legal_rules_logo_path,
            "rules_loaded": _loaded.legal_rules_logo is not None,
        })

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e),
            "traceback": traceback.format_exc(),
        }), 500


@app.route('/convert_excel_logo', methods=['POST'])
def convert_excel_logo():
    global _loaded

    try:
        if 'excel' not in request.files:
            return jsonify({"status": "error", "message": "No Excel file provided"}), 400

        excel_file = request.files['excel']
        sheet_name = request.form.get('sheet_name', 'Logo List - 21A')

        with tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx') as tmp_excel:
            excel_file.save(tmp_excel.name)
            excel_path = tmp_excel.name

        try:
            terms = _excel_logo_sheet_to_legal_terms(excel_path=excel_path, sheet_name=sheet_name)

            with tempfile.NamedTemporaryFile(mode='w', suffix='_legal_terms_logo.json', delete=False, encoding='utf-8') as tmp_terms:
                import json

                json.dump(terms, tmp_terms, ensure_ascii=False, indent=2)
                terms_path = tmp_terms.name

            rules_path = tempfile.mktemp(suffix='_legal_rules_logo.json')

            provider = env_str("PARSER_PROVIDER", env_str("SYMBOL_VALIDATOR_PROVIDER", env_str("VALIDATOR_PROVIDER", "fireworks") or "fireworks") or "fireworks") or "fireworks"
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
                _loaded.legal_rules_logo = rules
                _loaded.legal_rules_logo_path = rules_path
                try:
                    _loaded.legal_rules_logo_mtime = os.path.getmtime(rules_path)
                except OSError:
                    _loaded.legal_rules_logo_mtime = None

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


@app.route('/detect_symbols', methods=['POST'])
def detect_symbols():
    global _loaded

    if _loaded is None or _loaded.yolo_model is None:
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
        country = config.get('country', 'USA')
        confidence_threshold = float(config.get('confidence_threshold', 0.25))

        _ = country

        result = detect_symbols_from_pdf_file(
            models=_loaded,
            pdf_file_storage=pdf_file,
            dpi=dpi,
            confidence_threshold=confidence_threshold,
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


@app.route('/validate_symbols', methods=['POST'])
def validate_symbols():
    global _loaded

    if _loaded is None or not _loaded.legal_rules_logo_path:
        return jsonify({
            "status": "error",
            "message": "Legal rules not loaded. Call /load_models first.",
        }), 400

    try:
        maybe_reload_logo_rules(_loaded)
        if _loaded.legal_rules_logo is None:
            return jsonify({
                "status": "error",
                "message": "Legal rules not loaded. Ensure rules file exists and is valid JSON.",
                "rules_path": _loaded.legal_rules_logo_path,
            }), 400

        data = request.get_json(force=True) or {}
        detections = data.get('detections', [])
        country = data.get('country', 'USA')
        product_metadata = data.get('product_metadata', {})

        payload = validate_symbols_with_llm_engine(
            detections=detections,
            legal_rules_logo=_loaded.legal_rules_logo,
            country=country,
            product_metadata=product_metadata,
        )

        return jsonify({
            "status": "success",
            **payload,
        })

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e),
            "traceback": traceback.format_exc(),
        }), 500
