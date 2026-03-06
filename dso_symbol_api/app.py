from __future__ import annotations

import os
import json
import tempfile
import traceback
from typing import Optional

from flask import Flask, jsonify, request

from dso_common.config import env_str
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


@app.route('/convert_excel_logo', methods=['POST'])
def convert_excel_logo():
    global _loaded

    try:
        if 'excel' not in request.files:
            return jsonify({"status": "error", "message": "No Excel file provided"}), 400

        excel_file = request.files['excel']
        sheet_name = request.form.get('sheet_name', 'Logo List - 21A')

        rules_path = env_str(
            "LEGAL_RULES_LOGO_PATH",
            os.path.join(os.getcwd(), "data", "rules", "logo_rules.json"),
        )
        os.makedirs(os.path.dirname(rules_path), exist_ok=True)

        import pandas as pd
        import re

        with tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx') as tmp_excel:
            excel_file.save(tmp_excel.name)
            excel_path = tmp_excel.name

        try:
            def read_sheet_try_multi(excel_path_: str, sheet_name_: str):
                try:
                    df_ = pd.read_excel(excel_path_, sheet_name=sheet_name_, header=[0, 1])
                    return df_, True
                except Exception:
                    df_ = pd.read_excel(excel_path_, sheet_name=sheet_name_, header=0)
                    return df_, False

            df_raw, is_multi = read_sheet_try_multi(excel_path, sheet_name)

            if is_multi:
                new_cols, lang_cols, ref_cols, remarks_cols = [], [], [], []
                for col in df_raw.columns:
                    p = str(col[0]).strip() if pd.notna(col[0]) else ""
                    s = str(col[1]).strip() if pd.notna(col[1]) else ""
                    p_low = p.lower()
                    if p_low.startswith("language reference"):
                        col_name = s if s else p
                        lang_cols.append(col_name)
                    else:
                        if p_low.startswith("reference"):
                            col_name = f"Reference - {s}" if s else "Reference"
                            ref_cols.append(col_name)
                        elif p_low.startswith("remarks"):
                            col_name = f"Remarks - {s}" if s else "Remarks"
                            remarks_cols.append(col_name)
                        else:
                            if p and s and "unnamed" not in s.lower():
                                col_name = f"{p} - {s}"
                            elif p:
                                col_name = p
                            elif s:
                                col_name = s
                            else:
                                col_name = ""
                    new_cols.append(col_name)
                df = df_raw.copy()
                df.columns = [str(c).strip() for c in new_cols]
            else:
                df = df_raw.copy()
                df.columns = [str(c).strip() for c in df.columns]
                possible_langs = [
                    "English",
                    "French",
                    "German",
                    "Italian",
                    "Dutch",
                    "Spanish",
                    "Portuguese",
                    "Swedish",
                    "Finnish",
                    "Danish",
                    "Norwegian",
                    "Russian",
                    "Polish",
                    "Czech",
                    "Slovak",
                    "Hungarian",
                    "Romanian",
                    "Greek",
                    "Turkish",
                    "Ukrainian",
                    "Arabic",
                ]
                lang_cols = [c for c in df.columns if any(pl.lower() in c.lower() for pl in possible_langs)]
                ref_cols = [c for c in df.columns if "reference" in c.lower()]
                remarks_cols = [c for c in df.columns if "remark" in c.lower()]

            df.columns = [str(c).strip() for c in df.columns]

            def find_col_like(candidates):
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
            dim_req = find_col_like(["Dimension Requirement"])

            if not ref_cols:
                ref_cols = [c for c in df.columns if c.lower().startswith("reference")]
            if not remarks_cols:
                remarks_cols = [c for c in df.columns if c.lower().startswith("remarks")]
            if not lang_cols:
                exclude = {type_col, no_col, item_col, requirement_col, dim_req}
                lang_cols = [
                    c
                    for c in df.columns
                    if c not in exclude and not any(c.startswith(pref) for pref in ("Reference", "Remarks"))
                ]

            if type_col:
                df[type_col] = df[type_col].ffill()
            if item_col:
                df[item_col] = df[item_col].ffill()

            def row_has_any_info(r):
                if no_col and str(r.get(no_col, "")).strip():
                    return True
                if requirement_col and str(r.get(requirement_col, "")).strip():
                    return True
                for lc in lang_cols:
                    if str(r.get(lc, "")).strip():
                        return True
                for rc in ref_cols:
                    if str(r.get(rc, "")).strip():
                        return True
                for rc in remarks_cols:
                    if str(r.get(rc, "")).strip():
                        return True
                return False

            df = df[df.apply(row_has_any_info, axis=1)].reset_index(drop=True)
            df = df.fillna("")

            def sort_key_no(no: str):
                if not no:
                    return (9999,)
                parts = re.findall(r"\d+", str(no))
                return tuple(int(x) for x in parts) if parts else (9999,)

            groups_order, groups_map = [], {}
            for _, row in df.iterrows():
                t = row.get(type_col, "").strip() if type_col else ""
                no = str(row.get(no_col, "")).strip() if no_col else ""

                if t not in groups_map:
                    groups_map[t] = []
                    groups_order.append(t)

                item_text = str(row.get(item_col, "")).strip() if item_col else ""
                requirement_text = str(row.get(requirement_col, "")).strip() if requirement_col else ""
                dimension_requirement = str(row.get(dim_req, "")).strip() if dim_req else ""

                _ = lang_cols

                groups_map[t].append(
                    {
                        "no": no,
                        "item": item_text,
                        "requirement": requirement_text,
                        "dimension_requirement": dimension_requirement,
                    }
                )

            result = []
            for t in groups_order:
                items = groups_map[t]
                if not items:
                    continue
                items_sorted = sorted(items, key=lambda x: sort_key_no(x.get("no", "")))
                result.append({"type": t, "item": items_sorted})

            with open(rules_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)

            if _loaded is not None:
                _loaded.legal_rules_logo_path = rules_path
                _loaded.legal_rules_logo_mtime = None
                maybe_reload_logo_rules(_loaded)

            return jsonify({
                "status": "success",
                "message": "Excel converted successfully",
                "rules_path": rules_path,
                "sections_count": len(result),
                "total_items": sum(len(s.get('item', [])) for s in result),
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
