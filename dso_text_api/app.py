from __future__ import annotations

import json
import os
import tempfile
import traceback
from typing import Any, Dict, Optional

from flask import Flask, jsonify, request

from dso_common.config import env_bool, env_int, env_str
from dso_common.llm_client import build_llm_client
from dso_legal_tools.rule_extraction import process_legal_terms
from dso_text_api.service import LoadedTextModels, detect_text_from_pdf_file, load_text_models


app_text = Flask(__name__)

_loaded: Optional[LoadedTextModels] = None


@app_text.route('/convert_excel_text', methods=['POST'])
def convert_excel_text():
    global _loaded

    try:
        if 'excel' not in request.files:
            return jsonify({"status": "error", "message": "No Excel file provided"}), 400

        excel_file = request.files['excel']
        sheet_name = request.form.get('sheet_name', 'Text Availability - 21A')

        terms_path = env_str(
            "LEGAL_TERMS_TEXT_PATH",
            os.path.join(os.getcwd(), "data", "rules", "text_terms.json"),
        )
        rules_path = env_str(
            "LEGAL_RULES_TEXT_PATH",
            os.path.join(os.getcwd(), "data", "rules", "text_rules.json"),
        )
        os.makedirs(os.path.dirname(terms_path), exist_ok=True)
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
            reference_base = find_col_like(["Reference"])
            remarks_base = find_col_like(["Remarks"])

            if not ref_cols:
                ref_cols = [c for c in df.columns if c.lower().startswith("reference")]
            if not remarks_cols:
                remarks_cols = [c for c in df.columns if c.lower().startswith("remarks")]
            if not lang_cols:
                exclude = {type_col, no_col, item_col, requirement_col, reference_base, remarks_base}
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

            def collect_list(row, cols):
                out = []
                for c in cols:
                    v = str(row.get(c, "")).strip()
                    if v:
                        out.append(v)
                return out

            groups_order, groups_map = [], {}

            for _, row in df.iterrows():
                t = row.get(type_col, "").strip() if type_col else ""
                no = str(row.get(no_col, "")).strip() if no_col else ""

                if t not in groups_map:
                    groups_map[t] = []
                    groups_order.append(t)

                if not no:
                    if groups_map[t]:
                        last = groups_map[t][-1]
                        extra_ref = collect_list(row, ref_cols) or ([str(row.get(reference_base, "")).strip()] if reference_base else [])
                        extra_rem = collect_list(row, remarks_cols) or ([str(row.get(remarks_base, "")).strip()] if remarks_base else [])
                        extra_ref = [r for r in extra_ref if r and r.strip()]
                        extra_rem = [r for r in extra_rem if r and r.strip()]
                        if extra_ref:
                            last["reference"].extend(extra_ref)
                        if extra_rem:
                            last["remarks"].extend(extra_rem)
                    continue

                item_text = str(row.get(item_col, "")).strip() if item_col else ""
                requirement_text = str(row.get(requirement_col, "")).strip() if requirement_col else ""
                reference_list = collect_list(row, ref_cols) or ([str(row.get(reference_base, "")).strip()] if reference_base else [])
                remarks_list = collect_list(row, remarks_cols) or ([str(row.get(remarks_base, "")).strip()] if remarks_base else [])

                reference_list = [r for r in reference_list if r and r.strip()]
                remarks_list = [r for r in remarks_list if r and r.strip()]

                lang_list = []
                for lc in lang_cols:
                    if not lc:
                        continue
                    v = str(row.get(lc, "")).strip()
                    if v == "":
                        v = "-"
                    lang_list.append({lc: v})

                groups_map[t].append(
                    {
                        "no": no,
                        "item": item_text,
                        "requirement": requirement_text,
                        "reference": reference_list,
                        "remarks": remarks_list,
                        "language reference": lang_list,
                    }
                )

            for t in groups_order:
                for it in groups_map[t]:
                    it["reference"] = list(dict.fromkeys(it.get("reference") or []))
                    it["remarks"] = list(dict.fromkeys(it.get("remarks") or []))

            def sort_key_no(no: str):
                if not no:
                    return (9999,)
                parts = re.findall(r"\d+", str(no))
                return tuple(int(x) for x in parts) if parts else (9999,)

            terms_payload = []
            for t in groups_order:
                items = groups_map[t]
                if not items:
                    continue
                items_sorted = sorted(items, key=lambda x: sort_key_no(x.get("no", "")))
                terms_payload.append({"type": t, "item": items_sorted})

            with open(terms_path, 'w', encoding='utf-8') as f:
                json.dump(terms_payload, f, indent=2, ensure_ascii=False)

            provider = env_str("PARSER_PROVIDER", env_str("VALIDATOR_PROVIDER", "fireworks") or "fireworks") or "fireworks"
            api_key = env_str("PARSER_API_KEY")
            if not api_key:
                api_key = env_str("OPENAI_API_KEY") if provider == "openai" else env_str("FIREWORKS_API_KEY")
            if not api_key:
                return jsonify({
                    "status": "error",
                    "message": "Missing PARSER_API_KEY (or OPENAI_API_KEY/FIREWORKS_API_KEY)",
                }), 400

            model = env_str("PARSER_MODEL", "accounts/fireworks/models/llama-v3p1-70b-instruct") or "accounts/fireworks/models/llama-v3p1-70b-instruct"
            delay = float(env_str("PARSER_DELAY", "0.5") or "0.5")

            process_legal_terms(
                input_path=terms_path,
                output_path=rules_path,
                delay_between_calls=delay,
                parser_api_key=api_key,
                parser_provider=provider,
                parser_model=model,
            )

            legal_rules_text = None
            if os.path.exists(rules_path):
                with open(rules_path, 'r', encoding='utf-8') as f:
                    legal_rules_text = json.load(f)

            if _loaded is not None:
                _loaded.legal_rules_text = legal_rules_text

            return jsonify({
                "status": "success",
                "message": "Excel converted successfully",
                "legal_terms_path": terms_path,
                "legal_rules_path": rules_path,
                "sections_count": len(terms_payload),
                "total_items": sum(len(s.get('item', [])) for s in terms_payload),
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
