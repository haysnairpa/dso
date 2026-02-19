from __future__ import annotations

import os
import traceback
from typing import Any, Dict, Optional

from flask import Flask, jsonify, request

from dso_common.config import env_bool, env_int, env_str
from dso_common.llm_client import build_llm_client
from dso_text_api.service import LoadedTextModels, detect_text_from_pdf_file, load_text_models


app_text = Flask(__name__)

_loaded: Optional[LoadedTextModels] = None


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
