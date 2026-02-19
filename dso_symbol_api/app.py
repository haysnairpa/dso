from __future__ import annotations

import os
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
