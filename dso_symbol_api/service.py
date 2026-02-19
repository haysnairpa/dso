from __future__ import annotations

import gc
import os
import tempfile
from dataclasses import dataclass
from typing import Any, Dict, Optional

from dso_symbol_api.yolo import detect_symbols_from_pil_image, load_yolo_model

from dso_common.config import env_bool, env_str
from dso_text_api.panel import VLMPanelDetector


@dataclass
class LoadedSymbolModels:
    yolo_model: Any
    legal_rules_logo: Optional[Any]
    legal_rules_logo_path: Optional[str]
    legal_rules_logo_mtime: Optional[float]


def clear_gpu_memory(*, unload: Optional[LoadedSymbolModels] = None) -> None:
    try:
        import torch

        if unload is not None:
            if unload.yolo_model is not None:
                del unload.yolo_model
            unload.yolo_model = None  # type: ignore[attr-defined]

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    except Exception:
        gc.collect()


def _load_rules_json_if_present(*, path: str | None) -> tuple[Optional[Any], Optional[float]]:
    if not path:
        return None, None
    if not os.path.exists(path):
        return None, None

    import json

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f), os.path.getmtime(path)


def maybe_reload_logo_rules(models: LoadedSymbolModels) -> None:
    path = models.legal_rules_logo_path
    if not path:
        return
    if not os.path.exists(path):
        return

    try:
        mtime = os.path.getmtime(path)
    except OSError:
        return

    if models.legal_rules_logo_mtime is not None and mtime <= models.legal_rules_logo_mtime:
        return

    rules, rules_mtime = _load_rules_json_if_present(path=path)
    models.legal_rules_logo = rules
    models.legal_rules_logo_mtime = rules_mtime


def load_symbol_models(*, symbol_model_path: str, legal_rules_logo_path: str | None) -> LoadedSymbolModels:
    yolo_model = load_yolo_model(model_path=symbol_model_path)

    legal_rules_logo, legal_rules_logo_mtime = _load_rules_json_if_present(path=legal_rules_logo_path)

    return LoadedSymbolModels(
        yolo_model=yolo_model,
        legal_rules_logo=legal_rules_logo,
        legal_rules_logo_path=legal_rules_logo_path,
        legal_rules_logo_mtime=legal_rules_logo_mtime,
    )


def save_filestorage_to_temp_pdf(*, pdf_file_storage) -> str:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_pdf:
        pdf_file_storage.save(tmp_pdf.name)
        return tmp_pdf.name


def pdf_first_page_to_pil(*, pdf_path: str, dpi: int):
    import fitz
    from PIL import Image

    doc = fitz.open(pdf_path)
    try:
        page = doc[0]
        pix = page.get_pixmap(dpi=dpi)
        img_data = pix.tobytes("png")
    finally:
        doc.close()

    import io

    img = Image.open(io.BytesIO(img_data))
    return img


def detect_symbols_from_pdf_file(
    *,
    models: LoadedSymbolModels,
    pdf_file_storage,
    dpi: int,
    confidence_threshold: float,
) -> Dict[str, Any]:
    if models.yolo_model is None:
        raise ValueError("Models not loaded. Call /load_models first.")

    pdf_path = save_filestorage_to_temp_pdf(pdf_file_storage=pdf_file_storage)
    try:
        image = pdf_first_page_to_pil(pdf_path=pdf_path, dpi=dpi)

        detection_payload = detect_symbols_from_pil_image(
            yolo_model=models.yolo_model,
            image=image,
            dpi=dpi,
            confidence_threshold=confidence_threshold,
        )

        enable_panel_detection = env_bool("ENABLE_SYMBOL_PANEL_DETECTION", True)
        panel_model = env_str("PANEL_DETECTOR_MODEL", "gpt-4o") or "gpt-4o"
        openai_api_key = env_str("OPENAI_API_KEY")

        panels: list[Dict[str, Any]] = []
        if enable_panel_detection and openai_api_key:
            temp_img_path = pdf_path.replace(".pdf", "_temp.jpg")
            try:
                image.convert("RGB").save(temp_img_path)

                detector = VLMPanelDetector(api_key=openai_api_key, model=panel_model)
                panels_data = detector.detect_panels(temp_img_path)
                if panels_data and "panels" in panels_data:
                    img_width, img_height = image.size

                    for panel in panels_data.get("panels", []):
                        bbox_percent = panel.get("bbox_percent") or {}
                        x_min = int(bbox_percent.get("x_min", 0) * img_width / 100)
                        y_min = int(bbox_percent.get("y_min", 0) * img_height / 100)
                        x_max = int(bbox_percent.get("x_max", 0) * img_width / 100)
                        y_max = int(bbox_percent.get("y_max", 0) * img_height / 100)

                        width_px = max(0, x_max - x_min)
                        height_px = max(0, y_max - y_min)
                        area_mm2 = (width_px / dpi) * 25.4 * (height_px / dpi) * 25.4

                        panels.append(
                            {
                                "panel_id": panel.get("type", "unknown"),
                                "bbox_pixels": {"x": x_min, "y": y_min, "width": width_px, "height": height_px},
                                "area_mm2": float(area_mm2),
                            }
                        )
            finally:
                try:
                    os.unlink(temp_img_path)
                except Exception:
                    pass

        if panels:
            def _intersect_area(a: Dict[str, float], b: Dict[str, float]) -> float:
                ax1, ay1 = a["x"], a["y"]
                ax2, ay2 = ax1 + a["width"], ay1 + a["height"]
                bx1, by1 = b["x"], b["y"]
                bx2, by2 = bx1 + b["width"], by1 + b["height"]
                inter_w = max(0.0, min(ax2, bx2) - max(ax1, bx1))
                inter_h = max(0.0, min(ay2, by2) - max(ay1, by1))
                return inter_w * inter_h

            for det in detection_payload.get("detections", []):
                bbox = det.get("box_pixels") or {}
                overlaps = []
                for p in panels:
                    overlap = _intersect_area(
                        {
                            "x": float(bbox.get("x", 0)),
                            "y": float(bbox.get("y", 0)),
                            "width": float(bbox.get("width", 0)),
                            "height": float(bbox.get("height", 0)),
                        },
                        {
                            "x": float(p["bbox_pixels"]["x"]),
                            "y": float(p["bbox_pixels"]["y"]),
                            "width": float(p["bbox_pixels"]["width"]),
                            "height": float(p["bbox_pixels"]["height"]),
                        },
                    )
                    if overlap > 0:
                        overlaps.append((overlap, p["panel_id"]))
                if overlaps:
                    overlaps.sort(reverse=True)
                    det["panel_id"] = overlaps[0][1]

        detection_payload["panels"] = panels
        return detection_payload
    finally:
        try:
            os.unlink(pdf_path)
        except OSError:
            pass
