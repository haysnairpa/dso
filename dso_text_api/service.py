from __future__ import annotations

import gc
import io
import os
import tempfile
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import fitz
import numpy as np
import torch
from PIL import Image

from dso_text_api.ocr import load_models as load_ocr_models
from dso_text_api.ocr import predict_hi_mask
from dso_text_api.panel import VLMPanelDetector


@dataclass
class LoadedTextModels:
    hi_sam_model: Any
    parseq_model: Any
    device: str
    legal_rules_text: Optional[Any]
    panel_detector: Optional[VLMPanelDetector]


def clear_gpu_memory() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()


def load_text_models(
    *,
    hi_sam_checkpoint: str,
    legal_rules_text_path: str,
    enable_panel_detection: bool,
    openai_api_key: Optional[str],
    panel_model: str,
) -> LoadedTextModels:
    clear_gpu_memory()

    hi_sam_model, parseq_model, device = load_ocr_models(hi_sam_checkpoint=hi_sam_checkpoint)

    legal_rules_text = None
    if legal_rules_text_path and os.path.exists(legal_rules_text_path):
        import json

        with open(legal_rules_text_path, 'r', encoding='utf-8') as f:
            legal_rules_text = json.load(f)

    panel_detector = None
    if enable_panel_detection:
        if not openai_api_key:
            raise ValueError("OPENAI_API_KEY is required when ENABLE_PANEL_DETECTION=true")
        panel_detector = VLMPanelDetector(api_key=openai_api_key, model=panel_model)

    return LoadedTextModels(
        hi_sam_model=hi_sam_model,
        parseq_model=parseq_model,
        device=device,
        legal_rules_text=legal_rules_text,
        panel_detector=panel_detector,
    )


def pdf_first_page_to_numpy(*, pdf_path: str, dpi: int) -> Tuple[np.ndarray, int, int]:
    doc = fitz.open(pdf_path)
    try:
        page = doc[0]
        pix = page.get_pixmap(dpi=dpi)
        img_data = pix.tobytes("png")
    finally:
        doc.close()

    img = Image.open(io.BytesIO(img_data))
    img_array = np.array(img)
    img_width, img_height = img.size
    return img_array, int(img_width), int(img_height)


def detect_text_from_pdf_file(
    *,
    models: LoadedTextModels,
    pdf_file_storage,
    dpi: int,
    detect_panels: bool,
) -> Dict[str, Any]:
    clear_gpu_memory()

    if models.hi_sam_model is None or models.parseq_model is None:
        raise ValueError("Models not loaded. Call /load_models first.")

    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_pdf:
        pdf_file_storage.save(tmp_pdf.name)
        pdf_path = tmp_pdf.name

    try:
        img_array, img_width, img_height = pdf_first_page_to_numpy(pdf_path=pdf_path, dpi=dpi)

        panels: Dict[str, Any] = {}
        panels_bbox: Dict[str, Dict[str, int]] = {}

        if detect_panels and models.panel_detector:
            temp_img_path = pdf_path.replace('.pdf', '_temp.jpg')
            Image.fromarray(img_array).save(temp_img_path)
            try:
                panels_data = models.panel_detector.detect_panels(temp_img_path)
                if panels_data and 'panels' in panels_data:
                    for panel in panels_data['panels']:
                        panel_type = panel['type']
                        bbox_percent = panel['bbox_percent']
                        panels_bbox[panel_type] = {
                            'x_min': int(bbox_percent['x_min'] * img_width / 100),
                            'y_min': int(bbox_percent['y_min'] * img_height / 100),
                            'x_max': int(bbox_percent['x_max'] * img_width / 100),
                            'y_max': int(bbox_percent['y_max'] * img_height / 100),
                        }
                        panels[panel_type] = panels_bbox[panel_type]
            finally:
                try:
                    os.unlink(temp_img_path)
                except OSError:
                    pass

        ocr_results: List[Dict[str, Any]] = []

        if panels_bbox:
            for panel_name, bbox in panels_bbox.items():
                panel_img = img_array[bbox['y_min']:bbox['y_max'], bbox['x_min']:bbox['x_max']]
                ocr_data = predict_hi_mask(
                    image=panel_img,
                    amg_model=models.hi_sam_model,
                    parseq_model=models.parseq_model,
                    dpi=dpi,
                )
                for para in ocr_data.get('paragraphs', []):
                    ocr_results.append(
                        {
                            "Paragraph ID": para['id'],
                            "Text": para['text'],
                            "Bounding Box": {
                                "xmin": int(para['xmin'] + bbox['x_min']),
                                "ymin": int(para['ymin'] + bbox['y_min']),
                                "xmax": int(para['xmax'] + bbox['x_min']),
                                "ymax": int(para['ymax'] + bbox['y_min']),
                            },
                            "Font Size": para.get('font_size', '1.0 mm'),
                            "panel": panel_name,
                            "source": "hi_sam_parseq",
                        }
                    )
        else:
            ocr_data = predict_hi_mask(
                image=img_array,
                amg_model=models.hi_sam_model,
                parseq_model=models.parseq_model,
                dpi=dpi,
            )
            for para in ocr_data.get('paragraphs', []):
                ocr_results.append(
                    {
                        "Paragraph ID": int(para['id']),
                        "Text": para['text'],
                        "Bounding Box": {
                            "xmin": int(para['xmin']),
                            "ymin": int(para['ymin']),
                            "xmax": int(para['xmax']),
                            "ymax": int(para['ymax']),
                        },
                        "Font Size": para.get('font_size', '1.0 mm'),
                        "panel": "unknown",
                        "source": "hi_sam_parseq",
                    }
                )

        clear_gpu_memory()

        return {
            "ocr_results": ocr_results,
            "total_blocks": len(ocr_results),
            "panels": panels,
        }

    finally:
        try:
            os.unlink(pdf_path)
        except OSError:
            pass
