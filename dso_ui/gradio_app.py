import os
from typing import Optional

import base64
import io
import json

import fitz
import gradio as gr
import numpy as np
import requests
from PIL import Image, ImageDraw


SYMBOL_API = os.environ.get("SYMBOL_API", "http://localhost:5000")
TEXT_API = os.environ.get("TEXT_API", "http://localhost:5001")


def _safe_json(resp: requests.Response) -> dict:
    try:
        return resp.json()
    except Exception:
        return {"status": "error", "message": f"Non-JSON response ({resp.status_code})", "raw": resp.text[:2000]}


def generate_symbol_visualization(pdf_bytes: bytes, symbol_detections: list, dpi: int = 300) -> Image.Image:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    page = doc[0]
    pix = page.get_pixmap(dpi=dpi)
    img_data = pix.tobytes("png")
    img = Image.open(io.BytesIO(img_data)).convert("RGB")
    doc.close()

    draw = ImageDraw.Draw(img)

    for det in symbol_detections or []:
        bbox = det.get("bbox") or det.get("box") or det.get("xyxy")
        if not bbox or len(bbox) != 4:
            continue
        x1, y1, x2, y2 = bbox
        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)

    return img


def generate_text_visualization(pdf_bytes: bytes, text_detections: list, dpi: int = 300) -> Image.Image:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    page = doc[0]
    pix = page.get_pixmap(dpi=dpi)
    img_data = pix.tobytes("png")
    img = Image.open(io.BytesIO(img_data)).convert("RGB")
    doc.close()

    draw = ImageDraw.Draw(img)

    for det in text_detections or []:
        bbox = det.get("bbox") or det.get("box") or det.get("xyxy")
        if not bbox or len(bbox) != 4:
            continue
        x1, y1, x2, y2 = bbox
        draw.rectangle([x1, y1, x2, y2], outline="green", width=3)

    return img


def run_validation(
    excel_file,  # tempfile.NamedTemporaryFile from gradio
    pdf_file,
    dpi: int,
    confidence_threshold: float,
    product_type: str,
    packaging_width: float,
    packaging_height: float,
):
    if excel_file is None or pdf_file is None:
        return "Missing Excel or PDF", None, None

    with open(excel_file.name, "rb") as f:
        resp = requests.post(f"{SYMBOL_API}/convert_excel_logo", files={"excel": f}, timeout=600)
    if resp.status_code != 200:
        j = _safe_json(resp)
        return f"Symbol Excel conversion failed: {j.get('message', 'Unknown error')}", None, None

    with open(excel_file.name, "rb") as f:
        resp = requests.post(f"{TEXT_API}/convert_excel_text", files={"excel": f}, timeout=600)
    if resp.status_code != 200:
        j = _safe_json(resp)
        return f"Text Excel conversion failed: {j.get('message', 'Unknown error')}", None, None

    resp = requests.post(f"{SYMBOL_API}/load_models", timeout=60)
    if resp.status_code != 200:
        j = _safe_json(resp)
        return f"Symbol model loading failed: {j.get('message', 'Unknown error')}", None, None

    resp = requests.post(f"{TEXT_API}/load_models", timeout=180)
    if resp.status_code != 200:
        j = _safe_json(resp)
        return f"Text model loading failed: {j.get('message', 'Unknown error')}", None, None

    with open(pdf_file.name, "rb") as f:
        pdf_bytes = f.read()

    with open(pdf_file.name, "rb") as f:
        files = {"pdf": f}
        data = {"dpi": str(dpi), "confidence_threshold": str(confidence_threshold)}
        resp = requests.post(f"{SYMBOL_API}/detect_symbols", files=files, data=data, timeout=180)
    if resp.status_code != 200:
        j = _safe_json(resp)
        return f"Symbol detection failed: {j.get('message', 'Unknown error')}", None, None

    detections_payload = _safe_json(resp)
    symbol_detections = detections_payload.get("detections", [])
    symbol_viz = generate_symbol_visualization(pdf_bytes, symbol_detections, dpi=dpi)

    resp = requests.post(
        f"{SYMBOL_API}/validate_symbols",
        json={
            "detections": symbol_detections,
            "country": "Default",
            "product_metadata": {"type": product_type, "width_cm": packaging_width, "height_cm": packaging_height},
        },
        timeout=300,
    )
    if resp.status_code != 200:
        j = _safe_json(resp)
        return f"Symbol validation failed: {j.get('message', 'Unknown error')}", symbol_viz, None

    symbol_validation = _safe_json(resp)

    try:
        requests.post(f"{SYMBOL_API}/clear_gpu", timeout=10)
    except Exception:
        pass

    with open(pdf_file.name, "rb") as f:
        files = {"pdf": f}
        data = {"dpi": str(dpi), "detect_panels": "false"}
        resp = requests.post(f"{TEXT_API}/detect_text", files=files, data=data, timeout=180)
    if resp.status_code != 200:
        j = _safe_json(resp)
        return f"Text detection failed: {j.get('message', 'Unknown error')}", symbol_viz, None

    ocr_payload = _safe_json(resp)
    text_detections = ocr_payload.get("ocr_results", [])
    text_viz = generate_text_visualization(pdf_bytes, text_detections, dpi=dpi)

    resp = requests.post(
        f"{TEXT_API}/validate_text",
        json={
            "ocr_results": text_detections,
            "country": "Default",
            "product_metadata": {"type": product_type, "width_cm": packaging_width, "height_cm": packaging_height},
        },
        timeout=300,
    )
    if resp.status_code != 200:
        j = _safe_json(resp)
        return f"Text validation failed: {j.get('message', 'Unknown error')}", symbol_viz, text_viz

    text_validation = _safe_json(resp)

    report = {
        "symbol": {"detections": detections_payload, "validation": symbol_validation},
        "text": {"ocr": ocr_payload, "validation": text_validation},
    }

    return json.dumps(report, indent=2)[:200000], symbol_viz, text_viz


def build_demo() -> gr.Blocks:
    with gr.Blocks(title="Packaging Compliance Validator") as demo:
        gr.Markdown("# Packaging Compliance Validator")

        with gr.Row():
            excel = gr.File(label="Excel (rules)", file_types=[".xlsx"])
            pdf = gr.File(label="PDF (packaging)", file_types=[".pdf"])

        with gr.Row():
            dpi = gr.Slider(72, 600, value=300, step=1, label="DPI")
            conf = gr.Slider(0.01, 0.99, value=0.25, step=0.01, label="Symbol confidence threshold")

        with gr.Row():
            product_type = gr.Textbox(value="Default", label="Product type")
            width_cm = gr.Number(value=10.0, label="Packaging width (cm)")
            height_cm = gr.Number(value=10.0, label="Packaging height (cm)")

        btn = gr.Button("Run validation")

        with gr.Row():
            out_json = gr.Textbox(label="Report (JSON)", lines=20)

        with gr.Row():
            out_symbol_viz = gr.Image(label="Symbol visualization")
            out_text_viz = gr.Image(label="Text visualization")

        btn.click(
            fn=run_validation,
            inputs=[excel, pdf, dpi, conf, product_type, width_cm, height_cm],
            outputs=[out_json, out_symbol_viz, out_text_viz],
        )

    return demo


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "7860"))
    demo = build_demo()
    demo.launch(server_name="0.0.0.0", server_port=port)
