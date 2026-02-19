from __future__ import annotations

import base64
import json
import re
from typing import Any, Dict, Optional

from openai import OpenAI


class VLMPanelDetector:
    """Uses Vision Language Model (GPT-4 Vision) to detect packaging panels."""

    def __init__(self, api_key: str, model: str = "gpt-4o"):
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def encode_image(self, image_path: str) -> str:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    def detect_packaging_region(self, image_path: str) -> Optional[Dict[str, Any]]:
        base64_image = self.encode_image(image_path)

        prompt = """Analyze this image and identify ONLY the actual PACKAGING region.

                  **CRITICAL: Identify ONLY the packaging that will be physically printed and manufactured.**

                  **EXCLUDE (DO NOT INCLUDE):**
                  - Mockup previews (areas labeled "MOCK UP")
                  - Approval lists (areas with "Approver List", names, signatures)
                  - Job information headers (areas with "Job Name", "Pack No", "Rev")
                  - Metadata tables (gray areas with technical info)
                  - Design annotations or notes
                  - Any text outside the main packaging design

                  **INCLUDE ONLY:**
                  - The actual packaging design that will be printed
                  - Usually the main colored area (often in center or left side)
                  - The packaging template that shows how it will be folded
                  - Areas with product branding, legal text, barcodes that are PART of the packaging

                  **Return ONLY valid JSON:**
                  {
                    "packaging_region": {
                      "x_min": 10,
                      "y_min": 5,
                      "x_max": 60,
                      "y_max": 95,
                      "description": "Main packaging template area"
                    }
                  }

                  **Use percentage coordinates (0-100) relative to full image dimensions.**
                  """

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}", "detail": "high"}},
                ],
            }],
            max_tokens=500,
            temperature=0.0,
        )

        result_text = response.choices[0].message.content
        json_match = re.search(r"\{.*\}", result_text or "", re.DOTALL)
        if not json_match:
            return None
        return json.loads(json_match.group())

    def detect_panels(self, image_path: str) -> Optional[Dict[str, Any]]:
        base64_image = self.encode_image(image_path)

        prompt = """Analyze this packaging image and identify the different panels.

                **Your task:**
                1. Identify the **PDP (Principal Display Panel)** - the main front panel with product branding, logo, and product image
                2. Identify the **Bottom Panel** - the bottom flap/section of the packaging (usually has barcode)
                3. Identify the **Back Panel** - the back side with legal text, warnings, and instructions
                4. Identify any **Side Panels** or **Top Panels**

                For each panel, provide the bounding box coordinates as percentages of the image dimensions.

                **Return ONLY valid JSON in this exact format:**
                {
                  "panels": [
                    {
                      "type": "pdp",
                      "bbox_percent": {"x_min": 10, "y_min": 15, "x_max": 90, "y_max": 65},
                      "description": "Front panel"
                    }
                  ],
                  "image_dimensions": {"width": 1000, "height": 1500}
                }

                **Important:**
                - Use percentage coordinates (0-100) relative to image dimensions
                - Be precise with panel boundaries
                - Include all visible panels
                - PDP is the most prominent front-facing panel
                - Bottom panel is typically at the bottom with barcode
                """

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}", "detail": "high"}},
                ],
            }],
            max_tokens=1000,
            temperature=0.0,
        )

        result_text = response.choices[0].message.content
        json_match = re.search(r"\{.*\}", result_text or "", re.DOTALL)
        if not json_match:
            return None
        return json.loads(json_match.group())

    def get_text_panel(self, text_bbox: dict, panels_data: dict, image_height: int, image_width: int) -> str:
        if not panels_data or "panels" not in panels_data:
            return "unknown"

        text_x_center = ((text_bbox["xmin"] + text_bbox["xmax"]) / 2) / image_width * 100
        text_y_center = ((text_bbox["ymin"] + text_bbox["ymax"]) / 2) / image_height * 100

        for panel in panels_data.get("panels", []):
            bbox = panel.get("bbox_percent", {})
            if (
                bbox.get("x_min", 0) <= text_x_center <= bbox.get("x_max", 100)
                and bbox.get("y_min", 0) <= text_y_center <= bbox.get("y_max", 100)
            ):
                return panel.get("type", "unknown")

        return "unknown"
