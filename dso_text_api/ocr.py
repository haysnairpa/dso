from __future__ import annotations

import re
from typing import Dict, List

import cv2
import numpy as np
import pyclipper
import torch
from PIL import Image
from shapely.geometry import Polygon
from torchvision import transforms as T


def unclip(p, unclip_ratio: float = 2.0):
    poly = Polygon(p)
    distance = poly.area * unclip_ratio / poly.length
    offset = pyclipper.PyclipperOffset()
    offset.AddPath(p, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
    expanded = offset.Execute(distance)
    if not expanded:
        return []
    return [np.array(poly).astype(np.int32) for poly in expanded]


def finding_region_from_points(points):
    x_coords, y_coords = [p[0] for p in points], [p[1] for p in points]
    return min(x_coords), min(y_coords), max(x_coords), max(y_coords)


def build_parseq_img_transform():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    img_transform = T.Compose([
        T.Resize((32, 128), T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=(0.694, 0.695, 0.693), std=(0.299, 0.301, 0.301)),
    ])
    return img_transform, device


def scr_predict(img_crop: np.ndarray, parseq_model, img_transform, device: str) -> str:
    if img_crop.size == 0:
        return ""
    try:
        img = Image.fromarray(img_crop)
        img_tensor = img_transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = parseq_model(img_tensor)
            pred = logits.softmax(-1)
            label, _ = parseq_model.tokenizer.decode(pred)
        return label[0] if label else ""
    except Exception:
        return ""


class DisjointSet:
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x: int, y: int) -> None:
        px, py = self.find(x), self.find(y)
        if px == py:
            return
        if self.rank[px] < self.rank[py]:
            px, py = py, px
        self.parent[py] = px
        if self.rank[px] == self.rank[py]:
            self.rank[px] += 1

    def to_group(self):
        groups = {}
        for i in range(len(self.parent)):
            root = self.find(i)
            groups.setdefault(root, []).append(i)
        return list(groups.values())


def predict_hi_mask(image: np.ndarray, amg_model, parseq_model, dpi: int) -> Dict[str, List[Dict]]:
    img_transform, device = build_parseq_img_transform()

    img_h, img_w = image.shape[:2]
    amg_model.set_image(image)
    masks, scores, affinity = amg_model.predict(
        from_low_res=False,
        fg_points_num=2100,
        batch_points_num=100,
        score_thresh=0.7,
        nms_thresh=0.6,
    )
    if masks is None:
        return {"lines": [], "paragraphs": []}

    masks = (masks[:, 0, :, :]).astype(np.uint8)
    lines, line_indices = [], []

    for index, mask in enumerate(masks):
        contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue

        words_in_line, heights = [], []
        line_bbox = {"xmin": 1e9, "ymin": 1e9, "xmax": 0, "ymax": 0}

        for cont in contours:
            approx = cv2.approxPolyDP(cont, 0.002 * cv2.arcLength(cont, True), True)
            points = approx.reshape((-1, 2))
            if points.shape[0] < 4:
                continue

            pts = unclip(points)
            if len(pts) != 1:
                continue

            pts = pts[0].astype(np.int32)
            if Polygon(pts).area < 100:
                continue

            pts[:, 0] = np.clip(pts[:, 0], 0, img_w)
            pts[:, 1] = np.clip(pts[:, 1], 0, img_h)
            xmin, ymin, xmax, ymax = finding_region_from_points(pts)

            line_bbox["xmin"] = min(line_bbox["xmin"], xmin)
            line_bbox["ymin"] = min(line_bbox["ymin"], ymin)
            line_bbox["xmax"] = max(line_bbox["xmax"], xmax)
            line_bbox["ymax"] = max(line_bbox["ymax"], ymax)

            text = scr_predict(image[ymin:ymax + 1, xmin:xmax + 1], parseq_model, img_transform, device)
            if not text or len(text) < 2:
                continue

            if len(text) > 2:
                alnum_ratio = sum(c.isalnum() or c.isspace() for c in text) / len(text)
                if alnum_ratio < 0.5:
                    continue

            heights.append(ymax - ymin)
            words_in_line.append({"text": text, "xmin": xmin, "ymin": ymin, "xmax": xmax, "ymax": ymax})

        if not words_in_line:
            continue

        sorted_words = sorted(words_in_line, key=lambda x: x["xmin"])
        line_text = " ".join(word["text"] for word in sorted_words if word.get("text"))
        avg_height_px = sum(heights) / len(heights)
        font_size_cm = (avg_height_px / dpi) * 2.54

        lines.append(
            {
                "text": line_text,
                "xmin": line_bbox["xmin"],
                "ymin": line_bbox["ymin"],
                "xmax": line_bbox["xmax"],
                "ymax": line_bbox["ymax"],
                "font_size": font_size_cm * 10,
            }
        )
        line_indices.append(index)

    if not lines:
        return {"lines": [], "paragraphs": []}

    line_grouping = DisjointSet(len(line_indices))
    affinity = affinity[line_indices][:, line_indices]

    for i1, i2 in zip(*np.where(affinity > 0.3)):
        line_grouping.union(int(i1), int(i2))

    line_groups = line_grouping.to_group()

    paragraphs = []
    for para_index, line_group in enumerate(line_groups):
        para_texts = []
        para_bbox = {"xmin": 1e9, "ymin": 1e9, "xmax": 0, "ymax": 0}
        font_sizes = []

        group_lines = [lines[i] for i in line_group]
        sorted_lines = sorted(group_lines, key=lambda x: x["ymin"])

        for line in sorted_lines:
            if line.get("text"):
                para_texts.append(line["text"])

            para_bbox["xmin"] = min(para_bbox["xmin"], line["xmin"])
            para_bbox["ymin"] = min(para_bbox["ymin"], line["ymin"])
            para_bbox["xmax"] = max(para_bbox["xmax"], line["xmax"])
            para_bbox["ymax"] = max(para_bbox["ymax"], line["ymax"])
            font_sizes.append(line["font_size"])

        if not para_texts:
            continue

        clean_para_text = " ".join(" ".join(para_texts).split())
        paragraphs.append(
            {
                "id": para_index,
                "text": clean_para_text,
                "xmin": para_bbox["xmin"],
                "ymin": para_bbox["ymin"],
                "xmax": para_bbox["xmax"],
                "ymax": para_bbox["ymax"],
                "font_size": f"{sum(font_sizes) / len(font_sizes):.4f} mm",
            }
        )

    return {"lines": lines, "paragraphs": paragraphs}


def load_models(*, hi_sam_checkpoint: str):
    from hisam.hi_sam.modeling.build import model_registry
    from hisam.hi_sam.modeling.auto_mask_generator import AutoMaskGenerator

    class args_hisam:
        model_type = 'vit_h'
        checkpoint = hi_sam_checkpoint
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        input_size = [1024, 1024]
        hier_det = True
        attn_layers = 1
        prompt_len = 12
        layout_thresh = 0.5

    args = args_hisam()
    hisam = model_registry[args.model_type](args)
    hisam.eval()
    hisam.to(args.device)
    amg = AutoMaskGenerator(hisam)

    parseq = torch.hub.load('baudm/parseq', 'parseq', pretrained=True, map_location='cpu').eval()
    parseq = parseq.to(args.device).eval()

    return amg, parseq, args.device
