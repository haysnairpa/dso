from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class LoadedSymbolModels:
    yolo_model: Any
    legal_rules_logo: Optional[Any]


def load_yolo_model(*, model_path: str) -> Any:
    from ultralytics import YOLO

    return YOLO(model_path)


def detect_symbols_from_pil_image(
    *,
    yolo_model: Any,
    image,
    dpi: int,
    confidence_threshold: float,
    tile_size: int = 1280,
    overlap: int = 200,
) -> Dict[str, Any]:
    import numpy as np
    from sklearn.cluster import DBSCAN

    tiles_info = tile_image(image, tile_size=tile_size, overlap=overlap)

    all_detections: List[Dict[str, Any]] = []

    for tile_data in tiles_info:
        tile_img = tile_data["tile"]
        offset_x = tile_data["offset_x"]
        offset_y = tile_data["offset_y"]

        results = yolo_model(tile_img, conf=confidence_threshold)
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0].cpu().numpy())
                cls = int(box.cls[0].cpu().numpy())
                label = result.names[cls]

                x1_full = x1 + offset_x
                y1_full = y1 + offset_y
                x2_full = x2 + offset_x
                y2_full = y2 + offset_y

                width_pixels = float(x2_full - x1_full)
                height_pixels = float(y2_full - y1_full)
                width_mm = (width_pixels / dpi) * 25.4
                height_mm = (height_pixels / dpi) * 25.4

                color_stats = None
                try:
                    x1_crop = max(0, int(x1_full))
                    y1_crop = max(0, int(y1_full))
                    x2_crop = min(image.width, int(x2_full))
                    y2_crop = min(image.height, int(y2_full))

                    crop = image.crop((x1_crop, y1_crop, x2_crop, y2_crop))
                    if crop.mode != "RGB":
                        crop = crop.convert("RGB")
                    crop_np = np.array(crop)
                    if crop_np.ndim == 3 and crop_np.shape[2] >= 3 and crop_np.size:
                        pixels = crop_np[:, :, :3].reshape(-1, 3).astype(np.float32)

                        avg = pixels.mean(axis=0)
                        foreground_rgb = [int(avg[0]), int(avg[1]), int(avg[2])]
                        background_rgb = [255, 255, 255]

                        def _channel_lum(c: float) -> float:
                            c /= 255.0
                            return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4

                        def _relative_luminance(rgb):
                            r, g, b = [_channel_lum(float(v)) for v in rgb]
                            return 0.2126 * r + 0.7152 * g + 0.0722 * b

                        def _contrast_ratio(a, b) -> float:
                            lum_a = _relative_luminance(a)
                            lum_b = _relative_luminance(b)
                            lighter = max(lum_a, lum_b)
                            darker = min(lum_a, lum_b)
                            return (lighter + 0.05) / (darker + 0.05)

                        color_stats = {
                            "foreground_rgb": foreground_rgb,
                            "background_rgb": background_rgb,
                            "contrast_ratio": float(_contrast_ratio(foreground_rgb, background_rgb)),
                        }
                except Exception:
                    color_stats = None

                all_detections.append(
                    {
                        "label": label,
                        "confidence": conf,
                        "panel_id": "unknown",
                        "box_pixels": {
                            "x": float(x1_full),
                            "y": float(y1_full),
                            "width": float(width_pixels),
                            "height": float(height_pixels),
                        },
                        "box_mm": {
                            "x1": float((x1_full / dpi) * 25.4),
                            "y1": float((y1_full / dpi) * 25.4),
                            "x2": float((x2_full / dpi) * 25.4),
                            "y2": float((y2_full / dpi) * 25.4),
                            "width": float(width_mm),
                            "height": float(height_mm),
                        },
                        "width_pixels": width_pixels,
                        "height_pixels": height_pixels,
                        "color_stats": color_stats,
                        "patch_path": None,
                    }
                )

    final_detections: List[Dict[str, Any]] = []
    if len(all_detections) > 0:
        detections_by_label: Dict[str, List[Dict[str, Any]]] = {}
        for det in all_detections:
            label = det["label"]
            detections_by_label.setdefault(label, []).append(det)

        for label, dets in detections_by_label.items():
            if len(dets) == 1:
                final_detections.append(dets[0])
                continue

            coords = np.array([[d["box_pixels"]["x"], d["box_pixels"]["y"]] for d in dets])
            clustering = DBSCAN(eps=50, min_samples=1).fit(coords)
            for cluster_id in set(clustering.labels_):
                cluster_dets = [dets[i] for i in range(len(dets)) if clustering.labels_[i] == cluster_id]
                best_det = max(cluster_dets, key=lambda x: x["confidence"])
                final_detections.append(best_det)

    return {
        "detections": final_detections,
        "total_detections": len(final_detections),
        "image_size": {"width": image.width, "height": image.height},
    }


def tile_image(image, tile_size: int, overlap: int) -> List[Dict[str, Any]]:
    tiles_info: List[Dict[str, Any]] = []

    width, height = image.size
    step = tile_size - overlap

    for y in range(0, height, step):
        for x in range(0, width, step):
            tile = image.crop((x, y, min(x + tile_size, width), min(y + tile_size, height)))
            tiles_info.append({"tile": tile, "offset_x": x, "offset_y": y})

    return tiles_info
