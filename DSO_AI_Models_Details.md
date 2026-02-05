# DSO AI Models Technical Details

## Overview of AI Components

The DSO project utilizes multiple AI models to achieve accurate detection and validation of both symbols and text on packaging materials. This document provides detailed information about these models, their implementation, and how they work together.

## Symbol Detection Model

### Model Architecture
- **Base Model**: YOLOv8 (You Only Look Once, version 8)
- **Model Type**: Object detection
- **File Location**: `models/symbol/best.pt` (PyTorch) and `models/symbol/best.torchscript` (TorchScript)
- **Framework**: PyTorch/Ultralytics

### Training Details
- **Training Data**: Custom dataset of packaging symbols and regulatory marks
- **Classes**: Various regulatory symbols (CE mark, recycling symbols, warning labels, etc.)
- **Optimization**: Transfer learning from pre-trained YOLO weights

### Implementation Details
- **Inference Process**:
  ```python
  # Load model
  from ultralytics import YOLO
  symbol_model = YOLO('models/symbol/best.pt')
  
  # Run inference
  results = symbol_model(image)
  
  # Extract detections
  for box in results[0].boxes:
      x1, y1, x2, y2 = box.xyxy[0].tolist()
      cls_id = int(box.cls[0].item())
      label = results[0].names[cls_id]
      confidence = float(box.conf[0].item())
  ```

- **Tiling Implementation**: For large images, the system uses a tiling approach to maintain detection accuracy
  ```python
  # Tile parameters
  tile_size = 1280
  overlap = 200
  
  # Process tiles
  tiles_info = tile_image(image, tile_size, overlap)
  for tile_data in tiles_info:
      tile_img = tile_data['tile']
      offset_x = tile_data['offset_x']
      offset_y = tile_data['offset_y']
      
      # Run YOLO on tile
      results = yolo_model(tile_img)
      
      # Adjust coordinates to full image
      for box in results[0].boxes:
          x1_full = x1 + offset_x
          y1_full = y1 + offset_y
          x2_full = x2 + offset_x
          y2_full = y2 + offset_y
  ```

- **Duplicate Detection Handling**: DBSCAN clustering is used to group similar detections
  ```python
  from sklearn.cluster import DBSCAN
  import numpy as np
  
  # Group by label
  detections_by_label = {}
  for det in all_detections:
      label = det['label']
      if label not in detections_by_label:
          detections_by_label[label] = []
      detections_by_label[label].append(det)
  
  # Apply clustering per label
  final_detections = []
  for label, dets in detections_by_label.items():
      if len(dets) == 1:
          final_detections.append(dets[0])
      else:
          # Use DBSCAN for clustering
          coords = np.array([[d['box_pixels']['x1'], d['box_pixels']['y1']] for d in dets])
          clustering = DBSCAN(eps=50, min_samples=1).fit(coords)
          
          for cluster_id in set(clustering.labels_):
              cluster_dets = [dets[i] for i in range(len(dets)) if clustering.labels_[i] == cluster_id]
              # Keep detection with highest confidence
              best_det = max(cluster_dets, key=lambda x: x['confidence'])
              final_detections.append(best_det)
  ```

## Text Detection System

### Hi-SAM Model
- **Base Model**: Hierarchical Segment Anything Model
- **Purpose**: Text region segmentation
- **File Location**: `pretrained_checkpoint/hisam_state-001.pt`
- **Framework**: PyTorch

### Parseq OCR Model
- **Base Model**: Parseq (Paragraph Sequence Recognition)
- **Purpose**: Text recognition from segmented regions
- **File Location**: `pretrained_checkpoint/parseq_state.pt` and `pretrained_checkpoint/parseq_tokenizer.pkl`
- **Framework**: PyTorch

### Implementation Details
- **Hi-SAM Text Region Detection**:
  ```python
  # Load Hi-SAM model
  from hisam.hi_sam.modeling.build import model_registry
  from hisam.hi_sam.modeling.auto_mask_generator import AutoMaskGenerator
  
  # Initialize model
  hi_sam_model = model_registry["default"](checkpoint="pretrained_checkpoint/hisam_state-001.pt")
  hi_sam_model.to(device)
  
  # Generate masks
  mask_generator = AutoMaskGenerator(hi_sam_model)
  masks = mask_generator.generate(image)
  ```

- **Parseq OCR Processing**:
  ```python
  # Load Parseq model
  parseq_model = torch.hub.load('baudm/parseq', 'parseq', pretrained=True)
  parseq_model.load_state_dict(torch.load("pretrained_checkpoint/parseq_state.pt"))
  parseq_model.to(device)
  
  # Process text regions
  for mask in masks:
      # Extract text region
      text_region = extract_region(image, mask)
      
      # Perform OCR
      text = parseq_model(text_region)
  ```

- **Combined Text Detection Pipeline**:
  ```python
  def predict_hi_mask(image, amg_model, parseq_model, dpi=300):
      # Generate masks
      masks = amg_model.generate(image)
      
      # Process masks
      paragraphs = []
      for i, mask in enumerate(masks):
          # Extract region
          region = extract_region(image, mask)
          
          # OCR with Parseq
          text = parseq_model(region)
          
          # Calculate font size
          font_size = estimate_font_size(mask, dpi)
          
          paragraphs.append({
              'id': i,
              'text': text,
              'xmin': mask['bbox'][0],
              'ymin': mask['bbox'][1],
              'xmax': mask['bbox'][2],
              'ymax': mask['bbox'][3],
              'font_size': font_size
          })
      
      return {'paragraphs': paragraphs}
  ```

## Panel Detection (Optional Component)

- **Purpose**: Identify specific regions of the packaging (front panel, back panel, etc.)
- **Implementation**: Vision-Language Model (VLM) based approach
- **Process**:
  ```python
  # Panel detection
  panels = {}
  if detect_panels and panel_detector:
      # Save temp image
      temp_img_path = pdf_path.replace('.pdf', '_temp.jpg')
      img.save(temp_img_path)
      
      # Use panel detection
      panels_data = panel_detector.detect_panels(temp_img_path)
      
      if panels_data and 'panels' in panels_data:
          for panel in panels_data['panels']:
              panel_type = panel['type']
              bbox_percent = panel['bbox_percent']
              
              panels[panel_type] = {
                  'x_min': int(bbox_percent['x_min'] * img_width / 100),
                  'y_min': int(bbox_percent['y_min'] * img_height / 100),
                  'x_max': int(bbox_percent['x_max'] * img_width / 100),
                  'y_max': int(bbox_percent['y_max'] * img_height / 100)
              }
  ```

## Validation Engine

### Symbol Validation
- **Input**: Detected symbols with bounding boxes and dimensions
- **Process**: Compares detected symbols against regulatory requirements
- **Output**: Compliance status for each requirement

### Text Validation
- **Input**: Detected text blocks with content and font sizes
- **Process**: Compares detected text against regulatory requirements
- **Output**: Compliance status for each requirement

### Implementation Details
```python
# Symbol validation
validation_response = requests.post(
    f"{SYMBOL_API}/validate_symbols",
    json={
        'detections': symbol_detections,
        'country': "Default",
        'product_metadata': {
            'type': product_type,
            'width_cm': packaging_width,
            'height_cm': packaging_height
        }
    }
)

# Text validation
validation_response = requests.post(
    f"{TEXT_API}/validate_text",
    json={
        'ocr_results': text_detections,
        'country': "Default",
        'product_metadata': {
            'type': product_type,
            'width_cm': packaging_width,
            'height_cm': packaging_height
        }
    }
)
```

## Performance Metrics

### Symbol Detection
- **Average Precision**: ~85-90% on test dataset
- **Inference Time**: ~1-3 seconds per image (depending on size and GPU)
- **Minimum Symbol Size**: Can detect symbols as small as 5mm

### Text Detection
- **Text Recognition Accuracy**: ~92-95% on clear text
- **Font Size Estimation Accuracy**: ±0.5mm
- **Inference Time**: ~3-5 seconds per image (depending on text density)

## Model Limitations

1. **Symbol Detection**:
   - Limited to trained symbol classes
   - May struggle with highly distorted or partially occluded symbols
   - Performance degrades with very low-resolution images

2. **Text Detection**:
   - Challenges with highly stylized fonts
   - May struggle with text on complex backgrounds
   - Limited language support (primarily focused on Latin scripts)

3. **General Limitations**:
   - Single-page processing only
   - GPU memory requirements can be high
   - Processing time increases with image resolution

## Future Model Improvements

1. **Symbol Detection**:
   - Expand training dataset with more symbol variations
   - Implement instance segmentation for more precise symbol boundary detection
   - Add support for symbol color validation

2. **Text Detection**:
   - Improve multi-language support
   - Enhance font style and size estimation
   - Implement text orientation detection for rotated text

3. **System Integration**:
   - End-to-end model that combines symbol and text detection
   - On-device optimization for faster inference
   - Support for multi-page document processing
