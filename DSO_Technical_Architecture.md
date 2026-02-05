# DSO Technical Architecture

## System Architecture Overview

The DSO (Document Symbol and OCR) system is built with a microservices architecture consisting of two primary services that work together to provide a comprehensive packaging compliance validation solution.

```
┌───────────────────┐     ┌───────────────────┐
│                   │     │                   │
│  Symbol Detection │     │  Text Detection   │
│      Service      │     │      Service      │
│                   │     │                   │
└─────────┬─────────┘     └─────────┬─────────┘
          │                         │
          │                         │
          ▼                         ▼
┌─────────────────────────────────────────────┐
│                                             │
│            Gradio Web Interface             │
│                                             │
└─────────────────────────────────────────────┘
```

## Symbol Detection Service

### Components

- **YOLO Model**: Custom-trained YOLOv8 model for detecting regulatory symbols and logos
- **Symbol Validation Engine**: Validates detected symbols against regulatory requirements
- **Flask API Server**: Exposes endpoints for model loading, detection, and validation

### API Endpoints

- `/load_models`: Loads the YOLO model into memory
- `/detect_symbols`: Processes a PDF and returns detected symbols with bounding boxes
- `/validate_symbols`: Validates detected symbols against regulatory requirements
- `/clear_gpu`: Clears GPU memory to optimize resource usage

### Detection Process

1. PDF is converted to image at specified DPI
2. For large images, tiling is applied (1280px tiles with 200px overlap)
3. YOLO model processes each tile
4. Detections are combined and duplicates removed using DBSCAN clustering
5. Symbol dimensions are calculated in both pixels and millimeters

## Text Detection Service

### Components

- **Hi-SAM Model**: Hierarchical Segment Anything Model for text region segmentation
- **Parseq OCR Engine**: OCR model for text recognition
- **Text Validation Engine**: Validates detected text against regulatory requirements
- **Flask API Server**: Exposes endpoints for model loading, detection, and validation

### API Endpoints

- `/load_models`: Loads Hi-SAM and Parseq models into memory
- `/detect_text`: Processes a PDF and returns detected text with bounding boxes
- `/validate_text`: Validates detected text against regulatory requirements

### Detection Process

1. PDF is converted to image at specified DPI
2. Optional panel detection to identify specific regions of the packaging
3. Hi-SAM segments text regions hierarchically
4. Parseq OCR extracts text content from segmented regions
5. Text blocks are organized with bounding boxes, font sizes, and panel information

## Data Processing Pipeline

### Excel to JSON Conversion

- **Symbol Requirements**: Converts "Logo List - 21A" sheet to structured JSON
- **Text Requirements**: Converts "Text Availability - 21A" sheet to structured JSON

### PDF Processing

- Uses PyMuPDF (fitz) for PDF to image conversion
- Supports various DPI settings for different quality requirements

### Validation Logic

- Symbol validation checks:

  - Presence of required symbols
  - Symbol dimensions against minimum size requirements
  - Symbol placement (if panel detection is enabled)
- Text validation checks:

  - Presence of required text
  - Font size against minimum requirements
  - Language-specific text requirements

## Resource Management

### GPU Memory Optimization

- Explicit GPU memory clearing between detection tasks
- Model unloading when not in use
- Tiling for large images to reduce memory requirements

### Error Handling

- Robust error handling with detailed logging
- Retry mechanism for network-related failures
- Graceful degradation when resources are constrained

## Deployment Configuration

### Symbol Detection Service

- Deployed via ngrok: `https://ae9299c4f2d0.ngrok-free.app`
- Requires GPU for optimal performance

### Text Detection Service

- Deployed via ngrok: `https://b8bbb7ea43a3.ngrok-free.app`
- Requires GPU with at least 8GB VRAM for Hi-SAM model

### Gradio Interface

- Deployed locally on port 7860
- Connects to both services via their respective APIs

## Performance Considerations

### Optimization Techniques

- Image tiling for efficient processing of large PDFs
- Non-Maximum Suppression (NMS) for reducing duplicate detections
- DBSCAN clustering for merging similar detections
- Panel detection to focus processing on relevant areas

### Resource Requirements

- **GPU**: NVIDIA GPU with at least 8GB VRAM recommended
- **RAM**: Minimum 16GB system RAM
- **Storage**: ~5GB for models and temporary files

## Security Considerations

- Input validation for all API endpoints
- Temporary file cleanup after processing
- No persistent storage of customer data
- API endpoints do not require authentication (intended for internal use only)

## Integration Points

### Input Integration

- Excel files with regulatory requirements
- PDF files with packaging designs

### Output Integration

- JSON results for programmatic consumption
- Visual results for human review
- Summary reports for compliance documentation
