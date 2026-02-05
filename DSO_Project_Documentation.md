# DSO Project Documentation

## Project Overview

The DSO (Document Symbol and OCR) project is an AI-driven system for automated validation of packaging compliance. The system analyzes packaging documents (PDFs) to detect and validate both symbols/logos and text content against regulatory requirements. This ensures packaging meets legal standards across different countries and regions.

## Key Components

### 1. Symbol Detection System
- **Purpose**: Detects and validates regulatory symbols, logos, and marks on packaging
- **Technology**: YOLO (You Only Look Once) object detection model
- **Model File**: `models/symbol/best.pt` (PyTorch) and `models/symbol/best.torchscript` (TorchScript)
- **Functionality**:
  - Detects symbols like CE mark, recycling symbols, warning labels, etc.
  - Measures symbol dimensions to ensure compliance with size requirements
  - Validates symbol presence against country-specific regulatory requirements

### 2. Text Detection System
- **Purpose**: Detects and validates legal text on packaging
- **Technology**: Hi-SAM (Hierarchical Segment Anything Model) + Parseq OCR
- **Model Files**: 
  - `pretrained_checkpoint/hisam_state-001.pt`
  - `pretrained_checkpoint/parseq_state.pt`
  - `pretrained_checkpoint/parseq_tokenizer.pkl`
- **Functionality**:
  - Identifies text blocks on packaging
  - Performs OCR to extract text content
  - Validates text against regulatory requirements (warnings, legal statements, etc.)
  - Measures font sizes to ensure readability compliance

### 3. Validation Engine
- **Purpose**: Checks detected symbols and text against regulatory requirements
- **Data Source**: Excel files with regulatory requirements (`dso.xlsx`)
- **Functionality**:
  - Converts Excel regulatory data to JSON format for processing
  - Validates detected elements against requirements
  - Generates compliance reports with pass/fail status

### 4. User Interface
- **Technology**: Gradio web interface
- **Features**:
  - Simple 3-step process: Upload Excel (requirements), Upload PDF (packaging), Validate
  - Visual results showing detected symbols (red boxes) and text (green boxes)
  - Compliance summary with pass/fail status
  - Detailed JSON output for technical users

## Workflow

1. **Data Preparation**:
   - Regulatory requirements are stored in Excel format (`dso.xlsx`)
   - The system converts these requirements to JSON format for processing

2. **Model Loading**:
   - Symbol detection model (YOLO)
   - Text detection models (Hi-SAM + Parseq)

3. **Document Processing**:
   - PDF is converted to image at specified DPI
   - For large images, tiling is applied to improve detection accuracy

4. **Symbol Detection**:
   - YOLO model detects symbols and logos
   - Non-Maximum Suppression (NMS) is applied to remove duplicate detections
   - Symbol dimensions are calculated in pixels and mm

5. **Text Detection**:
   - Hi-SAM segments text regions
   - Parseq OCR extracts text content
   - Text blocks are organized with bounding boxes and metadata

6. **Validation**:
   - Detected symbols and text are validated against regulatory requirements
   - Compliance report is generated with pass/fail status
   - Visual output shows detected elements on the packaging

7. **Results Presentation**:
   - Summary compliance report
   - Detailed JSON output
   - Visual representation of detected elements

## Technical Architecture

### Backend Services
- Symbol detection API (YOLO-based)
- Text detection API (Hi-SAM + Parseq)
- Both services expose RESTful endpoints

### Frontend
- Gradio web interface for user interaction
- Simple workflow with progress indicators

### Data Flow
1. User uploads Excel requirements and PDF packaging
2. System converts Excel to JSON rules
3. PDF is processed for symbol and text detection
4. Detected elements are validated against rules
5. Results are presented to the user

## Models and AI Components

### Symbol Detection (YOLO)
- Custom-trained YOLO model for packaging symbols
- Detects regulatory marks, logos, and warning symbols
- Provides bounding boxes, confidence scores, and dimensional measurements

### Text Detection (Hi-SAM + Parseq)
- Hi-SAM: Segments text regions hierarchically
- Parseq: Performs OCR on segmented regions
- Combined approach provides accurate text extraction with spatial context

## Deployment

The system is deployed as two separate services:
- Symbol detection service (accessible via API)
- Text detection service (accessible via API)

A Gradio web interface connects to both services to provide a unified user experience.

## Future Enhancements

1. **Multi-page Support**: Currently processes only the first page of PDFs
2. **Additional Languages**: Expand text recognition capabilities
3. **More Regulatory Regions**: Add support for additional country-specific requirements
4. **Integration with PLM Systems**: Connect with Product Lifecycle Management systems
5. **Automated Correction Suggestions**: Provide recommendations to fix compliance issues

## Conclusion

The DSO project provides an automated solution for packaging compliance validation, reducing manual review time and ensuring regulatory requirements are met across different markets. By combining advanced computer vision and OCR technologies, the system delivers accurate detection and validation of both symbols and text on packaging materials.
