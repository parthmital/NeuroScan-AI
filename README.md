# Neuroscan AI

A comprehensive medical imaging platform for brain tumor analysis using advanced deep learning models. This full-stack application combines a powerful FastAPI backend with a modern React frontend to provide automated detection, classification, and segmentation of brain tumors from MRI scans.

## Overview

Neuroscan AI is a production-ready medical imaging analysis platform that combines three specialized deep learning models with an intuitive web interface to provide comprehensive brain tumor analysis:

- **Tumor Detection**: Binary classification to identify presence/absence of tumors (80.07% accuracy, 85.99% AUC)
- **Tumor Classification**: Multi-class classification (glioma, meningioma, pituitary, no tumor) with 87.41% accuracy
- **Tumor Segmentation**: 3D volumetric segmentation for precise tumor delineation (WT: 84.7%, TC: 75.3%, ET: 76.1% Dice)

## Architecture

The application follows a modern full-stack architecture with clear separation of concerns:

### Backend (FastAPI)

- **FastAPI Web Framework**: High-performance async API server
- **SQLModel Database**: SQLite-based data persistence with async support
- **JWT Authentication**: Secure token-based user authentication
- **AI Pipeline**: Integrated deep learning models for comprehensive analysis
- **File Management**: Structured upload and processing of MRI files

### Frontend (React)

- **React 18.3.1**: Modern React with TypeScript for type-safe development
- **Medical Imaging**: Cornerstone.js and VTK.js for professional visualization
- **Real-time Updates**: Live processing status and progress tracking
- **Responsive Design**: Mobile-first design with dark/light theme support

## AI Models

| Model              | Architecture                 | Task                          | Performance                          |
| ------------------ | ---------------------------- | ----------------------------- | ------------------------------------ |
| **Detection**      | ResNet50 (Transfer Learning) | Binary Tumor Detection        | 80.07% Accuracy, 85.99% AUC          |
| **Classification** | ResNet50 (Multi-class)       | Tumor Type Classification     | 87.41% Accuracy                      |
| **Segmentation**   | 3D nnU-Net                   | Volumetric Tumor Segmentation | WT: 84.7%, TC: 75.3%, ET: 76.1% Dice |

## Key Features

### Medical Imaging

- **Multi-planar Views**: Axial, sagittal, and coronal planes
- **Window/Level Adjustment**: Interactive contrast and brightness controls
- **Zoom and Pan**: Smooth navigation within medical images
- **Measurement Tools**: Distance, angle, and area measurements
- **Segmentation Overlay**: AI-generated tumor segmentation visualization
- **NIfTI Support**: Direct loading of medical imaging formats

### AI Analysis Integration

- **Real-time Processing**: Live status updates during analysis
- **Multi-modal Support**: FLAIR, T1, T1CE, T2 MRI sequences
- **Results Visualization**: Interactive display of detection, classification, and segmentation results
- **Confidence Scores**: Transparent AI model confidence metrics
- **Volume Analysis**: 3D tumor volume calculations and tracking

### User Interface

- **Comprehensive Dashboard**: Statistics, recent scans, and volume analysis charts
- **Patient Management**: Complete scan library with detailed metadata
- **Secure Authentication**: JWT-based user authentication and authorization
- **Responsive Design**: Mobile-first design with dark/light theme support
- **Accessibility**: WCAG 2.1 AA compliance with semantic HTML

## Getting Started

### Prerequisites

- Python 3.8+ (for backend)
- Node.js 18+ (for frontend)
- Git

### Backend Setup

1. Navigate to the Backend directory
2. Install dependencies: `pip install -r requirements.txt`
3. Start the server: `uvicorn main:app --reload`

### Frontend Setup

1. Navigate to the Frontend directory
2. Install dependencies: `npm install`
3. Start the development server: `npm run dev`

## API Endpoints

### Authentication

| Method | Endpoint             | Description                     |
| ------ | -------------------- | ------------------------------- |
| POST   | `/api/auth/register` | User registration               |
| POST   | `/api/auth/login`    | User login and token generation |
| GET    | `/api/auth/me`       | Get current user information    |
| PUT    | `/api/auth/me`       | Update current user profile     |

### Scan Management

| Method | Endpoint               | Description                           |
| ------ | ---------------------- | ------------------------------------- |
| GET    | `/api/scans`           | List all scans for authenticated user |
| GET    | `/api/scans/{scan_id}` | Get specific scan details             |
| POST   | `/api/process-mri`     | Upload and process MRI files          |
| PUT    | `/api/scans/{scan_id}` | Update scan metadata                  |
| DELETE | `/api/scans/{scan_id}` | Delete scan and associated files      |

### Image Retrieval

| Method | Endpoint                                        | Description                             |
| ------ | ----------------------------------------------- | --------------------------------------- |
| GET    | `/api/scans/{scan_id}/slice/{slice_idx}`        | Get MRI slice as JPEG image             |
| GET    | `/api/scans/{scan_id}/segmentation/{slice_idx}` | Get segmentation slice as PNG           |
| GET    | `/api/scans/{scan_id}/download/{key}`           | Download original files or segmentation |

## Processing Pipeline

1. **File Upload**: Users upload multiple MRI modalities (FLAIR, T1, T1CE, T2)
2. **File Organization**: Files are organized by scan ID and modality
3. **Detection**: Binary tumor detection using ResNet50
4. **Classification**: Multi-class tumor type classification (if tumor detected)
5. **Segmentation**: 3D volumetric segmentation (if all 4 modalities available)
6. **Results Storage**: Analysis results stored in database with file references

## Supported MRI Modalities

- **FLAIR**: Fluid-attenuated inversion recovery
- **T1**: T1-weighted imaging
- **T1CE**: T1-weighted with contrast enhancement
- **T2**: T2-weighted imaging

## Analysis Results

Each processed scan returns comprehensive results including:

```json
{
  "detected": true/false,
  "classification": "glioma/meningioma/pituitary/notumor",
  "confidence": 0.95,
  "tumorVolume": 123.45,
  "wtVolume": 156.78,
  "tcVolume": 89.12,
  "etVolume": 34.56
}
```

## Technology Stack

### Backend

- **FastAPI**: High-performance async web framework
- **SQLModel**: Modern ORM with Pydantic validation
- **TensorFlow/Keras**: Deep learning model deployment
- **PyTorch**: 3D segmentation model support
- **JWT**: Secure authentication

### Frontend

- **React 18.3.1**: Modern React with TypeScript
- **Vite**: Fast development server and optimized builds
- **Tailwind CSS**: Utility-first CSS framework
- **shadcn/ui**: High-quality component library
- **Cornerstone.js**: Professional medical image viewer
- **VTK.js**: Advanced 3D visualization

## Security Features

- **JWT Authentication**: Secure token-based authentication
- **Password Hashing**: bcrypt for secure password storage
- **CORS Support**: Configurable cross-origin resource sharing
- **File Access Control**: User-specific file access restrictions
- **Input Validation**: Comprehensive request validation

## Model Details

### Detection Model (ResNet50)

- **Architecture**: ResNet50 with custom classification head
- **Training Data**: BR35H, NAV, and MOS datasets
- **Input**: 224×224 RGB images
- **Output**: Binary tumor presence probability

### Classification Model (ResNet50)

- **Architecture**: ResNet50 with frozen base layers
- **Training Data**: 11 merged brain MRI datasets (46,920 images)
- **Classes**: Glioma, Meningioma, Pituitary, No Tumor
- **Input**: 224×224 RGB images
- **Output**: 4-class probability distribution

### Segmentation Model (3D nnU-Net)

- **Architecture**: 3D U-Net with deep supervision
- **Training Data**: BraTS2020 dataset (369 cases)
- **Input**: 128×128×128 patches from 4 modalities
- **Output**: 3-channel segmentation (WT, TC, ET)
