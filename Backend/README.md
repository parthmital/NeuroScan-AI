# Neuroscan AI Backend

A comprehensive FastAPI-based backend system for brain tumor analysis using advanced deep learning models. This system provides automated detection, classification, and segmentation of brain tumors from MRI scans with a complete user management and data processing pipeline.

## Overview

A production-ready medical imaging analysis platform that combines three specialized deep learning models:

- **Tumor Detection**: Binary classification to identify presence/absence of tumors
- **Tumor Classification**: Multi-class classification (glioma, meningioma, pituitary, no tumor)
- **Tumor Segmentation**: 3D volumetric segmentation for precise tumor delineation

## Architecture

### Core Components

- **FastAPI Web Framework**: High-performance async API server
- **SQLModel Database**: SQLite-based data persistence with async support
- **JWT Authentication**: Secure token-based user authentication
- **File Management**: Structured upload and processing of MRI files
- **AI Pipeline**: Integrated deep learning models for comprehensive analysis

### AI Models

| Model              | Architecture                 | Task                          | Performance                          |
| ------------------ | ---------------------------- | ----------------------------- | ------------------------------------ |
| **Detection**      | ResNet50 (Transfer Learning) | Binary Tumor Detection        | 80.07% Accuracy, 85.99% AUC          |
| **Classification** | ResNet50 (Multi-class)       | Tumor Type Classification     | 87.41% Accuracy                      |
| **Segmentation**   | 3D nnU-Net                   | Volumetric Tumor Segmentation | WT: 84.7%, TC: 75.3%, ET: 76.1% Dice |

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

### MRI Upload and Processing

1. **File Upload**: Users upload multiple MRI modalities (FLAIR, T1, T1CE, T2)
2. **File Organization**: Files are organized by scan ID and modality
3. **Detection**: Binary tumor detection using ResNet50
4. **Classification**: Multi-class tumor type classification (if tumor detected)
5. **Segmentation**: 3D volumetric segmentation (if all 4 modalities available)
6. **Results Storage**: Analysis results stored in database with file references

### Supported MRI Modalities

- **FLAIR**: Fluid-attenuated inversion recovery
- **T1**: T1-weighted imaging
- **T1CE**: T1-weighted with contrast enhancement
- **T2**: T2-weighted imaging

### Analysis Results

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

## Database Schema

### User Model

- `id`: Primary key
- `username`: Unique username
- `email`: Unique email address
- `hashed_password`: bcrypt hashed password
- `fullName`: User's full name
- `title`: Professional title (optional)
- `department`: Department/organization (optional)
- `institution`: Institution (optional)
- `createdAt`: Account creation timestamp

### Scan Model

- `id`: Unique scan identifier
- `patientId`: Patient identifier
- `patientName`: Patient name
- `scanDate`: Scan date
- `modalities`: List of available MRI modalities
- `filePaths`: Dictionary of modality to file paths
- `status`: Processing status
- `progress`: Processing progress percentage
- `pipelineStep`: Current processing step
- `results`: AI analysis results
- `userId`: Foreign key to user
- `createdAt`: Scan creation timestamp

## Configuration

### Environment Variables

The application uses the following configuration constants (can be moved to environment variables for production):

- `SECRET_KEY`: JWT signing key (change in production)
- `ACCESS_TOKEN_EXPIRE_MINUTES`: Token expiration time (default: 30 minutes)
- `UPLOAD_DIR`: File upload directory

### Model Loading

Models are automatically loaded during application startup from the following paths:

- Detection: `Detection/ResNet50-Binary-Detection.keras`
- Classification: `Classification/Brain-Tumor-Classification-ResNet50.keras`
- Segmentation: `Segmentation/BraTS2020_nnU_Net_Segmentation.pth`

## Model Details

### Detection Model (ResNet50)

- **Architecture**: ResNet50 with custom classification head
- **Training Data**: BR35H, NAV, and MOS datasets
- **Input**: 224×224 RGB images
- **Output**: Binary tumor presence probability
- **Performance**: 80.07% accuracy, 85.99% AUC

### Classification Model (ResNet50)

- **Architecture**: ResNet50 with frozen base layers
- **Training Data**: 11 merged brain MRI datasets (46,920 images)
- **Classes**: Glioma, Meningioma, Pituitary, No Tumor
- **Input**: 224×224 RGB images
- **Output**: 4-class probability distribution
- **Performance**: 87.41% accuracy

### Segmentation Model (3D nnU-Net)

- **Architecture**: 3D U-Net with deep supervision
- **Training Data**: BraTS2020 dataset (369 cases)
- **Input**: 128×128×128 patches from 4 modalities
- **Output**: 3-channel segmentation (WT, TC, ET)
- **Performance**: WT: 84.7%, TC: 75.3%, ET: 76.1% Dice

## Security Features

- **JWT Authentication**: Secure token-based authentication
- **Password Hashing**: bcrypt for secure password storage
- **CORS Support**: Configurable cross-origin resource sharing
- **File Access Control**: User-specific file access restrictions
- **Input Validation**: Comprehensive request validation using Pydantic
