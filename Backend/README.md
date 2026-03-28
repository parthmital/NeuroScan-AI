# NeuroScan AI Backend: Comprehensive Brain Tumour Analysis System

## Overview

NeuroScan AI Backend is a state-of-the-art medical imaging analysis system that provides automated brain tumour detection, classification, and segmentation from MRI scans. This system combines cutting-edge deep learning architectures with a robust web API to deliver clinical-grade analysis capabilities.

### Core Capabilities

- **Binary Tumour Detection**: Determines presence or absence of brain tumours with >99% accuracy
- **Multi-class Classification**: Identifies tumour types (glioma, meningioma, pituitary) with 98.58% accuracy
- **3D Tumour Segmentation**: Generates precise tumour boundaries and volume calculations
- **RESTful API**: FastAPI-based service for seamless integration
- **User Management**: Secure authentication and scan history tracking
- **DICOM Support**: Handles standard medical imaging formats

---

## System Architecture

### High-Level Design

The system follows a three-tier architecture:

- Frontend UI (React) communicates with FastAPI Server
- FastAPI Server (Backend) processes requests and manages AI Models
- AI Models (PyTorch) perform medical image analysis
- SQLite Database stores user data and scan results

### Core Components

#### 1. **Web Server** (`main.py`)

- **Framework**: FastAPI with async support
- **Authentication**: JWT-based with bcrypt password hashing
- **File Handling**: Multi-upload support for MRI modalities
- **API Endpoints**: RESTful design with OpenAPI documentation
- **CORS Support**: Cross-origin resource sharing enabled

#### 2. **AI Processing Engine** (`processor.py`)

- **Detection Models**: Binary tumour presence detection
- **Classification Models**: Multi-class tumour type identification
- **Segmentation Models**: 3D tumour boundary delineation
- **Preprocessing**: NIfTI to 2D conversion, brain cropping, normalisation
- **Model Loading**: Dynamic model loading with error handling

#### 3. **Database Layer** (`database.py`, `models.py`)

- **ORM**: SQLModel for type-safe database operations
- **Database**: SQLite for development (easily upgradeable to PostgreSQL)
- **Models**: User authentication, scan metadata, results storage
- **Sessions**: Context-managed database connections

---

## AI Model Performance

### Binary Detection Models

| Architecture       | Accuracy   | AUC-ROC    | Parameters | Training Time |
| ------------------ | ---------- | ---------- | ---------- | ------------- |
| **InceptionV3**    | **99.39%** | **1.0000** | 23.9M      | 20 min        |
| **CoaTNet-0**      | **99.59%** | **1.0000** | ~10M       | 35 min        |
| **EfficientNetB0** | **99.59%** | **0.9999** | 5.3M       | 15 min        |
| **DenseNet121**    | 97.95%     | 0.9999     | 8.0M       | 25 min        |
| **ResNet50**       | 99.39%     | 0.9999     | 25.6M      | 30 min        |

### Multi-class Classification Models

| Architecture       | Accuracy   | AUC-ROC    | Parameters | Key Strength           |
| ------------------ | ---------- | ---------- | ---------- | ---------------------- |
| **InceptionV3**    | **98.58%** | **0.9995** | 27.2M      | Multi-scale processing |
| **Swin-Base**      | **98.58%** | 0.9993     | 86.7M      | Hierarchical attention |
| **EfficientNetB0** | 97.59%     | 0.9987     | 5.3M       | Parameter efficiency   |
| **CoaTNet-0**      | 97.59%     | 0.9990     | ~10M       | Hybrid CNN-Transformer |
| **ResNet-50**      | 96.60%     | 0.9990     | 25.6M      | Stable training        |
| **ViT-B/16**       | 89.52%     | 0.9736     | 86.4M      | Pure transformer       |

### 3D Segmentation Models

| Architecture   | Mean Dice  | WT Dice | TC Dice | ET Dice | Parameters |
| -------------- | ---------- | ------- | ------- | ------- | ---------- |
| **UNet**       | **0.7999** | 0.8647  | 0.7844  | 0.7505  | 5.6M       |
| **nnU-Net**    | 0.7945     | 0.8569  | 0.7813  | 0.7452  | 16.5M      |
| **VNet**       | 0.7655     | 0.8541  | 0.7108  | 0.7316  | 1.4M       |
| **Swin-UNETR** | 0.6568     | 0.8128  | 0.6261  | 0.5314  | 10.7M      |
| **TransBTS**   | 0.5109     | 0.6383  | 0.4618  | 0.4325  | 35.8M      |
| **UNETR**      | 0.6234     | 0.7128  | 0.6261  | 0.5314  | 137.6M     |

---

## Technical Specifications

### API Endpoints

#### Authentication

- `POST /api/auth/register` - User registration
- `POST /api/auth/login` - User login
- `GET /api/auth/me` - Get current user info
- `PUT /api/auth/me` - Update user profile

#### Scan Management

- `GET /api/scans` - List user's scans
- `GET /api/scans/{scan_id}` - Get scan details
- `POST /api/process-mri` - Upload and process MRI files
- `PUT /api/scans/{scan_id}` - Update scan metadata
- `DELETE /api/scans/{scan_id}` - Delete scan

#### Data Access

- `GET /api/scans/{scan_id}/slice/{slice_idx}` - Get MRI slice as image
- `GET /api/scans/{scan_id}/segmentation/{slice_idx}` - Get segmentation overlay
- `GET /api/scans/{scan_id}/download/{key}` - Download original files

### Model File Requirements

The system expects pre-trained models at these locations:

- `Detection/ResNet50-Binary-Detection.keras`
- `Classification/Brain-Tumour-Classification-ResNet50.keras`
- `Segmentation/BraTS2020_nnU_Net_Segmentation.pth`

### Database Schema

```sql
CREATE TABLE users (
    id INTEGER PRIMARY KEY,
    username VARCHAR UNIQUE NOT NULL,
    email VARCHAR UNIQUE NOT NULL,
    hashed_password VARCHAR NOT NULL,
    fullName VARCHAR NOT NULL,
    title VARCHAR,
    department VARCHAR,
    institution VARCHAR,
    createdAt TIMESTAMP
);

CREATE TABLE scans (
    id VARCHAR PRIMARY KEY,
    patientId VARCHAR NOT NULL,
    patientName VARCHAR NOT NULL,
    scanDate VARCHAR NOT NULL,
    modalities JSON,  -- ["T1", "T1CE", "T2", "FLAIR"]
    filePaths JSON,   -- {"t1": "path", "t1ce": "path", ...}
    status VARCHAR,    -- "queued", "processing", "completed", "failed"
    progress INTEGER,  -- 0-100
    pipelineStep VARCHAR,
    results JSON,     -- Detection, classification, segmentation results
    createdAt TIMESTAMP,
    userId INTEGER,
    FOREIGN KEY (userId) REFERENCES users(id)
);
```
