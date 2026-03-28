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

#### Authentication Endpoints

- `POST /api/auth/register`
  - **Purpose**: Create new user account
  - **Request Body**: UserCreate model (username, email, password, fullName, title, department, institution)
  - **Response**: JWT access token + user information
  - **Authentication**: None required

- `POST /api/auth/login`
  - **Purpose**: Authenticate existing user
  - **Request Body**: OAuth2PasswordRequestForm (username, password)
  - **Response**: JWT access token + user information
  - **Authentication**: None required

- `GET /api/auth/me`
  - **Purpose**: Get current authenticated user profile
  - **Headers**: Bearer token required
  - **Response**: Complete user object
  - **Authentication**: JWT token required

- `PUT /api/auth/me`
  - **Purpose**: Update current user profile information
  - **Headers**: Bearer token required
  - **Request Body**: UserUpdate model (optional fields)
  - **Response**: Updated user object
  - **Authentication**: JWT token required

#### Scan Management Endpoints

- `GET /api/scans`
  - **Purpose**: List all scans for current user
  - **Headers**: Bearer token required
  - **Response**: Array of Scan objects ordered by creation date (newest first)
  - **Authentication**: JWT token required

- `GET /api/scans/{scan_id}`
  - **Purpose**: Get specific scan details
  - **Headers**: Bearer token required
  - **Parameters**: scan_id (string)
  - **Response**: Complete Scan object with results
  - **Authentication**: JWT token required

- `POST /api/process-mri`
  - **Purpose**: Upload and process MRI files for analysis
  - **Headers**: Bearer token required
  - **Request**: Multi-part form data
    - files: Array of MRI files (T1, T1CE, T2, FLAIR)
    - patientName: Patient display name
    - patientId: Patient identifier (optional, auto-generated if not provided)
  - **Response**: Complete Scan object with analysis results
  - **Processing**: Full AI pipeline (detection + classification + segmentation)
  - **Authentication**: JWT token required

- `PUT /api/scans/{scan_id}`
  - **Purpose**: Update scan metadata
  - **Headers**: Bearer token required
  - **Parameters**: scan_id (string)
  - **Request Body**: ScanUpdate model (patientName, patientId)
  - **Response**: Updated Scan object
  - **Authentication**: JWT token required

- `DELETE /api/scans/{scan_id}`
  - **Purpose**: Delete scan and associated files
  - **Headers**: Bearer token required
  - **Parameters**: scan_id (string)
  - **Response**: Success message
  - **Side Effects**: Removes scan directory and database record
  - **Authentication**: JWT token required

#### Data Access Endpoints

- `GET /api/scans/{scan_id}/slice/{slice_idx}`
  - **Purpose**: Get specific MRI slice as JPEG image
  - **Headers**: Bearer token required
  - **Parameters**:
    - scan_id: Scan identifier
    - slice_idx: Slice index (integer)
    - modality: MRI modality (t1, t1ce, t2, flair) - optional
  - **Response**: JPEG image with slice data
  - **Headers**: X-Total-Slices (total number of slices)
  - **Authentication**: JWT token required

- `GET /api/scans/{scan_id}/segmentation/{slice_idx}`
  - **Purpose**: Get segmentation mask overlay for specific slice
  - **Headers**: Bearer token required
  - **Parameters**: scan_id, slice_idx
  - **Response**: PNG image with segmentation overlay
  - **Authentication**: JWT token required

- `GET /api/scans/{scan_id}/download/{key}`
  - **Purpose**: Download original files or segmentation results
  - **Headers**: Bearer token required
  - **Parameters**:
    - scan_id: Scan identifier
    - key: File identifier ("segmentation", "t1", "t1ce", "t2", "flair", or original filename)
  - **Response**: File download with appropriate MIME type
  - **Authentication**: JWT token required

#### Authentication & Security

- **Token Type**: JWT Bearer tokens
- **Token Expiry**: 30 minutes (configurable via ACCESS_TOKEN_EXPIRE_MINUTES)
- **Password Security**: bcrypt hashing with salt
- **CORS**: Enabled for all origins, methods, and headers
- **Rate Limiting**: Not implemented (add for production)

#### Error Handling

- **Standard HTTP Status Codes**:
  - 200: Success
  - 201: Created
  - 400: Bad Request
  - 401: Unauthorized
  - 403: Forbidden
  - 404: Not Found
  - 422: Validation Error
  - 500: Internal Server Error

- **Error Response Format**:
  ```json
  {
  	"detail": "Error description message"
  }
  ```

### Model File Requirements

The system expects pre-trained models at these locations:

- `Detection/ResNet50-Binary-Detection.keras`
- `Classification/Brain-Tumour-Classification-ResNet50.keras`
- `Segmentation/BraTS2020_nnU_Net_Segmentation.pth`
- `Segmentation/BraTS2020_UNETR_Segmentation.pth` # New model added

### Database Schema

#### Database Design

The NeuroScan AI system uses SQLite for development with SQLModel ORM, designed for easy upgrade to PostgreSQL in production.

#### Tables Structure

**Users Table**:

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
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

**Scans Table**:

```sql
CREATE TABLE scans (
    id VARCHAR PRIMARY KEY,
    patientId VARCHAR NOT NULL,
    patientName VARCHAR NOT NULL,
    scanDate VARCHAR NOT NULL,
    modalities JSON,
    filePaths JSON,
    status VARCHAR DEFAULT 'queued',
    progress INTEGER DEFAULT 0,
    pipelineStep VARCHAR DEFAULT 'queued',
    results JSON,
    createdAt TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    userId INTEGER,
    FOREIGN KEY (userId) REFERENCES users(id)
);
```

#### Database Relationships

**One-to-Many Relationship**:

- Each user can have multiple scans
- Each scan belongs to exactly one user
- Cascade delete: Deleting a user removes associated scans

#### Data Types

**JSON Fields**:

- `modalities`: Array of strings representing MRI sequences
- `filePaths`: Object mapping modality names to relative file paths
- `results`: Nested object containing:
  - `detected`: Boolean - tumor presence
  - `classification`: String - tumor type (glioma, meningioma, pituitary, notumor)
  - `confidence`: Float - classification confidence score
  - `tumorVolume`: Float - total tumor volume in mm³
  - `wtVolume`: Float - whole tumor volume
  - `tcVolume`: Float - tumor core volume
  - `etVolume`: Float - enhancing tumor volume

#### Database Configuration

**Development Setup**:

- **Database**: SQLite file (`neuroscan.db`)
- **Connection**: Thread-safe with `check_same_thread=False`
- **ORM**: SQLModel for type-safe database operations
- **Sessions**: Context-managed with automatic cleanup

**Production Considerations**:

- **Upgrade Path**: Easy migration to PostgreSQL
- **Connection Pooling**: Configure for high-traffic deployment
- **Backup Strategy**: Regular database backups
- **Indexing**: Add indexes for performance optimization
- **Security**: Database connection encryption and access controls

---

## Python Files Explanation

### main.py - FastAPI Web Server

**Purpose**: Main application entry point and REST API server

**Key Components**:

- **Authentication System**: JWT-based with bcrypt password hashing
- **File Upload Handling**: Multi-file MRI upload support
- **API Endpoints**: Complete RESTful interface for all operations
- **Security**: CORS middleware for cross-origin requests
- **Database Integration**: Session management for user data persistence

**Core Functions**:

- `verify_password()`: Secure password verification using bcrypt
- `get_password_hash()`: Generate secure password hashes
- `create_access_token()`: JWT token generation for authentication
- `get_current_user()`: Dependency injection for protected routes

**API Endpoints Structure**:

- Authentication: `/api/auth/*` (register, login, profile management)
- Scan Management: `/api/scans/*` (CRUD operations)
- Data Access: `/api/scans/{id}/slice/*`, `/api/scans/{id}/download/*`

### database.py - Database Configuration

**Purpose**: Database setup and session management

**Key Components**:

- **SQLite Database**: File-based database for development
- **SQLModel Integration**: Type-safe ORM operations
- **Session Management**: Context-managed database connections

**Core Functions**:

- `create_db_and_tables()`: Initialize database schema
- `get_session()`: Generator for database sessions with automatic cleanup

**Database Configuration**:

- Database file: `neuroscan.db` in project directory
- Thread safety: `check_same_thread=False` for concurrent access
- Connection pooling: Efficient session management

### models.py - Data Models Definition

**Purpose**: SQLModel data models for database entities

**Key Models**:

**User Model**:

- Fields: id, username, email, hashed_password, fullName, title, department, institution, createdAt
- Constraints: Unique username and email
- Purpose: User authentication and profile management

**Scan Model**:

- Fields: id, patientId, patientName, scanDate, modalities, filePaths, status, progress, pipelineStep, results, createdAt, userId
- JSON Fields: modalities (list), filePaths (dict), results (dict)
- Purpose: MRI scan metadata and results storage

**Helper Models**:

- `UserCreate`: Registration form model
- `UserLogin`: Login form model
- `ScanCreate`: Scan creation form model
- `ScanUpdate`: Scan modification form model

### processor.py - AI Model Inference Engine

**Purpose**: Core AI processing engine for medical image analysis

**Key Components**:

**Model Architecture Definitions**:

- `ConvBlock`: 3D convolutional block with InstanceNorm and LeakyReLU
- `nnUNet`: Custom 3D UNet implementation for brain tumour segmentation

**BrainProcessor Class**:

- **Device Management**: Automatic GPU/CPU detection
- **Model Loading**: Dynamic loading of pre-trained models
- **Preprocessing Pipeline**: NIfTI to 2D conversion, brain cropping, normalisation

**Core Functions**:

- `_nib_load()`: Safe NIfTI file loading with error handling
- `_nib_save()`: Safe NIfTI file saving
- `nifti_to_2d()`: Convert 3D medical images to 2D for CNN processing
- `crop_brain()`: Extract brain region from MRI images
- `preprocess_for_tf()`: TensorFlow-compatible preprocessing

**AI Model Capabilities**:

- **Binary Detection**: Tumour presence/absence classification
- **Multi-class Classification**: Tumour type identification (glioma, meningioma, pituitary)
- **3D Segmentation**: Volumetric tumour boundary delineation
- **Multi-modal Support**: T1, T1CE, T2, FLAIR MRI sequences

---

## System Integration

**Data Flow**:

1. **Upload** → `main.py` receives MRI files
2. **Processing** → `processor.py` analyzes images using AI models
3. **Storage** → `database.py` persists results via SQLModel
4. **Retrieval** → `models.py` defines data structures for API responses

**Security Architecture**:

- **Authentication**: JWT tokens with bcrypt password hashing
- **Authorization**: Role-based access control per user
- **Data Privacy**: De-identified patient data storage

**Scalability Features**:

- **Async Processing**: FastAPI async support for concurrent requests
- **Database Flexibility**: SQLite for development, easily upgradeable to PostgreSQL
- **Model Management**: Dynamic model loading with error handling
- **File Management**: Organized upload directory structure with automatic cleanup
