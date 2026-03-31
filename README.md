# NeuroScan AI

**AI-Powered Brain Tumour Detection, Classification and Segmentation Platform**

Deep Learning Meets Medical Imaging — From Research to Clinical Deployment

---

## Table of Contents

- [NeuroScan AI](#neuroscan-ai)
	- [Table of Contents](#table-of-contents)
	- [What is NeuroScan AI?](#what-is-neuroscan-ai)
		- [The Problem We Solve](#the-problem-we-solve)
		- [Our Solution](#our-solution)
	- [Key Features](#key-features)
		- [For Healthcare Professionals](#for-healthcare-professionals)
		- [For Technical Users](#for-technical-users)
	- [Technology Stack](#technology-stack)
		- [Frontend Stack](#frontend-stack)
		- [Backend Stack](#backend-stack)
		- [AI/ML Stack](#aiml-stack)
	- [Quick Start Guide](#quick-start-guide)
		- [One-Command Startup (Windows)](#one-command-startup-windows)
		- [Manual Setup (Cross-Platform)](#manual-setup-cross-platform)
	- [Frontend Deep Dive](#frontend-deep-dive)
		- [Frontend Core Technologies](#frontend-core-technologies)
			- [React 18 with TypeScript](#react-18-with-typescript)
			- [Vite Build Tool](#vite-build-tool)
			- [Tailwind CSS and shadcn/ui](#tailwind-css-and-shadcnui)
		- [Component Architecture](#component-architecture)
			- [App.tsx — The Root Component](#apptsx--the-root-component)
			- [ProtectedRoute — Authentication Guard](#protectedroute--authentication-guard)
		- [State Management](#state-management)
			- [React Query (TanStack Query)](#react-query-tanstack-query)
			- [Auth Context](#auth-context)
		- [API Integration](#api-integration)
			- [lib/api.ts — HTTP Client](#libapits--http-client)
	- [Backend Deep Dive](#backend-deep-dive)
		- [Backend Core Technologies](#backend-core-technologies)
			- [FastAPI — Modern Python Web Framework](#fastapi--modern-python-web-framework)
			- [SQLModel — ORM and Pydantic](#sqlmodel--orm-and-pydantic)
			- [SQLite — Serverless Database](#sqlite--serverless-database)
		- [API Endpoints](#api-endpoints)
		- [Authentication Flow](#authentication-flow)
	- [AI/ML Pipeline and Comparative Analysis](#aiml-pipeline-and-comparative-analysis)
		- [Three-Stage Architecture](#three-stage-architecture)
		- [Stage 1: Tumour Detection — Comparative Analysis](#stage-1-tumour-detection--comparative-analysis)
			- [Models Evaluated](#models-evaluated)
			- [Detection Performance Results](#detection-performance-results)
			- [Key Findings from Detection Comparative Analysis](#key-findings-from-detection-comparative-analysis)
		- [Stage 2: Tumour Classification — Comparative Analysis](#stage-2-tumour-classification--comparative-analysis)
			- [Models Evaluated](#models-evaluated-1)
			- [Dataset Composition](#dataset-composition)
			- [Classification Performance Results](#classification-performance-results)
			- [Key Findings from Classification Comparative Analysis](#key-findings-from-classification-comparative-analysis)
		- [Stage 3: Tumour Segmentation — Comparative Analysis](#stage-3-tumour-segmentation--comparative-analysis)
			- [Models Evaluated](#models-evaluated-2)
			- [Dataset Composition](#dataset-composition-1)
			- [Segmentation Performance Results (Dice Scores)](#segmentation-performance-results-dice-scores)
			- [Computational Efficiency Analysis](#computational-efficiency-analysis)
			- [Key Findings from Segmentation Comparative Analysis](#key-findings-from-segmentation-comparative-analysis)
		- [Model Architectures](#model-architectures)
			- [EfficientNet-B0 (Detection and Classification)](#efficientnet-b0-detection-and-classification)
			- [VNet3D (Segmentation)](#vnet3d-segmentation)
		- [Training Methodology](#training-methodology)
			- [Standardised Training Configuration](#standardised-training-configuration)
			- [Multi-GPU Strategy](#multi-gpu-strategy)
			- [Data Sources](#data-sources)
			- [Preprocessing Pipeline](#preprocessing-pipeline)
	- [Installation and Setup](#installation-and-setup)
		- [Prerequisites](#prerequisites)
		- [Step-by-Step Setup](#step-by-step-setup)
			- [Step 1: Clone Repository](#step-1-clone-repository)
			- [Step 2: Create Python Virtual Environment](#step-2-create-python-virtual-environment)
			- [Step 3: Install Python Dependencies](#step-3-install-python-dependencies)
			- [Step 4: Install Node.js Dependencies](#step-4-install-nodejs-dependencies)
			- [Step 5: Download Pre-trained Models](#step-5-download-pre-trained-models)
		- [Running the Application](#running-the-application)
			- [Option A: One-Command Startup (Windows)](#option-a-one-command-startup-windows)
			- [Option B: Manual Terminal Sessions](#option-b-manual-terminal-sessions)
			- [Verify Installation](#verify-installation)
	- [Usage Guide](#usage-guide)
		- [For Medical Professionals](#for-medical-professionals)
			- [Uploading MRI Scans](#uploading-mri-scans)
			- [Interpreting Results](#interpreting-results)
			- [Generating Reports](#generating-reports)
		- [For Developers](#for-developers)
			- [API Usage Examples](#api-usage-examples)
			- [Extending the Pipeline](#extending-the-pipeline)
		- [For Researchers](#for-researchers)
			- [Reproducing Training Results](#reproducing-training-results)
	- [Detailed Research Documentation](#detailed-research-documentation)
		- [1. Binary Detection Research](#1-binary-detection-research)
		- [2. Classification Research](#2-classification-research)
		- [3. Segmentation Research](#3-segmentation-research)
	- [Contributing](#contributing)
		- [Development Workflow](#development-workflow)

---

## What is NeuroScan AI?

**NeuroScan AI** is a comprehensive, full-stack medical imaging platform that leverages cutting-edge **deep learning** and **computer vision** techniques to automate brain tumour analysis from **MRI scans**. Built with a modern tech stack and research-grade AI models, it bridges the gap between academic research and real-world clinical deployment.

### The Problem We Solve

Brain tumour diagnosis traditionally requires:

- **Hours of manual analysis** by specialised radiologists
- **Subjective interpretation** with inter-observer variability
- **Limited accessibility** in underserved regions
- **Delayed treatment planning** due to diagnostic bottlenecks

### Our Solution

NeuroScan AI provides **automated, AI-powered analysis** that delivers:

- **Detection**: Identifies presence of tumours with greater than 98% accuracy
- **Classification**: Categorises tumours into Glioma, Meningioma, or Pituitary types
- **Segmentation**: Generates pixel-precise 3D tumour boundaries for treatment planning
- **Volumetric Analysis**: Calculates tumour volume metrics for tracking progression

---

## Key Features

### For Healthcare Professionals

| Feature                   | Description                                                         | Benefit                   |
| ------------------------- | ------------------------------------------------------------------- | ------------------------- |
| **Drag-and-Drop Upload**  | Upload multiple MRI modalities (T1, T1CE, T2, FLAIR) simultaneously | Streamlined workflow      |
| **Real-Time Processing**  | AI analysis completes in under 30 seconds                           | Immediate results         |
| **Interactive 3D Viewer** | VTK.js-powered visualisation with slice navigation                  | Detailed examination      |
| **Segmentation Overlay**  | Colour-coded tumour sub-regions (WT, TC, ET)                        | Precise localisation      |
| **PDF Report Generation** | One-click export of diagnostic findings                             | Documentation and sharing |
| **Patient Management**    | Secure, per-user scan libraries with privacy                        | Data security             |

### For Technical Users

| Feature                | Technology                 | Purpose                              |
| ---------------------- | -------------------------- | ------------------------------------ |
| **JWT Authentication** | OAuth2 and bcrypt          | Secure, stateless auth               |
| **RESTful API**        | FastAPI and Pydantic       | Type-safe, auto-documented endpoints |
| **Database**           | SQLite and SQLModel        | ACID-compliant, ORM-native           |
| **ML Pipeline**        | PyTorch and TorchVision    | GPU-accelerated inference            |
| **Image Processing**   | OpenCV and NiBabel         | Medical-grade NIfTI handling         |
| **Modern Frontend**    | React 18, TypeScript, Vite | Fast, type-safe UI                   |

---

## Technology Stack

### Frontend Stack

| Category       | Technology   | Version | Purpose                       |
| -------------- | ------------ | ------- | ----------------------------- |
| **Framework**  | React        | 18.3.1  | UI component library          |
| **Language**   | TypeScript   | 5.6.0   | Type-safe development         |
| **Build Tool** | Vite         | 7.3.1   | Fast development and bundling |
| **Styling**    | Tailwind CSS | 3.4.17  | Utility-first CSS             |
| **Components** | shadcn/ui    | Latest  | Accessible UI primitives      |
| **Icons**      | Lucide React | 0.462.0 | Modern icon library           |
| **State**      | React Query  | 5.83.0  | Server state management       |
| **Routing**    | React Router | 6.30.1  | Client-side navigation        |
| **3D Viz**     | VTK.js       | 35.2.0  | Medical image visualisation   |
| **Charts**     | Recharts     | 2.15.4  | Data visualisation            |
| **Testing**    | Vitest       | 3.2.4   | Unit testing framework        |

### Backend Stack

| Category             | Technology  | Version  | Purpose                      |
| -------------------- | ----------- | -------- | ---------------------------- |
| **Framework**        | FastAPI     | Latest   | High-performance async API   |
| **Server**           | Uvicorn     | Standard | ASGI server                  |
| **ML Framework**     | PyTorch     | 2.0+     | Deep learning                |
| **Computer Vision**  | OpenCV      | Latest   | Image processing             |
| **Medical Imaging**  | NiBabel     | Latest   | NIfTI file I/O               |
| **Database**         | SQLModel    | Latest   | ORM and Pydantic integration |
| **Database Engine**  | SQLite      | 3.x      | Lightweight, serverless      |
| **Authentication**   | python-jose | Latest   | JWT token handling           |
| **Password Hashing** | bcrypt      | Latest   | Secure password storage      |
| **Validation**       | Pydantic    | Latest   | Data validation              |

### AI/ML Stack

| Component            | Architecture    | Framework        | Accuracy         |
| -------------------- | --------------- | ---------------- | ---------------- |
| **Tumour Detection** | EfficientNet-B0 | PyTorch          | Greater than 99% |
| **Classification**   | EfficientNet-B0 | PyTorch          | 97.59%           |
| **Segmentation**     | VNet 3D         | PyTorch (Custom) | WT/TC/ET volumes |

---

## Quick Start Guide

**Prerequisites**: Node.js 18+, Python 3.11+, PowerShell

### One-Command Startup (Windows)

```powershell
# Clone and navigate to the repository
git clone <repository-url>
cd "NeuroScan AI"

# Start everything (Frontend + Backend)
.\start.ps1
```

This script automatically:

1. Checks for Node.js and Python
2. Creates/activates Python virtual environment
3. Installs dependencies (npm + pip)
4. Starts Backend API (http://localhost:8000)
5. Starts Frontend Dev Server (http://localhost:8080)
6. Opens browser automatically

### Manual Setup (Cross-Platform)

```bash
# 1. Setup Backend
cd Backend
python -m venv ../.venv
source ../.venv/bin/activate  # Linux/Mac
# or: ..\.venv\Scripts\activate  # Windows
pip install -r requirements.txt
uvicorn main:app --reload --port 8000

# 2. Setup Frontend (new terminal)
cd Frontend
npm install
npm run dev

# 3. Open http://localhost:8080 in browser
```

---

## Frontend Deep Dive

### Frontend Core Technologies

#### React 18 with TypeScript

The frontend is built on **React 18** using **functional components** and **hooks** for state management. TypeScript provides compile-time type safety, catching errors before they reach production.

**Key Patterns:**

- **Functional Components**: Modern React with hooks (no class components)
- **Custom Hooks**: `useAuth`, `useConfig`, `use-toast` for reusable logic
- **Type Safety**: All props, API responses, and state are strictly typed

#### Vite Build Tool

**Vite** replaces webpack for instant Hot Module Replacement (HMR) and optimised production builds.

**Configuration highlights:**

```typescript
// vite.config.ts
export default defineConfig({
	plugins: [react()],
	server: { port: 8080 },
	resolve: {
		alias: {
			"@": path.resolve(__dirname, "./src"), // Clean imports: @/components/ui/button
		},
	},
});
```

#### Tailwind CSS and shadcn/ui

**Tailwind CSS** provides atomic utility classes for rapid styling without writing custom CSS.

**shadcn/ui** offers accessible, composable components built on Radix UI primitives:

- **Button**: Primary, secondary, ghost variants
- **Card**: Container with consistent styling
- **Dialog**: Modal overlays
- **Tabs**: Content organisation
- **Toast**: User notifications

### Component Architecture

#### App.tsx — The Root Component

```typescript
// src/App.tsx (simplified)
const App = () => {
  return (
    <QueryClientProvider client={queryClient}>
      <AuthProvider>
        <BrowserRouter>
          <Routes>
            <Route path="/login" element={<Login />} />
            <Route element={<ProtectedRoute><AppLayout /></ProtectedRoute>}>
              <Route path="/" element={<Dashboard />} />
              <Route path="/upload" element={<Upload />} />
              <Route path="/scan/:id" element={<ScanDetail />} />
            </Route>
          </Routes>
        </BrowserRouter>
      </AuthProvider>
    </QueryClientProvider>
  );
};
```

#### ProtectedRoute — Authentication Guard

Wraps all authenticated routes, redirects to `/login` if no valid JWT token exists.

### State Management

#### React Query (TanStack Query)

Manages server state with automatic caching, refetching, and optimistic updates.

```typescript
// Example: Fetching scans
const { data: scans, isLoading } = useQuery({
	queryKey: ["scans"],
	queryFn: fetchScans,
	staleTime: 30000, // Data fresh for 30 seconds
});
```

#### Auth Context

Global authentication state using React Context API:

- `isAuthenticated`: Boolean login status
- `user`: Current user object
- `login()/logout()`: Authentication methods
- Persists token in `localStorage`

### API Integration

#### lib/api.ts — HTTP Client

Centralised API functions using native `fetch`:

```typescript
const API_BASE_URL = "http://localhost:8000/api";

export const processMRI = async (files: File[], patientName?: string) => {
	const formData = new FormData();
	files.forEach((f) => formData.append("files", f));
	if (patientName) formData.append("patientName", patientName);

	const response = await fetch(`${API_BASE_URL}/process-mri`, {
		method: "POST",
		headers: { Authorization: `Bearer ${token}` },
		body: formData,
	});
	return handleResponse(response);
};
```

---

## Backend Deep Dive

### Backend Core Technologies

#### FastAPI — Modern Python Web Framework

**FastAPI** is a high-performance, async-capable web framework with automatic API documentation.

**Why FastAPI?**

- **Type Safety**: Native Pydantic integration for request/response validation
- **Async/Await**: Handles concurrent requests efficiently
- **Auto Documentation**: Interactive Swagger UI at `/docs`
- **Dependency Injection**: Clean, testable code structure

#### SQLModel — ORM and Pydantic

**SQLModel** (by the creator of FastAPI) combines SQLAlchemy ORM with Pydantic validation:

```python
class User(SQLModel, table=True):
  id: Optional[int] = Field(default=None, primary_key=True)
  username: str = Field(index=True, unique=True)
  email: str = Field(index=True, unique=True)
  hashed_password: str
  fullName: str
  createdAt: datetime = Field(default_factory=datetime.now)

class Scan(SQLModel, table=True):
  id: str = Field(primary_key=True)
  patientId: str
  patientName: str
  scanDate: str
  modalities: List[str] = Field(sa_column=Column(JSON))
  results: Optional[Dict[str, object]] = Field(default=None, sa_column=Column(JSON))
  userId: Optional[int] = Field(default=None, foreign_key="user.id")
```

#### SQLite — Serverless Database

**SQLite** chosen for:

- **Zero Configuration**: No separate server process
- **Single File**: Easy backup and portability
- **ACID Compliant**: Reliable transactions
- **Sufficient Scale**: Handles thousands of scans efficiently

### API Endpoints

| Method | Endpoint                             | Description            | Auth Required |
| ------ | ------------------------------------ | ---------------------- | ------------- |
| POST   | `/api/auth/register`                 | Create new account     | No            |
| POST   | `/api/auth/login`                    | Authenticate, get JWT  | No            |
| GET    | `/api/auth/me`                       | Get current user info  | Yes           |
| PUT    | `/api/auth/me`                       | Update profile         | Yes           |
| GET    | `/api/scans`                         | List user's scans      | Yes           |
| GET    | `/api/scans/{id}`                    | Get specific scan      | Yes           |
| POST   | `/api/process-mri`                   | Upload and analyse MRI | Yes           |
| GET    | `/api/scans/{id}/slice/{idx}`        | Get MRI slice image    | Yes           |
| GET    | `/api/scans/{id}/segmentation/{idx}` | Get segmentation mask  | Yes           |
| GET    | `/api/scans/{id}/download/{key}`     | Download NIfTI files   | Yes           |
| PUT    | `/api/scans/{id}`                    | Update scan metadata   | Yes           |
| DELETE | `/api/scans/{id}`                    | Delete scan            | Yes           |

### Authentication Flow

The authentication system uses JWT (JSON Web Tokens) with OAuth2PasswordBearer:

1. User submits username/password to `/api/auth/login`
2. Server verifies credentials using bcrypt-hashed passwords
3. Server issues JWT token valid for 30 minutes
4. Client stores token in localStorage
5. Subsequent requests include token in Authorization header
6. Server validates token and identifies user

---

## AI/ML Pipeline and Comparative Analysis

NeuroScan AI implements a **three-stage comparative analysis pipeline** for comprehensive brain tumour analysis. Each stage was developed through rigorous comparative evaluation of multiple state-of-the-art deep learning architectures under identical experimental conditions.

### Three-Stage Architecture

The AI pipeline consists of three sequential stages:

1. **Stage 1: Tumour Detection** — Binary classification (tumour present/absent)
2. **Stage 2: Tumour Classification** — Multi-class classification (Glioma/Meningioma/Pituitary)
3. **Stage 3: Tumour Segmentation** — 3D volumetric segmentation (WT/TC/ET regions)

Each stage employed **comparative evaluation** across multiple architectures to identify optimal models for clinical deployment.

### Stage 1: Tumour Detection — Comparative Analysis

The detection stage performs binary classification to determine whether a tumour is present in the MRI scan. We conducted a comprehensive comparative study of **9 state-of-the-art architectures** spanning classical CNNs, efficient CNNs, and transformer-based models.

#### Models Evaluated

| Architecture Family | Models                                         |
| ------------------- | ---------------------------------------------- |
| **Classical CNNs**  | ResNet50, VGG16, DenseNet121                   |
| **Efficient CNNs**  | EfficientNetB0, MobileNetV3-Large, InceptionV3 |
| **Transformers**    | ViT-B/16, Swin Transformer Base, CoaTNet-0     |

#### Detection Performance Results

| Model              | Test Accuracy | Test AUC   | Parameters | Training Time |
| ------------------ | ------------- | ---------- | ---------- | ------------- |
| **CoaTNet-0**      | **99.59%**    | **1.0000** | ~10M       | Moderate      |
| **EfficientNetB0** | **99.59%**    | **0.9999** | 5.3M       | Fastest       |
| **InceptionV3**    | **99.39%**    | **1.0000** | 23.9M      | Moderate      |
| **ResNet50**       | **99.39%**    | **0.9999** | 25.6M      | Fast          |
| **DenseNet121**    | **97.95%**    | **0.9999** | 8.0M       | Fast          |
| **Swin Base**      | 98.16%        | 0.9990     | 88M        | Slow          |
| **MobileNetV3**    | 91.39%        | 0.9982     | 5.4M       | Fast          |
| **ViT-B/16**       | 95.08%        | 0.9861     | 86M        | Slowest       |
| **VGG16**          | 83.20%        | 0.9247     | 138M       | Slow          |

#### Key Findings from Detection Comparative Analysis

- **Top Performers**: CoaTNet-0 and EfficientNetB0 achieved 99.59% accuracy with near-perfect AUC scores
- **Efficiency Champion**: EfficientNetB0 achieved 99.59% accuracy with only 5.3M parameters — ideal for resource-constrained deployment
- **Transformer Performance**: CoaTNet-0 (hybrid CNN-transformer) outperformed pure transformers (ViT-B/16 at 95.08%)
- **Classical CNNs**: ResNet50 and DenseNet121 showed strong performance (99.39% and 97.95%) but required more parameters
- **Underperformer**: VGG16 achieved only 83.20% accuracy despite having the highest parameter count (138M)

**Selected Model for Production**: **EfficientNetB0** — chosen for optimal balance of accuracy (99.59%) and computational efficiency (5.3M parameters).

### Stage 2: Tumour Classification — Comparative Analysis

The classification stage categorises detected tumours into three types: Glioma, Meningioma, and Pituitary. We evaluated **6 state-of-the-art architectures** using a combined dataset of 4,705 unique MRI images after rigorous perceptual hashing deduplication.

#### Models Evaluated

| Architecture Family | Models                                |
| ------------------- | ------------------------------------- |
| **CNNs**            | EfficientNetB0, InceptionV3, ResNet50 |
| **Transformers**    | ViT-B/16, Swin-Base                   |
| **Hybrid**          | CoaTNet-0                             |

#### Dataset Composition

| Tumour Type | Count     | Percentage |
| ----------- | --------- | ---------- |
| Glioma      | 2,000     | 42.5%      |
| Meningioma  | 1,304     | 27.7%      |
| Pituitary   | 1,401     | 29.8%      |
| **Total**   | **4,705** | **100%**   |

#### Classification Performance Results

| Model              | Test Accuracy | Test Loss  | Test AUC   | Parameters |
| ------------------ | ------------- | ---------- | ---------- | ---------- |
| **InceptionV3**    | **98.58%**    | 0.0389     | **0.9995** | 27.2M      |
| **Swin-Base**      | **98.58%**    | **0.0377** | 0.9993     | 86.7M      |
| **EfficientNetB0** | **97.59%**    | 0.0582     | 0.9987     | 5.3M       |
| **CoaTNet-0**      | **97.59%**    | 0.0638     | 0.9990     | ~10M       |
| **ResNet-50**      | **96.60%**    | 0.0740     | 0.9990     | 25.6M      |
| **ViT-B/16**       | 89.52%        | 0.2737     | 0.9736     | 86.4M      |

#### Key Findings from Classification Comparative Analysis

- **Top Performers**: InceptionV3 and Swin-Base tied at 98.58% accuracy with AUC greater than 0.999
- **Efficiency Leader**: EfficientNetB0 achieved 97.59% accuracy with only 5.3M parameters — 16x smaller than Swin-Base with only 1% accuracy drop
- **Convergence Patterns**: CNN architectures (InceptionV3, EfficientNetB0, ResNet-50) converged rapidly within 3-5 epochs; transformers required longer training
- **AUC-ROC Analysis**: Five of six models achieved AUC greater than 0.998, indicating excellent class separation capability
- **ViT Limitations**: ViT-B/16 showed significantly slower convergence (89.52% accuracy), likely due to smaller dataset size and lack of inductive biases

**Selected Model for Production**: **EfficientNetB0** — chosen for optimal accuracy-to-efficiency ratio, enabling deployment on standard clinical hardware.

### Stage 3: Tumour Segmentation — Comparative Analysis

The segmentation stage performs 3D volumetric segmentation to delineate tumour sub-regions: Whole Tumour (WT), Tumour Core (TC), and Enhancing Tumour (ET). We conducted a comprehensive benchmark of **6 state-of-the-art 3D segmentation architectures** on the BraTS 2020 dataset.

#### Models Evaluated

| Architecture Family | Models                      |
| ------------------- | --------------------------- |
| **CNN Encoders**    | 3D UNet, nnU-Net, VNet      |
| **Transformers**    | UNETR, Swin-UNETR, TransBTS |

#### Dataset Composition

| Metric                  | Value           |
| ----------------------- | --------------- |
| Total cases             | 368             |
| High-grade glioma (HGG) | 293             |
| Low-grade glioma (LGG)  | 75              |
| Original dimensions     | 240 x 240 x 155 |

#### Segmentation Performance Results (Dice Scores)

| Model          | WT Dice    | TC Dice    | ET Dice    | Mean Dice  | Test Loss | Parameters | Training Time |
| -------------- | ---------- | ---------- | ---------- | ---------- | --------- | ---------- | ------------- |
| **UNet**       | **0.8647** | **0.7844** | **0.7505** | **0.7999** | 0.4109    | 5.6M       | 32.4 min      |
| **nnU-Net**    | 0.8569     | 0.7813     | 0.7452     | 0.7945     | 0.4553    | 16.5M      | 55.1 min      |
| **VNet**       | 0.8541     | 0.7108     | 0.7316     | 0.7655     | 0.4880    | 1.4M       | 16.9 min      |
| **Swin-UNETR** | 0.8128     | 0.6261     | 0.5314     | 0.6568     | 1.3411    | 10.7M      | 49.5 min      |
| **UNETR**      | 0.7128     | 0.6261     | 0.5314     | 0.6234     | 1.3411    | 137.6M     | 117.2 min     |
| **TransBTS**   | 0.6383     | 0.4618     | 0.4325     | 0.5109     | 2.1940    | 35.8M      | 70.1 min      |

#### Computational Efficiency Analysis

| Model          | Mean Dice | Params (M) | Efficiency (Dice/M) | Training (min) | Time Efficiency (Dice/min) |
| -------------- | --------- | ---------- | ------------------- | -------------- | -------------------------- |
| **VNet**       | 0.7655    | 1.4        | **0.5468**          | 16.9           | **0.0453**                 |
| **UNet**       | 0.7999    | 5.6        | 0.1428              | 32.4           | 0.0247                     |
| **nnU-Net**    | 0.7945    | 16.5       | 0.0482              | 55.1           | 0.0144                     |
| **Swin-UNETR** | 0.6568    | 10.7       | 0.0614              | 49.5           | 0.0133                     |
| **TransBTS**   | 0.5109    | 35.8       | 0.0143              | 70.1           | 0.0073                     |
| **UNETR**      | 0.6234    | 137.6      | 0.0045              | 117.2          | 0.0053                     |

#### Key Findings from Segmentation Comparative Analysis

- **Best Overall Performance**: UNet achieved highest mean Dice (0.7999), demonstrating that classic architecture remains highly competitive when properly trained
- **Most Parameter-Efficient**: VNet achieved 76.5% of UNet's performance with only 25% of its parameters (1.4M vs 5.6M)
- **Transformer Underperformance**: Despite having more parameters, transformer models (UNETR: 0.6234, TransBTS: 0.5109) significantly underperformed CNNs
- **Best Transformer**: Swin-UNETR (0.6568) was the best transformer variant, likely due to hierarchical design matching CNN decoder requirements
- **Performance Gradient**: All models showed consistent pattern: WT (easiest) greater than TC greater than ET (most challenging)
- **Training Efficiency**: VNet demonstrated superior time efficiency (0.0453 Dice/min), making it optimal for rapid prototyping

**Selected Model for Production**: **VNet** — chosen for exceptional parameter efficiency (1.4M parameters), fast training (16.9 min), and strong performance (0.7655 mean Dice), making it ideal for resource-constrained clinical environments.

### Model Architectures

#### EfficientNet-B0 (Detection and Classification)

**Architecture**: Convolutional Neural Network with compound scaling

```python
# Key characteristics:
- Parameters: 5.3 million
- Input: 224 x 224 RGB images
- Compound scaling: depth, width, resolution jointly optimised
- Depthwise separable convolutions for efficiency

# Detection head: 1 output (sigmoid activation)
# Classification head: 3 outputs (softmax activation)
```

**Why EfficientNet?**

- **Best accuracy/efficiency tradeoff**: 97.59% to 99.59% accuracy with minimal parameters
- **Transfer learning**: Pre-trained on ImageNet, fine-tuned on medical data
- **Fast inference**: Suitable for real-time clinical use

#### VNet3D (Segmentation)

**Architecture**: 3D Fully Convolutional Network with Residual Blocks

```python
class VNet3D(nn.Module):
  """
  Encoder-Decoder with skip connections
  """
  # Encoder path (downsampling)
  e1: ResBlock(4 to 16 channels)   # Input: 4 MRI modalities
  d1: Conv3d(16 to 16, stride=2)  # Downsample
  e2: ResBlock(16 to 32)
  d2: Conv3d(32 to 32, stride=2)
  e3: ResBlock(32 to 64)
  d3: Conv3d(64 to 64, stride=2)

  # Bottleneck
  b:  ResBlock(64 to 128)

  # Decoder path (upsampling)
  u3: ConvTranspose3d(128 to 64)
  r3: ResBlock(128 to 64)  # Skip connection from e3
  u2: ConvTranspose3d(64 to 32)
  r2: ResBlock(64 to 32)   # Skip connection from e2
  u1: ConvTranspose3d(32 to 16)
  r1: ResBlock(32 to 16)   # Skip connection from e1

  # Output: 3-class segmentation
  out: Conv3d(16 to 3)
```

**Segmentation Classes (BraTS Standard)**:

- **WT (Whole Tumour)**: All abnormal tissue
- **TC (Tumour Core)**: Central necrotic/cystic areas
- **ET (Enhancing Tumour)**: Active, contrast-enhancing regions

### Training Methodology

All comparative analyses followed rigorous standardised protocols:

#### Standardised Training Configuration

| Parameter       | Value                      | Rationale                                         |
| --------------- | -------------------------- | ------------------------------------------------- |
| Epochs          | 10                         | Sufficient for convergence with transfer learning |
| Optimizer       | AdamW                      | Improved weight decay handling                    |
| Learning Rate   | 3e-4                       | Standard for fine-tuning                          |
| Scheduler       | Cosine Annealing           | Smooth learning rate decay                        |
| Loss Function   | Cross-Entropy / Dice + BCE | Task-appropriate                                  |
| Mixed Precision | FP16                       | Faster training, lower memory                     |

#### Multi-GPU Strategy

- **Parallelisation**: `nn.DataParallel` for data parallelism across 2 GPUs
- **Synchronisation**: Automatic gradient averaging
- **Hardware**: 2 x NVIDIA Tesla T4 (16GB each)

#### Data Sources

Training data compiled from multiple public datasets:

- **Sartaj Bhuvaji's Brain Tumour Classification MRI** (Kaggle)
- **Jun Cheng's Brain Tumour MRI Dataset** (Kaggle)
- **BraTS Challenge Data** (MICCAI)

#### Preprocessing Pipeline

1. **Deduplication** (Perceptual Hashing)
   - Algorithm: pHash (64-bit fingerprint)
   - Threshold: Hamming distance less than or equal to 5
   - Result: 19.3% reduction (5,828 to 4,705 unique images)

2. **Data Augmentation** (Training only)
   - Random horizontal flip
   - Random rotation (±10 degrees)
   - Colour jitter (brightness, contrast)
   - Normalisation: ImageNet statistics

3. **Train/Validation/Test Split**
   - 70% training
   - 15% validation
   - 15% test (stratified by class)

---

## Installation and Setup

### Prerequisites

| Software   | Minimum Version | Recommended | Download              |
| ---------- | --------------- | ----------- | --------------------- |
| Node.js    | 18.x            | 20.x LTS    | nodejs.org            |
| Python     | 3.11            | 3.11.x      | python.org            |
| Git        | 2.30            | Latest      | git-scm.com           |
| PowerShell | 5.1             | 7.x         | Included with Windows |

### Step-by-Step Setup

#### Step 1: Clone Repository

```bash
git clone https://github.com/yourusername/neuroscan-ai.git
cd "neuroscan-ai"
```

#### Step 2: Create Python Virtual Environment

```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# macOS/Linux
python3 -m venv .venv
source .venv/bin/activate
```

#### Step 3: Install Python Dependencies

```bash
cd Backend
pip install -r requirements.txt
```

**Key Dependencies**:

- `fastapi` — Web framework
- `uvicorn[standard]` — ASGI server
- `torch` — PyTorch for ML
- `torchvision` — Computer vision utilities
- `opencv-python-headless` — Image processing
- `nibabel` — NIfTI file I/O
- `sqlmodel` — Database ORM
- `python-jose[cryptography]` — JWT tokens
- `bcrypt` — Password hashing

#### Step 4: Install Node.js Dependencies

```bash
cd ../Frontend
npm install
```

**Key Dependencies**:

- `react` — UI framework
- `typescript` — Type safety
- `vite` — Build tool
- `tailwindcss` — Styling
- `@tanstack/react-query` — State management
- `react-router-dom` — Navigation

#### Step 5: Download Pre-trained Models

The model weights are included in the repository:

- `Backend/Binary Detection/EfficientNetB0.pth` (Tumour detection)
- `Backend/Classification/EfficientNetB0.pth` (Tumour classification)
- `Backend/Segmentation/VNet.pth` (3D segmentation)

### Running the Application

#### Option A: One-Command Startup (Windows)

```powershell
.\start.ps1
```

#### Option B: Manual Terminal Sessions

**Terminal 1 — Backend:**

```bash
cd Backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000
# API available at http://localhost:8000
# Docs at http://localhost:8000/docs
```

**Terminal 2 — Frontend:**

```bash
cd Frontend
npm run dev
# App available at http://localhost:8080
```

#### Verify Installation

1. Open browser to `http://localhost:8080`
2. Click "Register" to create account
3. Log in with credentials
4. Upload test MRI files (T1, T1CE, T2, FLAIR)
5. View AI-generated analysis

---

## Usage Guide

### For Medical Professionals

#### Uploading MRI Scans

1. Navigate to **Dashboard** → "New Scan"
2. Enter **Patient Information** (name, ID)
3. Drag-and-drop or select 4 MRI files:
   - **FLAIR** — Fluid-attenuated inversion recovery
   - **T1** — Standard T1-weighted
   - **T1CE** — T1 with contrast enhancement
   - **T2** — T2-weighted
4. Click **"Process MRI"** and wait 10-30 seconds

#### Interpreting Results

The system displays:

- **Detection**: Probability of tumour presence
- **Classification**: Tumour type (Glioma/Meningioma/Pituitary)
- **Confidence Score**: Model certainty (0-100%)
- **Volumetrics**: WT, TC, ET volumes in cubic centimetres
- **Segmentation**: Colour-coded overlay on MRI slices

#### Generating Reports

1. Open any completed scan
2. Click **"View Report"** tab
3. Review automated findings
4. Click **"Download PDF"** for documentation

### For Developers

#### API Usage Examples

**Authentication:**

```bash
# Get JWT token
curl -X POST http://localhost:8000/api/auth/login \
  -d "username=doctor1&password=securepass" \
  -H "Content-Type: application/x-www-form-urlencoded"

# Response: {"access_token": "eyJhbG...", "token_type": "bearer"}
```

**Upload Scan:**

```bash
curl -X POST http://localhost:8000/api/process-mri \
  -H "Authorization: Bearer <token>" \
  -F "files=@flair.nii" \
  -F "files=@t1.nii" \
  -F "files=@t1ce.nii" \
  -F "files=@t2.nii" \
  -F "patientName=John Doe" \
  -F "patientId=PT-001"
```

**Get Scan Results:**

```bash
curl http://localhost:8000/api/scans/scan-abc123 \
  -H "Authorization: Bearer <token>"
```

#### Extending the Pipeline

To add a new AI model:

1. **Train your model** and save weights to `Backend/YourModel/`
2. **Add loading code** in `processor.py`:
   ```python
   def load_models(self, ..., yourmodel_path: str):
       self.yourmodel = YourModel().to(self.device)
       self.yourmodel.load_state_dict(torch.load(yourmodel_path))
   ```
3. **Add inference method**:
   ```python
   def run_yourmodel(self, image):
       # Your inference logic
       return results
   ```
4. **Update API endpoint** in `main.py` to call new method

### For Researchers

#### Reproducing Training Results

All training notebooks are in `Backend/` subdirectories:

**Detection Models (9 architectures)**:

- `Backend/Binary Detection/` — ResNet50, VGG16, DenseNet121, EfficientNetB0, MobileNetV3, InceptionV3, ViT-B/16, Swin Base, CoaTNet-0

**Classification Models (6 architectures)**:

- `Backend/Classification/` — EfficientNetB0, InceptionV3, ResNet50, Swin-Base, ViT-B16, CoaTNet0

**Segmentation Models (6 architectures)**:

- `Backend/Segmentation/` — UNet, nnU-Net, VNet, UNETR, Swin-UNETR, TransBTS

Each notebook includes:

- Complete preprocessing pipeline
- Model architecture definition
- Training loop with metrics logging
- Evaluation on test set
- Visualisation of results

---

## Detailed Research Documentation

The repository contains three comprehensive research papers documenting our comparative analyses:

### 1. Binary Detection Research

**Location**: `Backend/Binary Detection/README.md`

Documents the comparative study of 9 deep learning architectures for binary brain tumour detection. Key findings include:

- CoaTNet-0 and EfficientNetB0 achieving 99.59% accuracy
- Comprehensive analysis across CNN and transformer families
- Parameter efficiency analysis showing EfficientNetB0 as optimal
- Detailed training dynamics and convergence analysis

### 2. Classification Research

**Location**: `Backend/Classification/README.md`

Presents a comparative study of 6 state-of-the-art architectures for multi-class brain tumour classification (Glioma, Meningioma, Pituitary). Key findings include:

- InceptionV3 and Swin-Base achieving 98.58% accuracy
- Perceptual hashing deduplication methodology
- Cross-architecture performance comparison under identical conditions
- Clinical deployment recommendations

### 3. Segmentation Research

**Location**: `Backend/Segmentation/README.md`

Comprehensive benchmark of 6 state-of-the-art 3D segmentation architectures on BraTS 2020 dataset. Key findings include:

- UNet achieving highest mean Dice (0.7999)
- VNet demonstrating exceptional parameter efficiency (1.4M parameters, 0.7655 Dice)
- Transformer underperformance analysis (UNETR: 0.6234, TransBTS: 0.5109)
- Per-region performance analysis (WT, TC, ET)

---

## Contributing

We welcome contributions from the community. Areas of interest:

- **Bug fixes** — Report issues via GitHub Issues
- **New features** — Propose enhancements
- **Documentation** — Improve guides and tutorials
- **Model improvements** — Better accuracy or efficiency
- **UI/UX** — Design improvements
- **Internationalization** — Multi-language support

### Development Workflow

1. Fork the repository
2. Create feature branch: `git checkout -b feature/your-feature`
3. Make changes with clear commit messages
4. Add tests if applicable
5. Submit Pull Request with detailed description
