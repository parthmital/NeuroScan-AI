# AGENTS.md — NeuroScan AI Developer & Agent Guide

This document defines core architectural patterns, operational conventions, workflows, commands, and constraints for coding agents working within the **NeuroScan AI** codebase.

---

## 1. Project Overview & Architecture

**NeuroScan AI** is an end-to-end medical imaging platform for automated brain tumour analysis from multi-parametric MRI scans (FLAIR, T1, T1CE, T2).

### Three-Stage AI/ML Pipeline

1. **Binary Detection**: EfficientNet-B0 (PyTorch) classifies MRI slices as tumour vs healthy.
2. **Multi-Class Classification**: EfficientNet-B0 (PyTorch) categorises detected tumours into _Glioma_, _Meningioma_, or _Pituitary_.
3. **3D Volumetric Segmentation**: Custom 3D VNet (PyTorch) extracts Whole Tumour (WT), Tumour Core (TC), and Enhancing Tumour (ET) sub-regions from 4-modality NIfTI volumes (`.nii`/`.nii.gz`).

### High-Level System Architecture

```
NeuroScan-AI/
├── Backend/                    # FastAPI async REST API & PyTorch pipeline
│   ├── main.py                 # FastAPI application, routes, and auth
│   ├── processor.py            # PyTorch inference engine (Detection, Classification, VNet3D)
│   ├── models.py               # SQLModel table & Pydantic request/response schemas
│   ├── database.py             # SQLite engine and session generator
│   ├── requirements.txt        # Backend Python dependencies
│   ├── Binary Detection/       # Research notebooks, metrics, and EfficientNetB0.pth
│   ├── Classification/         # Research notebooks, metrics, and EfficientNetB0.pth
│   ├── Segmentation/           # Research notebooks, metrics, and VNet.pth
│   ├── Materials/              # Academic papers (LaTeX), presentations, build-latex.ps1, tectonic.exe
│   └── uploads/                # Local directory for uploaded scans and generated segmentations
├── Frontend/                   # React 18 + TypeScript + Vite SPA
│   ├── src/
│   │   ├── pages/              # Route views (Index, Login, Register, ScanDetail, ScanLibrary, Reports, Settings)
│   │   ├── components/         # Modular UI components (auth, dashboard, layout, scan, ui, upload)
│   │   ├── lib/                # api.ts (HTTP client), types.ts, config.ts, utils.ts
│   │   ├── hooks/              # Custom React hooks (e.g. use-auth, use-toast)
│   │   └── test/               # Vitest suite
│   ├── package.json            # Frontend dependencies & npm scripts
│   ├── vite.config.ts          # Vite configuration (port 8080, GLSL plugin, @ alias)
│   └── tailwind.config.ts      # Tailwind CSS configuration & design tokens
├── start.ps1                   # Unified local development launcher
└── docker-compose.yml          # Container orchestration (Backend + Frontend Nginx)
```

---

## 2. Technology Stack & Key Dependencies

### Backend

- **Framework**: FastAPI (async routes, lifespan context manager)
- **Runtime**: Python 3.11+ via repo-local `.venv`
- **Database & ORM**: SQLModel (SQLAlchemy + Pydantic) on SQLite (`Backend/neuroscan.db`)
- **Authentication**: JWT (`python-jose` with HS256) + `bcrypt` password hashing
- **ML / Medical Imaging**: PyTorch (`torch`, `torchvision`), `nibabel` (NIfTI I/O), `OpenCV` (`cv2`), `Pillow`, `numpy`

### Frontend

- **Framework & Language**: React 18.3.1, TypeScript 5.6.0
- **Bundler & Dev Server**: Vite 7.3.1 (default port `8080`)
- **Styling**: Tailwind CSS 3.4.17, `tailwindcss-animate`, shadcn/ui patterns (Radix UI primitives)
- **State & Data Fetching**: TanStack React Query 5.83.0, React Context (`AuthContext`)
- **3D & Visualisation**: `@kitware/vtk.js` 35.2.0, `recharts` 2.15.4, `nifti-reader-js`
- **Testing**: Vitest 3.2.4, `@testing-library/react`

---

## 3. Essential Commands & Workflows

### Local Development (Windows / PowerShell)

```powershell
# Unified Startup (starts both backend on :8000 and frontend on :8080 with live logs)
.\start.ps1

# Backend Only (from repository root)
.\.venv\Scripts\Activate.ps1
cd Backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Frontend Only (from Frontend directory)
cd Frontend
npm run dev

# Frontend Production Build
cd Frontend
npm run build

# Frontend Tests
cd Frontend
npm run test
npm run test:watch

# Frontend Linting
cd Frontend
npm run lint

# Compile Academic LaTeX Papers
.\Backend\Materials\build-latex.ps1
```

### Dependency Installation

- **Python**: `.\.venv\Scripts\pip install -r Backend\requirements.txt`
- **Node**: `cd Frontend && npm install`

---

## 4. Conventions & Code Standards

### Python / Backend Guidelines

1. **Type Annotations**: Provide explicit type hints for function signatures and API route returns (`from typing import Optional, List, Dict, Tuple, Any`).
2. **Device Agnostic Inference**: Ensure PyTorch code supports CUDA acceleration while falling back cleanly to CPU (`torch.device("cuda" if torch.cuda.is_available() else "cpu")`).
3. **Database Sessions**: Use FastAPI dependency injection with `Session = Depends(get_session)` from `database.py`. Never leave sessions open manually.
4. **Ownership Verification**: All user-scoped routes (`/api/scans/{scan_id}`, etc.) must verify `scan.userId == current_user.id` and return `403 Forbidden` on mismatch.
5. **Safe File Operations**: Sanitize and construct file paths using `os.path.join` and verify directory boundaries against directory traversal.

### TypeScript / Frontend Guidelines

1. **Strict Typing**: Avoid `any`. Define and reuse shared types in `Frontend/src/lib/types.ts`.
2. **Path Aliasing**: Use `@/` for imports resolving to `Frontend/src/` (configured in `vite.config.ts` and `tsconfig.json`).
3. **Component Structure**:
   - Keep UI primitives in `Frontend/src/components/ui/`.
   - Feature-specific components belong in `components/auth/`, `components/dashboard/`, `components/scan/`, etc.
   - Route components live in `pages/`.
4. **API Requests**: Route all backend communications through `Frontend/src/lib/api.ts` or custom TanStack Query hooks. Ensure JWT auth headers are attached automatically via `getAuthHeader()`.
5. **Error & Loading States**: Always handle pending and error states gracefully in UI components using toast notifications (`sonner`) or skeleton placeholders.

---

## 5. Constraints & Rule Checklist for Agents

### DO:

- **Keep changes surgical**: Modify only files directly relevant to the user request.
- **Maintain backward compatibility**: Preserve API request/response schemas in `models.py` and `Frontend/src/lib/types.ts`.
- **Use local environment tools**: Always execute commands using the repo-local `.venv` (`.\.venv\Scripts\...`) and local `Frontend/node_modules`.
- **Handle NIfTI files properly**: Account for slice orientation, normalization, and volume dimensions when processing `.nii` / `.nii.gz` files in `processor.py`.

### DON'T:

- **Do not commit large binary artifacts**: Do not track `.pth`, `.keras`, `.h5`, `.db`, `.nii`, or `.nii.gz` in git commits beyond the existing baseline files.
- **Do not introduce global dependencies**: Avoid global `pip install` or `npm install -g`.
- **Do not hardcode API origins**: Rely on `.env` (`CORS_ORIGINS`, `VITE_API_URL`, `JWT_SECRET`) rather than fixed URLs.
- **Do not alter ML model architectures without updating weights**: Modifying `VNet3D` or `_get_efficientnet_model` will cause state-dict mismatches when loading existing `.pth` files.
- **Do not invent mock data or fake endpoints**: Use the verified schema defined in `models.py` and `main.py`.
