# NeuroScan AI

## Overview

- Upload MRI scans and get instant AI results for detection, classification, and segmentation.
- The backend runs on FastAPI and PyTorch while the browser shows dashboards, uploads, and 2D/3D viewers.
- Everything runs from one folder: set up the Python venv and npm modules, then use `start.ps1` or manual commands.
- This README explains each part in simple steps so anyone can use the app today.

## Repository layout

- `.venv/` keeps the Python environment so the backend stays isolated (do not check it in).
- `Backend/` contains FastAPI routes, the AI processor, SQLModel data models, SQLite database, and scan uploads.
- `Frontend/` hosts the Vite + React UI with Tailwind styles, Radix/Sonner components, Recharts charts, and vtk.js viewers.
- `start.ps1` runs both services together with colored logs, dependency checks, and browser launch.
- Config files such as `.gitignore`, `.gitattributes`, and `.prettierrc.json` keep formatting steady.

## Full tech stack

- Python 3.11 (FastAPI, Uvicorn, SQLModel, pydantic)
- SQLite (neuroscan.db)
- bcrypt (password hashing)
- python-jose (JWT tokens)
- OpenCV + scikit-image + numpy
- nibabel (NIfTI support)
- PyTorch + Torchvision (EfficientNetB0, custom VNet3D)
- PIL, cv2, nifti-reader-js (image helpers)
- Node.js 20+ + npm
- Vite 7 + React 18 + TypeScript
- Tailwind CSS 3 + tailwind-merge + clsx
- Radix UI primitives (dialogs, dropdowns, tabs, etc.)
- Lucide icons
- Sonner notifications
- TanStack Query v5 + React Router v6
- Recharts
- vtk.js (Kitware packages) for 3D volume rendering
- Browser APIs (fetch, localStorage)
- PowerShell scripting (start.ps1)
- Git

## Backend overview

- FastAPI serves routes from `Backend/main.py` and taps `processor.py` for AI work.
- JWT protects every route, bcrypt secures passwords, and SQLModel builds the SQLite schema.
- The AI processor loads EfficientNetB0 for detection/classification and a custom VNet3D for segmentation.
- Uploaded MRI files land in `Backend/uploads/<scan-id>/`, and segmentation masks save as NIfTI slices for viewer export.
- APIs cover auth, scan CRUD, processing, slice streaming, segmentation overlays, and downloads.

## Frontend overview

- `App.tsx` wires QueryClient, AuthProvider, tooltips, toasts, and the protected layout around routes.
- Layout includes a collapsible sidebar, top search bar, notification area, profile menu, and the main content frame.
- Pages: Dashboard (`Index`), Upload, Scan Library, Scan Detail, Reports, Settings, Login, Register, NotFound.
- Dashboard shows stats, recent scans, and tumour volume charts using Recharts.
- Upload drop zone auto-detects modalities, captures patient info, warns about missing files, and posts to `/api/process-mri`.
- Scan detail view toggles between the 2D slice viewer, vtk.js 3D viewer, pipeline timeline, and report cards.
- Shared UI components (buttons, cards, dialogs, dropdowns, tables, sliders, switches, toasts, tooltips) use Tailwind via the `cn()` helper.
- Hooks such as `useScans` and `useConfig` keep data in sync with React Query and a central text config.

## Key features

- Simple JWT login/register and `ProtectedRoute` to keep scans private.
- Drag-and-drop MRI upload with modality detection, missing modality warnings, and patient metadata.
- Dashboard metrics highlight total scans, processing state, completed results, and flagged findings.
- Scan library/recent scans tables let you view, edit, or delete scans quickly.
- Scan detail page offers 2D slices, vtk.js 3D volumes, segmentation toggle controls, pipeline status, and download buttons.
- Slice streaming, segmentation overlays, and downloads all use dedicated backend endpoints.
- `start.ps1` keeps backend and frontend together with colorized logs, dependency checks, and automatic browser launch.

## Running the full stack

- Prerequisites: Windows PowerShell 7+, Node.js 20+, Python 3.11+, optional CUDA for GPU acceleration.
- Set up the backend:
  - `python -m venv .venv`
  - `.\.venv\Scripts\Activate.ps1`
  - `pip install -r Backend\requirements.txt`
- Set up the frontend:
  - `cd Frontend`
  - `npm install`
- Launch services:
  - Run `start.ps1` from the repo root. It installs any missing dependencies, starts `uvicorn main:app --reload --host 0.0.0.0 --port 8000` and `npm run dev`, streams logs, and opens `http://localhost:8080`.
  - Or start backend and frontend manually in two terminals if preferred.
- Config reminders:
  - Replace hard-coded `SECRET_KEY`, `ACCESS_TOKEN_EXPIRE_MINUTES`, and `UPLOAD_DIR` in `Backend/main.py` with environment variables for production.
  - Swap `API_BASE_URL` in `Frontend/src/lib/api.ts` for a deployed backend host.
  - Clean up old scan folders under `Backend/uploads/` or delete scans via the UI to reclaim disk.

## API contract

- Authentication: `POST /api/auth/register`, `POST /api/auth/login`, `GET /api/auth/me`, `PUT /api/auth/me`.
- Scan endpoints: `GET /api/scans`, `GET /api/scans/{id}`, `POST /api/process-mri`, `PUT /api/scans/{id}`, `DELETE /api/scans/{id}`.
- Data streaming: `GET /api/scans/{id}/slice/{idx}` returns JPEG slices with `X-Total-Slices` header.
- Segmentation: `GET /api/scans/{id}/segmentation/{idx}` returns PNG masks with WT/TC/ET colors.
- Downloads: `GET /api/scans/{id}/download/{key}` fetches modality files or segmentation NIfTI.
- All endpoints require bearer tokens, reply with JSON `detail` on errors, and use standard HTTP codes.

## Data & model architecture

- Database: SQLite file `neuroscan.db`, `User` and `Scan` tables, JSON columns for modalities, file paths, and results.
- AI models: EfficientNetB0 for detection/classification (glioma, meningioma, pituitary), custom VNet3D for 3D segmentation.
- Upload flow: accepts `.nii`, `.nii.gz`, `.dcm`, `.dicom`, tags modalities by filename, generates unique scan IDs, and stores files per user.
- Segmentation masks save as categorical NIfTI (1=WT, 2=TC, 3=ET) and stream as PNG overlays for viewers.
- Frontend displays classification confidence, tumour volumes (WT/TC/ET), and pipeline status per scan.

## Testing, linting & quality

- Backend: `pip install -r Backend/requirements.txt` before adding tests (none supplied yet).
- Frontend: `npm run lint` (ESLint + TypeScript) and `npm run test` or `npm run test:watch` (Vitest).
- Formatting: `.prettierrc.json` keeps layout consistent for editors.
- Manual QA: drop sample MRI files, watch scans appear on dashboard, inspect 2D/3D viewers, toggle segmentation overlays, and try download buttons.

## Troubleshooting & maintenance

- Token loops: clear `token` and `user` from localStorage or log out; the backend auto-clears them on 401.
- Missing modality warnings: upload at least one of FLAIR, T1, T1CE, or T2, or toggle the modal selector before uploading.
- Segmentation missing: the backend needs all four modalities to run VNet; detection/classification still work if some are missing.
- Model load issues: ensure `.pth` files exist in `Backend/Binary Detection`, `Backend/Classification`, and `Backend/Segmentation` before starting FastAPI.
- Database cleanup: delete scans via the UI or `DELETE /api/scans/{id}` to remove uploads and records.
- 3D viewer lag: vtk.js and nifti-reader-js load lazily; wait 3�5 seconds for the volume to appear on slower machines.

## Next steps

- Replace the hard-coded JWT secret, upload path, and API base with `.env` values for production safety.
- Add sample MRI fixtures and integration tests to cover upload-to-detail flows.
- Deploy the backend to a GPU-enabled host and point the frontend to that API.
- Create cross-platform starters (bash scripts) so macOS/Linux teams can launch both services like `start.ps1` does.
- Add PDF/CSV export logic so clinicians can download readable reports straight from the UI.
