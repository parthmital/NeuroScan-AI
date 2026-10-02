# Architecture

NeuroScan AI is one deployable web application (a FastAPI backend and a React single-page app) plus self-contained Kaggle research notebooks. This document says where code lives, which way dependencies may point, and when the design should change.

| Document                              | Owns                                                    |
| ------------------------------------- | ------------------------------------------------------- |
| [README.md](README.md)                | Setup, configuration, API reference, troubleshooting    |
| ARCHITECTURE.md                       | Module map, dependency rules, decisions, growth signals |
| [DESIGN.md](DESIGN.md)                | Frontend design system and UI decisions                 |
| [SECURITY.md](SECURITY.md)            | Threat model, controls, review results, residual risks  |
| `Backend/<research folder>/README.md` | Notebook walkthroughs and results                       |

## Module map

### Backend (`Backend/neuroscan/`)

| Module             | Responsibility                                                                                               | Public surface                                             |
| ------------------ | ------------------------------------------------------------------------------------------------------------ | ---------------------------------------------------------- |
| `app.py`           | Composition root: builds settings, engine, storage, models, pipeline, middleware, routers; owns the lifespan | `create_app(settings=None, processor=None)`                |
| `config.py`        | Reads and validates every setting from the environment once                                                  | `Settings`, `get_settings()`                               |
| `observability.py` | JSON (production) and readable (development) log formatters, and the request-ID context variable             | `configure_logging`, `request_id_var`                      |
| `db.py`            | Engine construction (SQLite pragmas) and running Alembic migrations                                          | `build_engine`, `migrate`                                  |
| `models.py`        | Database tables and the status/step/modality enums                                                           | `User`, `Scan`, `ScanStatus`, `PipelineStep`, `Modality`   |
| `schemas.py`       | Request and response shapes, including input rules (password policy, patterns)                               | `UserCreate`, `ScanPublic`, ...                            |
| `api/`             | HTTP only: routers, request dependencies, ASGI middleware                                                    | `auth.router`, `scans.router`, `health.router`             |
| `scans/`           | Scan lifecycle: file storage, background pipeline, CSV/PDF exports                                           | `ScanStorage`, `PipelineRunner`, `process_scan`, `exports` |
| `inference/`       | Framework-free ML: network definitions, NIfTI I/O, `BrainProcessor`                                          | `BrainProcessor`, `volumes.*`, `networks.*`                |
| `security/`        | Primitives with no HTTP knowledge: password hashing, JWTs, TOTP, rate limiter                                | module functions and `RateLimiter`                         |
| `migrations/`      | Alembic environment and versioned migrations                                                                 | applied by `db.migrate`                                    |
| `cli.py`           | Operator account-recovery commands                                                                           | `python -m neuroscan.cli ...`                              |

`Backend/main.py` is only the ASGI entry point.

### Research and materials

`Backend/Binary Detection/`, `Backend/Classification/` and `Backend/Segmentation/` hold Kaggle notebooks and their saved outputs. The application never imports them; it only reads the deployed `.pth` files from these folders by default (paths are configurable). `Backend/Materials/` holds papers and decks and is not code.

### Frontend (`Frontend/src/`)

| Folder                  | Responsibility                                                                                                                                                       |
| ----------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `lib/`                  | Framework-light building blocks: the only HTTP client (`api.ts`), shared types, UI text (`config.ts`), and pure helpers (colour, formatting, password policy, theme) |
| `hooks/`                | Data and state hooks over `lib/`: scan queries and mutations, overlay colours, report downloads                                                                      |
| `components/ui/`        | Generic, app-agnostic primitives (shadcn/Radix based)                                                                                                                |
| `components/<feature>/` | Feature components: `auth`, `dashboard`, `layout`, `scan`, `settings`, `upload`                                                                                      |
| `pages/`                | Route screens that compose components and hooks                                                                                                                      |
| `App.tsx`               | Providers, routing, lazy loading, route guard                                                                                                                        |

## Dependency rules

Arrows point from the importer to what it may import. Anything not listed is forbidden.

```text
Backend:  app -> api -> scans -> inference
          api, scans, cli -> models, schemas, security, config, db
          inference -> (numpy, torch, nibabel, cv2 only)
          security  -> (no fastapi, starlette, api, scans, inference, db)

Frontend: pages -> components -> hooks -> lib
          components/ui -> lib only
```

Enforcement:

- Backend: `lint-imports` with the contracts in `Backend/pyproject.toml` (layers `app > api > scans > inference`; inference framework-free; security independent of HTTP, storage and inference).
- Frontend: `no-restricted-imports` rules in `Frontend/eslint.config.js` (`lib` cannot import components, hooks or pages; `hooks` cannot import components or pages; `components/ui` cannot import feature code; no component imports a page).
- Duplication: `npm run lint:dupes` runs jscpd over `Frontend/src` and `Backend/neuroscan` and fails on any clone of 60+ tokens.

All of these run in CI.

## Local development launch

`npm run dev` at the root is the only launcher. [`scripts/dev.mjs`](scripts/dev.mjs) uses only Node.js built-ins: it creates `.venv` and installs Python packages, runs `npm ci`, and writes a development `.env` on first run, skips setup later unless a requirements file or the npm lockfile hash changed, then starts the backend and frontend (one window each on Windows, prefixed output elsewhere) and opens the browser once both respond. The README describes its behaviour in full.

## Where new code goes

| Adding...                         | Put it in                                                                    |
| --------------------------------- | ---------------------------------------------------------------------------- |
| A new API route                   | A router in `api/` (new file for a new resource), registered in `app.py`     |
| Business logic behind a route     | `scans/` or a new sibling domain package, never in the router                |
| A new model or preprocessing step | `inference/`, behind `BrainProcessor` or a new class; keep it framework-free |
| A new table or column             | `models.py` plus a new Alembic migration (`alembic revision --autogenerate`) |
| A new setting                     | A field on `Settings` with validation, plus `.env.example` and the README    |
| A frontend API call               | A function in `lib/api.ts`; components never call `fetch` directly           |
| Server state in the UI            | A React Query hook in `hooks/`                                               |
| A reusable visual primitive       | `components/ui/`; a feature-specific one in `components/<feature>/`          |

## Decisions

| Decision                                                                        | Reason                                                                                                                                                                                                                                                                            |
| ------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Modular monolith, one process                                                   | One team, one deployment, one GPU/CPU inference budget. Modules are separated so any of them can be extracted later without rewriting.                                                                                                                                            |
| Background pipeline on a single in-process worker thread                        | Inference takes seconds and holds large tensors; one worker bounds memory and keeps request threads free. Jobs are idempotent, and unfinished scans are re-queued at start-up, so a restart loses no work. A bounded queue (`MAX_PENDING_JOBS`) applies back-pressure with `503`. |
| Cookie sessions instead of bearer tokens in `localStorage`                      | An `HttpOnly` cookie cannot be read by injected script. `SameSite=Strict` plus a required custom header blocks CSRF. A per-user token version gives server-side revocation without a session table.                                                                               |
| Same-origin deployment (Vite proxy in development, reverse proxy in production) | Keeps the cookie first-party and removes CORS from the normal path.                                                                                                                                                                                                               |
| Alembic migrations applied at start-up, with baseline stamping                  | The schema must evolve (auth hardening added columns) without losing existing databases created by `create_all`.                                                                                                                                                                  |
| SQLite with WAL                                                                 | The data volume is small and there is one writer process; WAL lets the worker write while requests read.                                                                                                                                                                          |
| Server-side CSV and PDF generation                                              | One implementation of report content and of spreadsheet-formula escaping.                                                                                                                                                                                                         |
| `weights_only=True` model loading                                               | Weight files are data; this refuses pickled code.                                                                                                                                                                                                                                 |
| Settings validated by Pydantic at import time                                   | A misconfigured server fails at start instead of running with a weak default.                                                                                                                                                                                                     |

## Allowed duplication

| Duplicate                                                                                    | Why it stays                                                                                                                                        |
| -------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------- |
| Modality detection from file names (`scans/storage.py` and `components/upload/DropZone.tsx`) | The upload screen labels files before sending them; the backend decides authoritatively. Both use the same order and markers; change them together. |
| Password policy (`schemas.py` and `lib/password.ts`)                                         | Client-side feedback before submit; the server is authoritative.                                                                                    |
| API shapes in `schemas.py` and `lib/types.ts`                                                | Two languages with no shared code generation yet (see growth signals).                                                                              |
| Kaggle notebooks repeating setup and training code                                           | Each notebook must run standalone on Kaggle; shared code is defined once per notebook, in an early cell.                                            |
| `migrations/versions/0001_baseline.py` restating the original tables                         | Applied migrations are history and are never edited.                                                                                                |
| shadcn components in `components/ui/`                                                        | Vendored generator output; excluded from the clone check.                                                                                           |
| Security headers in `api/middleware.py` and `Frontend/vite.config.ts`                        | Different processes and languages; the API sends a stricter CSP than the page. Change the shared headers together.                                  |

## Current fit

Re-evaluated on 2 October 2026 at commit `0a5a530`. The modular monolith still fits: one deployment, 3,436 lines in 30 backend files (`Backend/neuroscan/`) and 7,960 lines in 67 frontend files (`Frontend/src/` without tests), and every boundary check passes (3 import contracts kept, 0 frontend clones, strict mypy and TypeScript clean). No restructuring was needed in this pass.

Files to split the next time they change, because each now mixes several jobs:

| File                                                 | Lines | Mix                                                       | Split into                                                                 |
| ---------------------------------------------------- | ----- | --------------------------------------------------------- | -------------------------------------------------------------------------- |
| `Backend/neuroscan/scans/exports.py`                 | 786   | CSV export, PDF layout and canvas, clinical findings text | `scans/exports/csv.py` and `scans/exports/pdf.py`                          |
| `Frontend/src/components/scan/viewer/SliceView.tsx`  | 646   | Canvas drawing, pointer tools, keyboard, pane chrome      | Drawing helpers next to `slicing.ts`; the component keeps state and events |
| `Frontend/src/components/scan/viewer/VolumeView.tsx` | 565   | vtk.js scene setup, render presets, input handling        | A scene module without React; the component keeps lifecycle                |
| `Frontend/src/components/scan/viewer/ScanViewer.tsx` | 543   | Pane layout, tool state, measurements, capture            | Keep layout; move tool and measurement state into a hook                   |

## Growth signals

Revisit this design when any of these become true:

- **More than one backend instance is needed** (availability or throughput): move the job queue to a shared broker (for example Redis with RQ or Celery), the rate limiter to shared storage, and the database to PostgreSQL. `PipelineRunner.submit` and `RateLimiter.allow` are the seams.
- **Inference needs its own hardware or scaling**: extract `inference/` behind an internal HTTP or queue interface; it already has no web or database dependencies.
- **A second client or external consumers use the API**: generate `lib/types.ts` from the OpenAPI schema instead of maintaining it by hand, and version the API.
- **Several user roles or tenants appear**: add an authorisation layer next to `owned_scan` in `api/scans.py` rather than per-route checks.
- **Email becomes available**: replace operator-only recovery with a verified email reset flow.
- **A module passes about 500 lines or a second report format is added**: split it as in [Current fit](#current-fit).
