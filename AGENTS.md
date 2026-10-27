# Agent Guide

Purpose: rules and commands every agent task in this repository needs. Read when: always.
Last verified: 2026-10-27.

## Commands

- Set up and run everything: `npm run dev` (root; creates `.venv`, installs, writes a dev `.env`, starts backend on 8000 and frontend on 8080).
- Backend tests: `pytest` in `Backend/`; one file: `pytest tests/test_scans.py`.
- Backend lint and format: `ruff check neuroscan tests main.py` and `ruff format --check neuroscan tests main.py` in `Backend/`.
- Backend types and boundaries: `mypy` and `lint-imports` in `Backend/`.
- Backend security: `bandit -q -r neuroscan main.py` and `pip-audit -r requirements.txt -r requirements-dev.txt` in `Backend/`.
- Frontend, all checks: `npm run check` in `Frontend/` (typecheck, lint, Prettier, jscpd, Vitest, build); one test file: `npx vitest run src/test/api.test.ts`.
- Repository formatting: `npm run format:check` at the root (CI job `format`); `black .` for notebooks and Python outside `Backend/`.
- Papers: `npm run papers` (needs Tectonic 0.17.0 or later in `Materials/` or on `PATH`).
- CI (`.github/workflows/ci.yml`) runs all of the above plus `npm audit --audit-level=moderate` and gitleaks.

## Conventions

- Backend layers `app > api > scans > inference`; `inference/` and `security/` stay free of FastAPI and the database. See `ARCHITECTURE.md#dependency-rules`.
- Frontend calls the API only through `Frontend/src/lib/api.ts`; server state lives in React Query hooks in `hooks/`.
- Keep `Backend/neuroscan/schemas.py` and `Frontend/src/lib/types.ts` in sync by hand.
- Schema changes need a new Alembic migration; never edit an applied one.
- New settings go on `Settings` in `config.py`, plus `.env.example` and the README configuration table.
- Every result number in docs, papers, deck and report comes from `Materials/Notebooks/<task>/metrics/` or logs; never round or invent.
- Notebooks are never executed locally; they run on Kaggle (2x T4).

## Gotchas

- **Weights missing after clone**: `*.pth` is Git-ignored. `npm run dev` downloads them from the `weights-v1` GitHub release and verifies the SHA-256 hashes in `scripts/dev.mjs`; new weights need a new release and updated hashes. Without them the app starts but disables uploads or segmentation.
- **SegResNet fails to load**: `networks.segresnet()` and the `monai` pin must match the checkpoint. `tests/test_inference.py` guards the layout.
- **Modality detection disagrees between UI and API**: the order lives in `scans/storage.py` and `components/upload/DropZone.tsx`. Change both.
- **Supplementary references show `??`**: the classification paper's `main.tex` and `supplementary.tex` read each other's `.aux`. Build with `npm run papers -- Standardised`.
- **Prettier rewrites line endings on Windows**: `.gitattributes` forces LF. Do not change it.
- **Backslashes in shell commands**: write scripts and JSON with an editor tool, not through shell quoting.

## Docs

- `README.md`: setup, configuration, API, results summary, troubleshooting.
- `ARCHITECTURE.md`: module map, dependency rules, where new code goes.
- `DESIGN.md`: before any UI change.
- `SECURITY.md`: before touching auth, uploads, secrets, dependencies or CI.
- `RESEARCH.md`: before revisiting a model, dataset, library or paper reference.
- `Materials/Notebooks/README.md`: every notebook cell group and result.
