# Design

The design system and UI decisions of the NeuroScan AI frontend (`Frontend/src/`). Every value here is taken from the code; when the code changes, update this file in the same commit. Setup is in the [README](README.md), module boundaries in [ARCHITECTURE.md](ARCHITECTURE.md).

## Contents

- [Brief](#brief)
- [Palette](#palette)
- [Typography](#typography)
- [Spacing, layout and breakpoints](#spacing-layout-and-breakpoints)
- [Components](#components)
- [Scan viewer](#scan-viewer)
- [Icons](#icons)
- [Motion](#motion)
- [Copy and terminology](#copy-and-terminology)
- [Accessibility](#accessibility)
- [Native UI policy](#native-ui-policy)
- [Decisions](#decisions)
- [Known gaps](#known-gaps)

## Brief

| Item             | Value                                                                                                   |
| ---------------- | ------------------------------------------------------------------------------------------------------- |
| Subject          | AI-assisted brain tumour detection, classification and segmentation of MRI volumes                      |
| Users            | Clinicians and researchers reviewing a case's scans and AI results                                      |
| Primary task     | Upload a case, wait for processing, then read the result in the viewer and export a report              |
| Screens          | Login, Register, Dashboard, Upload Scan, Scan Library, Scan Detail (viewer), Reports, Settings, 404     |
| Success criteria | Results, state and next action readable at a glance; MRI greyscale shown without colour cast; any width |

The first screen after sign-in is the working dashboard, not a landing page.

## Palette

Tokens are HSL triplets on `:root` and `.dark` in [`index.css`](Frontend/src/index.css), exposed to Tailwind as `--color-*`. The palette is deliberately low-chroma so greyscale MRI reads true.

| Name     | Tokens                                                         | Light                               | Dark                               | Role                                                        |
| -------- | -------------------------------------------------------------- | ----------------------------------- | ---------------------------------- | ----------------------------------------------------------- |
| Graphite | `background`, `foreground`, `card`, `muted`, `border`, `input` | bg `220 14% 97%`, ink `222 20% 11%` | bg `220 10% 7%`, ink `220 12% 91%` | Neutral surfaces and text                                   |
| Cobalt   | `medical`, `ring`, `sidebar-ring`                              | `226 71% 48%`                       | `226 92% 70%`                      | The single accent: focus, selection, active state, progress |
| Moss     | `success`                                                      | `148 46% 30%`                       | `148 42% 56%`                      | Clear result, completed                                     |
| Ochre    | `warning`                                                      | `32 88% 36%`                        | `38 86% 62%`                       | Flagged (tumour detected), cautions                         |
| Brick    | `destructive`                                                  | `4 68% 44%`                         | `4 72% 60%`                        | Failed, delete, errors                                      |

Rules:

- `primary` is ink (near black in light, near white in dark), so primary buttons are neutral and Cobalt stays reserved for state.
- `medical-light` (`226 85% 96.5%` light, `226 40% 15%` dark) is the tinted Cobalt surface.
- Selection is Cobalt at 22% opacity; the text caret is Cobalt.
- Scrollbars use `--scrollbar` (`220 9% 78%` light, `220 7% 26%` dark).
- The viewer has its own fixed palette; see [Scan viewer](#scan-viewer).

## Typography

| Role | Family                                                                              | Used for                                                                                 |
| ---- | ----------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------- |
| Sans | IBM Plex Sans Variable (`@fontsource-variable/ibm-plex-sans`), then system UI fonts | All interface text                                                                       |
| Mono | IBM Plex Mono (`@fontsource/ibm-plex-mono`), then `ui-monospace`                    | Numbers (stat values with `tabular-nums`), segmented control labels, overlines, initials |

Fonts are bundled through npm, so no request leaves the origin (the CSP allows only `'self'`).

Scale in use (counted across `src/**/*.tsx`):

| Size               | Line height | Typical use                                                            |
| ------------------ | ----------- | ---------------------------------------------------------------------- |
| 28px               | 36px        | Dashboard stat values (mono)                                           |
| 24px (`text-2xl`)  | 32px        | Page titles                                                            |
| 16px (`text-base`) | 24px        | Long-form text                                                         |
| 14px (`text-sm`)   | 20px        | Body text, form fields                                                 |
| 13px               | 20px        | Dense UI text: tables, card titles, status labels (the most used size) |
| 12px (`text-xs`)   | 16px        | Descriptions, inline errors                                            |
| 10 to 11px         | -           | Mono uppercase overlines and segmented labels                          |

`index.css` restores Tailwind v3 absolute line heights. Headings use `text-wrap: balance`. Font size never scales with viewport width.

## Spacing, layout and breakpoints

- Spacing is the Tailwind 4px scale. Corner radius token `--radius` is `0.5rem` (lg); md and sm are 2px and 4px smaller.
- Breakpoints are Tailwind defaults: `sm` 640px, `md` 768px, `lg` 1024px, `xl` 1280px. `sm` and `lg` carry nearly all layout changes.

App shell ([`AppLayout.tsx`](Frontend/src/components/layout/AppLayout.tsx)):

| Part       | Below 1024px                                                                | 1024px and up        |
| ---------- | --------------------------------------------------------------------------- | -------------------- |
| Navigation | Drawer (Radix Dialog), 256px, max 85vw, slides from left                    | Fixed sidebar, 240px |
| Top bar    | 56px, sticky, translucent with blur; menu button, scan search, account menu | Same, no menu button |
| Content    | Max width 1320px, side padding 16px (24px from `sm`)                        | Side padding 40px    |

Dashboard stats sit in a single bordered grid with 1px gaps (2 columns, 4 from `lg`) instead of separate floating cards.

## Components

Primitives live in [`components/ui/`](Frontend/src/components/ui/) (shadcn generated on Radix, then restyled); feature components in `components/<feature>/`.

| Primitive                  | Base                           | Variants and notes                                                                                                         |
| -------------------------- | ------------------------------ | -------------------------------------------------------------------------------------------------------------------------- |
| `Button`                   | `cva` + Radix Slot             | `default`, `destructive`, `outline`, `secondary`, `ghost`, `link`; sizes `default` 36px, `sm` 32px, `lg` 40px, `icon` 36px |
| `Segmented`                | Custom, `aria-pressed` buttons | Single choice for modes and views; mono uppercase labels                                                                   |
| `Card`                     | Custom                         | Header 48px min with bottom border; used for framed tools and settings groups                                              |
| `Dialog`                   | Radix Dialog                   | Modals with overlay, focus trap, Escape to close                                                                           |
| `DropdownMenu`             | Radix                          | Account menu                                                                                                               |
| `Tooltip`                  | Radix                          | Replaces `title` tooltips everywhere                                                                                       |
| `Slider`, `Switch`, `Tabs` | Radix                          | Viewer controls, settings toggles, scan detail tabs                                                                        |
| `Input`, `Label`           | Native input, restyled         | Text entry only                                                                                                            |
| `Table`                    | Native table, restyled         | Scan library and reports                                                                                                   |
| `Toaster`                  | sonner                         | All success and error notifications                                                                                        |

Feature components worth knowing: `DropZone` (custom drop area that opens the hidden file input from code), `ProcessingPipeline` (step list for detection, classification, segmentation), `ResultStatus` (status dot plus label), `ScanDialogs` (edit and delete), `PageHeader`, `BrandMark`, and the settings forms.

States:

| State         | Treatment                                                                                                                |
| ------------- | ------------------------------------------------------------------------------------------------------------------------ |
| Hover         | Accent surface or stronger border; text moves from muted to foreground                                                   |
| Focus-visible | 2px Cobalt ring with offset; outline removed                                                                             |
| Active        | Darker fill (`primary/80`, `accent/70`)                                                                                  |
| Selected      | Card surface with hairline shadow in `Segmented`; `aria-pressed` or `aria-checked`                                       |
| Disabled      | 45% opacity, no pointer events, `not-allowed` cursor                                                                     |
| Loading       | `animate-pulse` skeletons sized like the final content; spinner on app load; pending button labels such as "Deleting..." |
| Error         | Inline `role="alert"` text in Brick under the field or form; toasts for actions                                          |
| Success       | Toast with a sentence saying what changed                                                                                |
| Empty         | Message plus next step, for example "No scans found" with "Try adjusting your search or filters."                        |

Status tones ([`ResultStatus.tsx`](Frontend/src/components/scan/ResultStatus.tsx)): flagged = Ochre dot, clear = Moss, processing = pulsing Cobalt, failed = Brick, no result = muted. The same dots label the dashboard stats.

## Scan viewer

[`components/scan/viewer/`](Frontend/src/components/scan/viewer/) is the signature of the product: linked axial, coronal and sagittal slice views plus a vtk.js 3D view, in a 2 x 2 grid (axial and 3D on top). Any pane can be maximised.

- **Fixed dark viewport.** Panes are `#0a0a0a` with white text in both themes, because MRI is read on black.
- **Plane colours** tie each pane to its crosshair lines and its outline in 3D: axial `#0891b2` (cyan), coronal `#16a34a` (green), sagittal `#d946ef` (magenta). Measurements are `#fde047` (yellow).
- **Overlay colours** default to WT `#eab308` (amber), TC `#ef4444` (red), ET `#3b82f6` (blue). Users choose from 8 swatches in Settings; stored values are accepted only if they match `#rrggbb`.
- **Tools:** Crosshair, Pan, Zoom (drag), Contrast (drag), Ruler. Zoom is clamped to 0.5x to 16x.
- **Controls panel:** Contrast presets (Auto, Full range, Tight), Slab mode (Thin, MIP, Mean, MinIP, up to 40 slices), 3D presets (Tissue, Tumour when a mask exists, MIP) and a cut plane (Off, Axial, Coronal, Sagittal).
- **Keyboard**, on a focused slice pane: arrows step one slice, Page Up and Page Down step 10, Home and End jump to the ends, `+` or `=` and `-` zoom, `0` resets the view. On the 3D pane: arrows rotate by 10 degrees, `+` and `-` dolly, `0` resets the camera.
- A scale bar picks the longest round length (1 to 200 mm) that fits.

## Icons

One set: `lucide-react`, imported per icon (22 import sites). `index.css` sets `stroke-width: 1.75` on every Lucide icon to match the Plex letterforms. Button icons are 16px (14px in small buttons). The brand mark is a custom SVG: an axial head outline with a dashed midline.

## Motion

| Motion                    | Value                                                           | Purpose                              |
| ------------------------- | --------------------------------------------------------------- | ------------------------------------ |
| Page enter                | `fade-in`: 0.3s ease-out, opacity plus 4px rise, keyed by route | Signals a page change                |
| Colour and border changes | Tailwind default transition; 150ms on viewer panes              | Press and hover feedback             |
| Drawer and dialogs        | 200ms slide or fade (`tailwindcss-animate`)                     | Shows where the overlay came from    |
| Theme switch              | View Transitions API cross-fade where supported                 | Avoids every colour flipping at once |
| Progress                  | Pulsing Cobalt status dot                                       | Live processing                      |

`prefers-reduced-motion: reduce` cuts every animation and transition to 0.01ms (spinners keep turning slowly) and skips the theme cross-fade.

## Copy and terminology

UI strings for pages live in [`lib/config.ts`](Frontend/src/lib/config.ts).

| Term                                                     | Meaning                                                                           |
| -------------------------------------------------------- | --------------------------------------------------------------------------------- |
| Scan                                                     | One uploaded case (up to four modality files)                                     |
| Upload Scan, Scan Library, Reports, Settings             | Navigation names; page titles match                                               |
| FLAIR, T1, T1CE, T2                                      | Modalities, always upper case                                                     |
| Flagged / No Tumour                                      | Detection outcome; a flagged scan shows its class (glioma, meningioma, pituitary) |
| Whole tumour, Tumour core, Enhancing tumour (WT, TC, ET) | Segmentation regions                                                              |
| Processing, Failed, Retry                                | Pipeline states and the recovery action                                           |

Rules: British spelling ("tumour", "colour"), action-named buttons ("Delete Scan", "Open PDF", "Export CSV"), and errors that say what to do next.

## Accessibility

- Every icon-only button has an `aria-label`; viewer panes are labelled sections; tool buttons have tooltips.
- Radix provides focus traps, Escape handling and ARIA roles for dialogs, menus, tabs, switches and sliders. The theme picker is a `radiogroup`.
- Loading text uses `role="status"`, errors `role="alert"`.
- Theme is applied before first paint by [`public/theme-init.js`](Frontend/public/theme-init.js) (an external script, so the strict CSP needs no inline script), from saved choice or `prefers-color-scheme`.
- Autofill, placeholder, selection, caret and scrollbar colours are styled so browser defaults never show.

## Native UI policy

No `alert`, `confirm`, `prompt`, `<select>`, `<datalist>`, `<details>`, `<dialog>`, `<progress>`, `<meter>`, native date or colour pickers, or `title` tooltips exist in `src/` (searched; `title=` matches are component props). Every `<form>` has `noValidate` and shows inline errors. The search clear button is hidden. The only native controls are restyled text inputs and the hidden file input that `DropZone` opens from code.

## Decisions

| Decision                                          | Reason                                                                                 |
| ------------------------------------------------- | -------------------------------------------------------------------------------------- |
| Low-chroma Graphite neutrals with one accent      | Colour in the chrome would compete with greyscale MRI and the overlay                  |
| Neutral ink primary button, Cobalt for state only | Keeps "where am I / what is selected" distinct from "what can I do"                    |
| Clinical state colours (Moss, Ochre, Brick)       | Flagged is a warning, not an error; failure is the only red                            |
| IBM Plex Sans and Mono, bundled                   | Technical, legible at dense sizes; mono for aligned numbers; no font CDN under the CSP |
| Always-dark viewer panes                          | Radiology convention; consistent contrast across themes                                |
| Fixed per-plane colours                           | Lets users match crosshair lines and 3D outlines to their pane                         |
| Dense 13px tables and controls                    | Scan lists and viewer controls favour scanning many rows                               |
| Restyled shadcn on Radix instead of a new kit     | Accessible behaviour without hand-built focus and keyboard handling                    |

## Known gaps

Found while writing this document; not fixed in this pass.

- **Body size below 16px.** 13px and 12px text dominate dense views. Readable on desktop, but below the 16px body guideline for long reading on small screens.
- **Metadata copy.** `Frontend/index.html` title, description and Open Graph text use promotional phrases ("Advanced", "cutting-edge", "Empowering physicians") that the in-app copy avoids. The app is `noindex`, so this only shows in tab titles and link previews.
- **Visual verification.** No browser screenshots were taken for this document. The images in [`Materials/Screenshots/`](Materials/Screenshots/) show the current desktop UI; phone and tablet rendering was not checked in this pass.
