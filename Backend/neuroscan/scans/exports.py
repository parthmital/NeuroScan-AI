"""CSV and PDF exports of scan results."""

import csv
import html
import io
from collections.abc import Iterable
from datetime import UTC, date, datetime
from functools import partial
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from reportlab.lib import colors
from reportlab.lib.enums import TA_RIGHT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfgen.canvas import Canvas
from reportlab.platypus import (
    CondPageBreak,
    Flowable,
    Image,
    KeepTogether,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from neuroscan.inference import volumes
from neuroscan.inference.processor import CLASSES, IMG_SIZE, SEG_SIZE, SEGMENTATION_CHANNELS
from neuroscan.models import Modality, Scan, User
from neuroscan.scans.pipeline import DETECTION_THRESHOLD, REPRESENTATIVE_ORDER
from neuroscan.scans.storage import ScanStorage

CSV_COLUMNS = [
    "Scan ID",
    "Patient ID",
    "Patient Name",
    "Scan Date",
    "Modalities",
    "Status",
    "Tumour Detected",
    "Classification",
    "Confidence (%)",
    "WT Volume (cm3)",
    "TC Volume (cm3)",
    "ET Volume (cm3)",
]
DISCLAIMER = (
    "This report was generated automatically by AI models and is intended to support, "
    "not replace, review by a qualified radiologist."
)
# Spreadsheet applications execute cells starting with these characters as formulas.
_FORMULA_PREFIXES = ("=", "+", "-", "@", "\t", "\r")


def _values(scan: Scan) -> list[str]:
    results = scan.results or {}
    detected = results.get("detected")
    confidence = results.get("confidence")
    values: list[object] = [
        scan.id,
        scan.patientId,
        scan.patientName,
        scan.scanDate,
        " ".join(scan.modalities or []),
        scan.status,
        "" if detected is None else ("Yes" if detected else "No"),
        results.get("classification"),
        f"{confidence * 100:.1f}" if isinstance(confidence, (int, float)) else "",
        results.get("wtVolume"),
        results.get("tcVolume"),
        results.get("etVolume"),
    ]
    return ["" if value is None else str(value) for value in values]


def _csv_cell(text: str) -> str:
    return f"'{text}" if text.startswith(_FORMULA_PREFIXES) else text


def scans_to_csv(scans: Iterable[Scan]) -> str:
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(CSV_COLUMNS)
    writer.writerows([_csv_cell(value) for value in _values(scan)] for scan in scans)
    return buffer.getvalue()


# PDF report: a formal, paginated MRI brain report in the layout of a radiology record.

_INK = colors.HexColor("#1a1d23")
_MUTED = colors.HexColor("#5b6472")
_NAVY = colors.HexColor("#1f3a5f")
_RULE = colors.HexColor("#c5ccd6")
_TINT = colors.HexColor("#f2f4f7")
_ALERT = colors.HexColor("#9b1c1c")
_ALERT_TINT = colors.HexColor("#fbf1f1")
_CLEAR = colors.HexColor("#1e5b3a")
_CLEAR_TINT = colors.HexColor("#eef6f1")

_PAGE_WIDTH, _PAGE_HEIGHT = A4
_MARGIN = 18 * mm
# SimpleDocTemplate insets its frame by 6 pt; the page margins subtract it back out.
_FRAME_PADDING = 6
_CONTENT_WIDTH = _PAGE_WIDTH - 2 * _MARGIN
_HEADER_HEIGHT = 22 * mm
_FOOTER_HEIGHT = 14 * mm

# Overlay colours match the viewer's defaults (Frontend/src/hooks/use-overlay-colors.ts), in BGR.
_REGION_COLOURS_BGR = {1: (8, 179, 234), 2: (68, 68, 239), 3: (246, 130, 59)}
_REGION_HEX = {1: "#eab308", 2: "#ef4444", 3: "#3b82f6"}
_OVERLAY_ALPHA = 0.45
_IMAGE_PIXELS = 512
# Separator for dimensions such as "240 x 240 x 155", with a real multiplication sign.
_BY = " \N{MULTIPLICATION SIGN} "

_SEQUENCE_NAMES = {
    Modality.FLAIR.value: "FLAIR",
    Modality.T1.value: "T1-weighted",
    Modality.T1CE.value: "T1-weighted post-contrast (T1CE)",
    Modality.T2.value: "T2-weighted",
}
_CLASS_NAMES = {
    "glioma": "Glioma",
    "meningioma": "Meningioma",
    "pituitary": "Pituitary tumour",
}
_REGIONS = (
    ("wtVolume", "Whole tumour (WT)", "Tumour core plus peritumoural oedema", 1),
    ("tcVolume", "Tumour core (TC)", "Enhancing tumour plus necrotic / non-enhancing core", 2),
    ("etVolume", "Enhancing tumour (ET)", "Contrast-enhancing tumour tissue", 3),
)


def _style(name: str, **overrides: Any) -> ParagraphStyle:
    base: dict[str, Any] = {
        "fontName": "Helvetica",
        "fontSize": 9,
        "leading": 12,
        "textColor": _INK,
    }
    return ParagraphStyle(name, **{**base, **overrides})


_STYLES = {
    "title": _style("title", fontName="Helvetica-Bold", fontSize=15, leading=19, textColor=_NAVY),
    "subtitle": _style("subtitle", fontSize=9.5, leading=13, textColor=_MUTED),
    "section": _style(
        "section", fontName="Helvetica-Bold", fontSize=9.5, leading=12, textColor=_NAVY
    ),
    "subsection": _style("subsection", fontName="Helvetica-Bold", fontSize=9, leading=12),
    "body": _style("body"),
    "small": _style("small", fontSize=8, leading=10.5, textColor=_MUTED),
    "label": _style("label", fontName="Helvetica-Bold", fontSize=7.5, leading=10, textColor=_MUTED),
    "value": _style("value", fontSize=9, leading=12),
    "cell": _style("cell", fontSize=8.5, leading=11),
    "head": _style("head", fontName="Helvetica-Bold", fontSize=8, leading=10, textColor=_NAVY),
    "caption": _style("caption", fontSize=7.5, leading=10, textColor=_MUTED),
    "status": _style(
        "status", fontName="Helvetica-Bold", fontSize=8, leading=10, alignment=TA_RIGHT
    ),
}


def _text(value: object) -> str:
    """User or file derived text, escaped so it is never parsed as Paragraph markup."""
    return html.escape(str(value))


def _para(markup: str, style: str = "body") -> Paragraph:
    return Paragraph(markup, _STYLES[style])


def _percent(value: object) -> str:
    return f"{float(value) * 100:.1f}%" if isinstance(value, (int, float)) else "Not recorded"


def _format_date(value: str) -> str:
    try:
        return date.fromisoformat(value).strftime("%d %b %Y")
    except ValueError:
        return value


def _format_timestamp(value: datetime) -> str:
    moment = value if value.tzinfo else value.replace(tzinfo=UTC)
    return moment.astimezone(UTC).strftime("%d %b %Y, %H:%M UTC")


def _section(title: str) -> list[Flowable]:
    """Section heading with a rule; asks for a new page rather than stranding the heading."""
    rule = Table([[""]], colWidths=[_CONTENT_WIDTH], rowHeights=[1])
    rule.setStyle(TableStyle([("LINEABOVE", (0, 0), (-1, -1), 0.8, _NAVY)]))
    return [CondPageBreak(40 * mm), Spacer(1, 5 * mm), _para(title.upper(), "section"), rule]


_NO_PADDING = [
    (side, (0, 0), (-1, -1), 0)
    for side in ("LEFTPADDING", "RIGHTPADDING", "TOPPADDING", "BOTTOMPADDING")
]


_CELLS = [
    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ("TOPPADDING", (0, 0), (-1, -1), 4),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ("LEFTPADDING", (0, 0), (-1, -1), 5),
    ("RIGHTPADDING", (0, 0), (-1, -1), 5),
]


def _table(
    header: list[str], rows: list[list[str]], widths: list[float], highlight: int | None = None
) -> Table:
    """Grid table. Cells are markup built from escaped values."""
    data = [[_para(cell, "head") for cell in header]]
    data += [[_para(cell, "cell") for cell in row] for row in rows]
    table = Table(data, colWidths=widths, repeatRows=1, hAlign="LEFT")
    style = TableStyle(
        [
            ("GRID", (0, 0), (-1, -1), 0.4, _RULE),
            ("BACKGROUND", (0, 0), (-1, 0), _TINT),
            *_CELLS,
        ]
    )
    if highlight is not None:
        style.add("BACKGROUND", (0, highlight + 1), (-1, highlight + 1), _ALERT_TINT)
    table.setStyle(style)
    return table


def _field_grid(pairs: list[tuple[str, str]]) -> Table:
    """Two columns of label-over-value fields, as on a patient demographics panel."""
    cells = [[_para(label.upper(), "label"), _para(value, "value")] for label, value in pairs]
    rows = []
    for index in range(0, len(cells), 2):
        left = cells[index]
        right = cells[index + 1] if index + 1 < len(cells) else ["", ""]
        rows.append([left[0], left[1], right[0], right[1]])
    width = _CONTENT_WIDTH / 2
    table = Table(rows, colWidths=[30 * mm, width - 30 * mm, 30 * mm, width - 30 * mm])
    table.setStyle(
        TableStyle(
            [
                ("BOX", (0, 0), (-1, -1), 0.6, _RULE),
                ("LINEBELOW", (0, 0), (-1, -2), 0.3, _RULE),
                ("LINEAFTER", (1, 0), (1, -1), 0.3, _RULE),
                ("BACKGROUND", (0, 0), (0, -1), _TINT),
                ("BACKGROUND", (2, 0), (2, -1), _TINT),
                *_CELLS,
            ]
        )
    )
    return table


def _callout(lines: list[str], alert: bool) -> Table:
    """Impression box with a coloured bar on the left edge."""
    table = Table([[_para(line)] for line in lines], colWidths=[_CONTENT_WIDTH])
    accent, tint = (_ALERT, _ALERT_TINT) if alert else (_CLEAR, _CLEAR_TINT)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), tint),
                ("LINEBEFORE", (0, 0), (0, -1), 3, accent),
                ("LEFTPADDING", (0, 0), (-1, -1), 9),
                ("TOPPADDING", (0, 0), (-1, 0), 7),
                ("BOTTOMPADDING", (0, -1), (-1, -1), 7),
                ("TOPPADDING", (0, 1), (-1, -1), 2),
                ("BOTTOMPADDING", (0, 0), (-1, -2), 2),
            ]
        )
    )
    return table


class _NumberedCanvas(Canvas):  # type: ignore[misc]  # reportlab ships no type hints
    """Draws the letterhead and footer on every page once the total page count is known."""

    def __init__(self, *args: Any, heading: tuple[str, str], footer: str, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._heading = heading
        self._footer = footer
        self._pages: list[dict[str, Any]] = []

    def showPage(self) -> None:
        self._pages.append(dict(self.__dict__))
        self._startPage()

    def save(self) -> None:
        total = len(self._pages)
        for state in self._pages:
            self.__dict__.update(state)
            self._decorate(total)
            super().showPage()
        super().save()

    def _decorate(self, total: int) -> None:
        organisation, accession = self._heading
        top = _PAGE_HEIGHT - _MARGIN
        self.setFillColor(_NAVY)
        self.setFont("Helvetica-Bold", 11)
        self.drawString(_MARGIN, top - 4 * mm, "NeuroScan AI")
        self.setFont("Helvetica", 8)
        self.setFillColor(_MUTED)
        self.drawString(_MARGIN, top - 8.5 * mm, organisation)
        self.setFont("Helvetica-Bold", 8.5)
        self.setFillColor(_INK)
        self.drawRightString(_PAGE_WIDTH - _MARGIN, top - 4 * mm, "MRI BRAIN REPORT")
        self.setFont("Helvetica", 8)
        self.setFillColor(_MUTED)
        self.drawRightString(_PAGE_WIDTH - _MARGIN, top - 8.5 * mm, f"Accession no. {accession}")
        self.setStrokeColor(_NAVY)
        self.setLineWidth(1.2)
        self.line(_MARGIN, top - 11.5 * mm, _PAGE_WIDTH - _MARGIN, top - 11.5 * mm)

        bottom = _MARGIN
        self.setStrokeColor(_RULE)
        self.setLineWidth(0.5)
        self.line(_MARGIN, bottom + 6 * mm, _PAGE_WIDTH - _MARGIN, bottom + 6 * mm)
        self.setFont("Helvetica", 7)
        self.setFillColor(_MUTED)
        self.drawString(
            _MARGIN, bottom + 2 * mm, "CONFIDENTIAL: contains patient health information."
        )
        self.drawCentredString(_PAGE_WIDTH / 2, bottom + 2 * mm, self._footer)
        self.drawRightString(
            _PAGE_WIDTH - _MARGIN, bottom + 2 * mm, f"Page {self._pageNumber} of {total}"
        )


def _png(image_bgr: np.ndarray, width: float) -> Image:
    upscaled = cv2.resize(image_bgr, (_IMAGE_PIXELS, _IMAGE_PIXELS), interpolation=cv2.INTER_CUBIC)
    ok, buffer = cv2.imencode(".png", upscaled)
    if not ok:
        raise ValueError("Could not encode report image")
    return Image(io.BytesIO(buffer.tobytes()), width=width, height=width)


def _overlay(base_bgr: np.ndarray, labels: np.ndarray) -> np.ndarray:
    coloured = base_bgr.copy()
    for label, colour in _REGION_COLOURS_BGR.items():
        coloured[labels == label] = colour
    blended = cv2.addWeighted(coloured, _OVERLAY_ALPHA, base_bgr, 1 - _OVERLAY_ALPHA, 0)
    result: np.ndarray = np.where((labels > 0)[..., None], blended, base_bgr)
    return result


def _sequence_rows(paths: dict[str, Path]) -> list[list[str]]:
    rows = []
    for modality in Modality:
        name = _SEQUENCE_NAMES[modality.value]
        path = paths.get(modality.value)
        if path is None or not path.is_file():
            rows.append([name, "Not provided", "-", "-", "-"])
            continue
        shape, spacing = volumes.geometry(path)
        rows.append(
            [
                name,
                "Received",
                _BY.join(str(d) for d in shape),
                _BY.join(f"{s:.2f}" for s in spacing),
                str(shape[2]),
            ]
        )
    return rows


def _impression(results: dict[str, Any]) -> tuple[list[str], bool]:
    detected = bool(results.get("detected"))
    probability = results.get("detectionProbability")
    if not detected:
        lines = [
            "<b>No tumour detected</b> on the analysed axial slice "
            f"(model confidence {_percent(results.get('confidence'))}).",
            "No further AI characterisation was performed.",
        ]
        return lines, False
    label = _CLASS_NAMES.get(str(results.get("classification")), _text(results["classification"]))
    detection = f" (tumour probability {_percent(probability)})" if probability is not None else ""
    lines = [
        f"<b>Findings suggestive of an intracranial tumour</b>{detection}.",
        f"Most likely tumour type: <b>{label}</b> "
        f"(classification confidence {_percent(results.get('confidence'))}).",
    ]
    wt = results.get("wtVolume")
    if isinstance(wt, (int, float)) and wt > 0:
        lines.append(
            f"Estimated volumes: whole tumour {wt:.2f} cm³, tumour core "
            f"{results.get('tcVolume', 0):.2f} cm³, enhancing tumour "
            f"{results.get('etVolume', 0):.2f} cm³."
        )
    lines.append("Correlation with clinical history and radiologist review is recommended.")
    return lines, True


def _detection_findings(results: dict[str, Any], analysed: str) -> list[Flowable]:
    probability = results.get("detectionProbability")
    rows = [
        ["Result", "<b>Tumour detected</b>" if results.get("detected") else "No tumour detected"],
        [
            "Tumour probability",
            _percent(probability) if probability is not None else "Not recorded for this scan",
        ],
        ["Decision threshold", f"{DETECTION_THRESHOLD * 100:.0f}%"],
        ["Input analysed", analysed],
    ]
    return [
        _para("Tumour detection", "subsection"),
        Spacer(1, 1.5 * mm),
        _table(["Parameter", "Value"], rows, [50 * mm, _CONTENT_WIDTH - 50 * mm]),
    ]


def _classification_findings(results: dict[str, Any]) -> list[Flowable]:
    heading: list[Flowable] = [Spacer(1, 4 * mm), _para("Tumour classification", "subsection")]
    if not results.get("detected"):
        return [*heading, _para("Not applicable: no tumour was detected.", "small")]
    predicted = str(results.get("classification"))
    probabilities = results.get("classProbabilities")
    if not isinstance(probabilities, dict) or not probabilities:
        probabilities = {predicted: results.get("confidence")}
    ordered = [name for name in CLASSES if name in probabilities]
    rows = [
        [
            _CLASS_NAMES.get(name, _text(name)),
            _percent(probabilities[name]),
            "<b>Predicted</b>" if name == predicted else "",
        ]
        for name in ordered
    ]
    highlight = ordered.index(predicted) if predicted in ordered else None
    widths = [70 * mm, 40 * mm, _CONTENT_WIDTH - 110 * mm]
    return [
        *heading,
        Spacer(1, 1.5 * mm),
        _table(["Tumour type", "Probability", "Model output"], rows, widths, highlight),
    ]


def _segmentation_findings(
    results: dict[str, Any], extent: tuple[int, int, int] | None
) -> list[Flowable]:
    heading: list[Flowable] = [
        Spacer(1, 4 * mm),
        _para("Volumetric segmentation", "subsection"),
    ]
    wt = results.get("wtVolume")
    if extent is None and not (isinstance(wt, (int, float)) and wt > 0):
        note = (
            "Not performed or no tumour voxels segmented. Segmentation requires all four "
            "sequences (FLAIR, T1, T1CE, T2) and the segmentation model."
        )
        return [*heading, _para(note, "small")]
    rows = []
    for key, name, definition, label in _REGIONS:
        volume = results.get(key)
        share = (
            f"{volume / wt * 100:.0f}%"
            if isinstance(volume, (int, float)) and isinstance(wt, (int, float)) and wt > 0
            else "-"
        )
        swatch = _swatch(label)
        rows.append(
            [
                f"{swatch} {name}",
                definition,
                f"{volume:.2f}" if isinstance(volume, (int, float)) else "-",
                share,
            ]
        )
    widths = [44 * mm, _CONTENT_WIDTH - 44 * mm - 48 * mm, 24 * mm, 24 * mm]
    flowables: list[Flowable] = [
        *heading,
        Spacer(1, 1.5 * mm),
        _table(["Region", "Definition", "Volume (cm³)", "Share of WT"], rows, widths),
    ]
    if extent is not None:
        first, last, total = extent
        flowables += [
            Spacer(1, 1.5 * mm),
            _para(
                f"Axial extent of segmented tumour: slices {first + 1} to {last + 1} of {total} "
                f"({last - first + 1} slices).",
                "small",
            ),
        ]
    return flowables


def _swatch(label: int) -> str:
    """Filled square in a region's colour (ZapfDingbats "n"; Helvetica has no square glyph)."""
    return f'<font name="ZapfDingbats" color="{_REGION_HEX[label]}">n</font>'


def _figure(image: Image, caption: str, width: float) -> Table:
    table = Table([[image], [_para(caption, "caption")]], colWidths=[width])
    table.setStyle(
        TableStyle(
            [
                ("BOX", (0, 0), (0, 0), 0.5, _RULE),
                ("BACKGROUND", (0, 0), (0, 0), colors.black),
                ("ALIGN", (0, 0), (0, 0), "CENTER"),
                ("LEFTPADDING", (0, 0), (-1, -1), 0),
                ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                ("TOPPADDING", (0, 0), (0, 0), 0),
                ("BOTTOMPADDING", (0, 0), (0, 0), 0),
                ("TOPPADDING", (0, 1), (0, 1), 3),
            ]
        )
    )
    return table


def _key_images(
    paths: dict[str, Path], mask: np.ndarray | None, representative: str
) -> tuple[list[Flowable], tuple[int, int, int] | None]:
    """Key image row plus the tumour's axial extent (first slice, last slice, total slices)."""
    gap = 4 * mm
    width = (_CONTENT_WIDTH - 2 * gap) / 3
    name = _SEQUENCE_NAMES[representative]
    middle = volumes.slice_count(paths[representative]) // 2
    image, total = volumes.render_slice(paths[representative], middle)
    figures = [
        _figure(
            _png(image, width),
            f"<b>A.</b> {name}, axial slice {middle + 1} of {total}. "
            "Slice analysed by the detection and classification models.",
            width,
        )
    ]
    extent = None
    flair = paths.get(Modality.FLAIR.value)
    if mask is not None and flair is not None:
        area = (mask > 0).sum(axis=(0, 1))
        occupied = np.flatnonzero(area)
        if occupied.size:
            peak = int(np.argmax(area))
            extent = (int(occupied[0]), int(occupied[-1]), int(mask.shape[2]))
            base, flair_total = volumes.render_slice(flair, peak)
            labels = np.rot90(mask[:, :, peak], 1)
            caption = (
                f"FLAIR, axial slice {peak + 1} of {flair_total} (largest tumour cross-section)"
            )
            figures += [
                _figure(_png(base, width), f"<b>B.</b> {caption}.", width),
                _figure(
                    _png(_overlay(base, labels), width),
                    f"<b>C.</b> Same slice with segmentation overlay: {_swatch(1)} WT "
                    f"{_swatch(2)} TC {_swatch(3)} ET.",
                    width,
                ),
            ]
    cells: list[Any] = [*figures, *([""] * (3 - len(figures)))]
    row = Table(
        [[cells[0], "", cells[1], "", cells[2]]],
        colWidths=[width, gap, width, gap, width],
        hAlign="LEFT",
    )
    row.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), *_NO_PADDING]))
    return [row], extent


def _methodology_rows() -> list[list[str]]:
    size = _BY.join(str(d) for d in SEG_SIZE)
    return [
        [
            "Detection",
            "EfficientNet-B0, binary",
            "Middle axial slice of the preferred sequence (FLAIR, then T1CE, T2, T1), "
            f"cropped to the brain and resized to {_BY.join(map(str, IMG_SIZE))}",
            f"Tumour probability; positive above {DETECTION_THRESHOLD * 100:.0f}%",
        ],
        [
            "Classification",
            "EfficientNet-B0, 3 classes",
            "Same slice as detection",
            "Probability for glioma, meningioma and pituitary tumour",
        ],
        [
            "Segmentation",
            "3D V-Net",
            f"{', '.join(m.upper() for m in SEGMENTATION_CHANNELS)} volumes, z-score normalised "
            f"and resampled to {size}",
            "WT, TC and ET masks (sigmoid above 0.5), resampled to FLAIR space",
        ],
    ]


def _signature_block() -> Table:
    labels = ["REVIEWING RADIOLOGIST", "SIGNATURE", "DATE"]
    gap = 8 * mm
    widths = [62 * mm, gap, 62 * mm, gap, _CONTENT_WIDTH - 124 * mm - 2 * gap]
    table = Table(
        [
            ["", "", "", "", ""],
            [
                _para(labels[0], "label"),
                "",
                _para(labels[1], "label"),
                "",
                _para(labels[2], "label"),
            ],
        ],
        colWidths=widths,
        rowHeights=[12 * mm, None],
        hAlign="LEFT",
    )
    table.setStyle(
        TableStyle(
            [
                *(("LINEBELOW", (col, 0), (col, 0), 0.6, _INK) for col in (0, 2, 4)),
                ("LEFTPADDING", (0, 0), (-1, -1), 0),
                ("TOPPADDING", (0, 1), (-1, 1), 3),
            ]
        )
    )
    return table


def _requested_by(user: User) -> str:
    name = _text(user.fullName)
    return f"{name}, {_text(user.title)}" if user.title else name


def _organisation(user: User) -> str:
    parts = [part for part in (user.department, user.institution) if part]
    return ", ".join(parts) if parts else "AI-assisted neuroimaging analysis"


def scan_to_pdf(scan: Scan, user: User, storage: ScanStorage) -> bytes:
    results: dict[str, Any] = dict(scan.results or {})
    paths = {
        modality: path
        for modality, path in storage.modality_paths(scan.filePaths or {}).items()
        if path.is_file()
    }
    mask_path = storage.mask_path(scan.id)
    mask = volumes.load_first_volume(mask_path)[0] if mask_path.is_file() else None
    representative = next((m for m in REPRESENTATIVE_ORDER if m in paths), None)
    generated = _format_timestamp(datetime.now(UTC))

    buffer = io.BytesIO()
    document = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        leftMargin=_MARGIN - _FRAME_PADDING,
        rightMargin=_MARGIN - _FRAME_PADDING,
        topMargin=_MARGIN + _HEADER_HEIGHT - 6 * mm,
        bottomMargin=_MARGIN + _FOOTER_HEIGHT - 4 * mm,
        title=f"MRI brain report {scan.id}",
        author="NeuroScan AI",
        subject="AI-assisted MRI brain tumour analysis",
    )

    sequences = ", ".join(m.upper() for m in scan.modalities or []) or "None"
    status_line = _para(
        '<font color="#9b1c1c">PRELIMINARY</font><br/>'
        '<font name="Helvetica" color="#5b6472">AI generated, pending radiologist review</font>',
        "status",
    )
    title_row = Table(
        [
            [
                [
                    _para("Magnetic Resonance Imaging: Brain", "title"),
                    _para(
                        "AI-assisted tumour detection, classification and segmentation", "subtitle"
                    ),
                ],
                status_line,
            ]
        ],
        colWidths=[_CONTENT_WIDTH - 60 * mm, 60 * mm],
    )
    title_row.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), *_NO_PADDING]))

    story: list[Flowable] = [
        title_row,
        Spacer(1, 5 * mm),
        _field_grid(
            [
                ("Patient name", f"<b>{_text(scan.patientName)}</b>"),
                ("Patient ID", _text(scan.patientId)),
                ("Study date", _text(_format_date(scan.scanDate))),
                ("Accession no.", _text(scan.id)),
                ("Examination", "MRI brain, multi-parametric"),
                ("Sequences", _text(sequences)),
                ("Received", _format_timestamp(scan.createdAt)),
                ("Report issued", generated),
                ("Requested by", _requested_by(user)),
                ("Institution", _text(user.institution or "Not recorded")),
            ]
        ),
        *_section("Impression"),
        Spacer(1, 2 * mm),
        _callout(*_impression(results)),
        *_section("Technique"),
        Spacer(1, 2 * mm),
        _para(
            "Multi-parametric MRI of the brain was submitted as NIfTI volumes and analysed "
            "automatically by the NeuroScan AI pipeline. Matrix size and voxel spacing are "
            "taken from the file headers."
        ),
        Spacer(1, 2 * mm),
        _table(
            ["Sequence", "Status", "Matrix (voxels)", "Voxel size (mm)", "Axial slices"],
            _sequence_rows(paths),
            [58 * mm, 24 * mm, 34 * mm, 34 * mm, _CONTENT_WIDTH - 150 * mm],
        ),
        *_section("Findings"),
        Spacer(1, 2 * mm),
    ]

    extent = None
    images: list[Flowable] = []
    if representative is not None:
        images, extent = _key_images(paths, mask, representative)
        middle = volumes.slice_count(paths[representative]) // 2
        analysed = f"{_SEQUENCE_NAMES[representative]}, axial slice {middle + 1}"
    else:
        analysed = "Source files are no longer available"
    story += [
        KeepTogether(_detection_findings(results, analysed)),
        KeepTogether(_classification_findings(results)),
        KeepTogether(_segmentation_findings(results, extent)),
    ]
    if images:
        story += [*_section("Key images"), Spacer(1, 3 * mm), *images]

    story += [
        *_section("Methodology"),
        Spacer(1, 2 * mm),
        _table(
            ["Stage", "Model", "Input", "Output"],
            _methodology_rows(),
            [26 * mm, 36 * mm, 62 * mm, _CONTENT_WIDTH - 124 * mm],
        ),
        Spacer(1, 2 * mm),
        _para(
            "Volumes are voxel counts on the resampled segmentation grid divided by 1000. They "
            "equal cm³ only when that grid has 1 mm isotropic spacing, so treat them as estimates "
            "for comparison between studies processed the same way.",
            "small",
        ),
        *_section("Limitations"),
        Spacer(1, 2 * mm),
        _para(
            "Detection and classification use a single axial slice, so lesions outside that "
            "slice can be missed. The classifier recognises only glioma, meningioma and "
            "pituitary tumours; other pathologies will be assigned to the closest of these "
            "classes. Model outputs are probabilities, not diagnoses. "
            f"{DISCLAIMER}"
        ),
        KeepTogether(
            [
                *_section("Verification"),
                Spacer(1, 2 * mm),
                _para(
                    "I have reviewed the images and the AI findings above. Amendments, if any, "
                    "are recorded below.",
                ),
                Spacer(1, 2 * mm),
                _signature_block(),
                Spacer(1, 6 * mm),
                _para(
                    f"Electronically generated by NeuroScan AI on {generated} for "
                    f"{_requested_by(user)}.",
                    "small",
                ),
            ]
        ),
    ]

    canvas_maker = partial(
        _NumberedCanvas,
        heading=(_organisation(user), scan.id),
        footer=f"Report issued {generated}",
    )
    document.build(story, canvasmaker=canvas_maker)
    return buffer.getvalue()
