"""CSV and PDF exports of scan results."""

import csv
import io
from collections.abc import Iterable
from datetime import UTC, datetime

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from neuroscan.models import Scan

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


def scan_to_pdf(scan: Scan) -> bytes:
    styles = getSampleStyleSheet()
    buffer = io.BytesIO()
    document = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        leftMargin=18 * mm,
        rightMargin=18 * mm,
        topMargin=18 * mm,
        bottomMargin=18 * mm,
        title=f"NeuroScan AI report {scan.id}",
    )
    generated = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    rows = [
        [Paragraph(f"<b>{name}</b>", styles["BodyText"]), value]
        for name, value in zip(CSV_COLUMNS, _values(scan), strict=True)
    ]
    table = Table(rows, colWidths=[55 * mm, 115 * mm])
    table.setStyle(
        TableStyle(
            [
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#d4d4d8")),
                ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#f4f4f5")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    # Table cells given as plain strings are drawn as text, never parsed as markup.
    document.build(
        [
            Paragraph("NeuroScan AI Analysis Report", styles["Title"]),
            Paragraph(f"Generated {generated}", styles["Normal"]),
            Spacer(1, 8 * mm),
            table,
            Spacer(1, 8 * mm),
            Paragraph(DISCLAIMER, styles["Italic"]),
        ]
    )
    return buffer.getvalue()
