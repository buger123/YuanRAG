"""Build 7 test fixtures (.pdf .docx .pptx .xlsx .html .md .txt).

Each file contains a unique sentinel token (format-name + serial) so the
test driver can verify the LLM's answer referenced the right file
contents. Questions are kept tiny to minimize token cost.
"""
from __future__ import annotations

from pathlib import Path

from docx import Document as DocxDocument
from openpyxl import Workbook
from pptx import Presentation
from pptx.util import Inches, Pt

FIXTURES = Path(__file__).parent

# ---- .txt ----
(FIXTURES / "alpha.txt").write_text(
    "Project Alpha Notes\n"
    "====================\n"
    "The magic sentinel token for this file is: A1PHA-CODE-9988-A1PHA.\n"
    "Headquarters: Mountain View. Lead engineer: Dr. Eliza Stone.\n"
    "Launch date: 2026-09-14. The totem animal is the silver falcon.\n",
    encoding="utf-8",
)

# ---- .md ----
(FIXTURES / "bravo.md").write_text(
    "# Bravo Briefing\n\n"
    "**Sentinel token:** BR4VO-CODE-3322-BR4VO.\n\n"
    "## Highlights\n\n"
    "- Architect: *Wen Lu*, located in Hangzhou.\n"
    "- Stack: Rust + React + Postgres.\n"
    "- Mascot: golden retriever named Mango.\n\n"
    "## Status\n\nOn track for the November demo.\n",
    encoding="utf-8",
)

# ---- .html ----
(FIXTURES / "charlie.html").write_text(
    "<!doctype html>\n"
    '<html lang="en"><head><meta charset="utf-8">'
    "<title>Charlie Memo</title></head>\n"
    '<body><h1>Charlie Memo</h1>'
    "<p>Sentinel token for Charlie: <strong>CH4RL-CODE-7766-CH4RL</strong>.</p>"
    "<p>Owner: Renee Park, based in Toronto.</p>"
    "<p>Deadline: 2026-10-30. KPI target is 42 percent growth.</p>"
    "</body></html>\n",
    encoding="utf-8",
)

# ---- .docx ----
doc = DocxDocument()
doc.add_heading("Delta Dossier", level=1)
doc.add_paragraph(
    "The sentinel token for the Delta file is: DE1TA-CODE-4455-DE1TA."
)
doc.add_paragraph("Author: Marcus Reid, based in Berlin.")
doc.add_paragraph("Subtitle line: \"Ocean currents in a coffee cup.\"")
doc.save(str(FIXTURES / "delta.docx"))

# ---- .pptx ----
prs = Presentation()
slide = prs.slides.add_slide(prs.slide_layouts[0])
slide.shapes.title.text = "Echo Overview"
slide.placeholders[1].text = (
    "Sentinel token for Echo: ECH0-CODE-1199-ECH0. "
    "Point of contact: Yuki Sato, Singapore office."
)
slide2 = prs.slides.add_slide(prs.slide_layouts[1])
slide2.shapes.title.text = "Key numbers"
slide2.shapes.placeholders[1].text = "Target Q4 revenue: $1.2M."
prs.save(str(FIXTURES / "echo.pptx"))

# ---- .xlsx ----
wb = Workbook()
ws = wb.active
ws.title = "Foxtrot"
ws["A1"] = "Field"
ws["B1"] = "Value"
ws["A2"] = "sentinel_token"
ws["B2"] = "FOXTR-CODE-2244-FOXTR"
ws["A3"] = "owner"
ws["B3"] = "Aida Karim (Cairo)"
ws["A4"] = "budget_usd"
ws["B4"] = 75000
ws["A5"] = "deadline"
ws["B5"] = "2026-12-15"
ws["A6"] = "motto"
ws["B6"] = "slow and steady wins the race"
wb.save(str(FIXTURES / "foxtrot.xlsx"))

# ---- .pdf (hand-crafted minimal PDF, no library needed) ----
pdf_path = FIXTURES / "golf.pdf"
pdf_bytes = (
    b"%PDF-1.4\n"
    b"1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n"
    b"2 0 obj<</Type/Pages/Count 1/Kids[3 0 R]>>endobj\n"
    b"3 0 obj<</Type/Page/Parent 2 0 R/MediaBox[0 0 612 792]"
    b"/Contents 4 0 R/Resources<</Font<</F1 5 0 R>>>>>>endobj\n"
    b"4 0 obj<</Length 220>>stream\n"
    b"BT\n"
    b"/F1 18 Tf\n"
    b"72 720 Td\n"
    b"(Golf Field Report) Tj\n"
    b"0 -28 Td\n"
    b"/F1 12 Tf\n"
    b"(Sentinel token: G01F-CODE-3030-G01F) Tj\n"
    b"0 -16 Td\n"
    b"(Site lead: Olivia Chen, Auckland office.) Tj\n"
    b"0 -16 Td\n"
    b"(Final milestone: 2027-01-20.) Tj\n"
    b"ET\n"
    b"endstream\nendobj\n"
    b"5 0 obj<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>endobj\n"
    b"xref\n"
    b"0 6\n"
    b"0000000000 65535 f \n"
    b"0000000009 00000 n \n"
    b"0000000052 00000 n \n"
    b"0000000098 00000 n \n"
    b"0000000206 00000 n \n"
    b"0000000478 00000 n \n"
    b"trailer<</Size 6/Root 1 0 R>>\n"
    b"startxref\n"
    b"538\n"
    b"%%EOF\n"
)
pdf_path.write_bytes(pdf_bytes)

print("Fixtures written:")
for p in sorted(FIXTURES.iterdir()):
    if p.is_file() and p.suffix.lower() in {
        ".pdf", ".docx", ".pptx", ".xlsx", ".html", ".md", ".txt"
    }:
        print(f"  {p.name:20s} {p.stat().st_size:>7} bytes")
