"""Generate final PDF report."""

import os
from pathlib import Path
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

def main():
    root = Path("d:/Projects/MultiMediaForensics/Spectral_Forensics")
    md_path = root / "docs" / "final_project_report.md"
    pdf_dir = root / "reports"
    pdf_dir.mkdir(exist_ok=True)
    pdf_path = pdf_dir / "final_project_report.pdf"
    
    with open(md_path, "r", encoding="utf-8") as f:
        content = f.read()
        
    doc = SimpleDocTemplate(str(pdf_path), pagesize=letter,
                            rightMargin=72, leftMargin=72,
                            topMargin=72, bottomMargin=18)
                            
    styles = getSampleStyleSheet()
    h1 = styles['Heading1']
    h2 = styles['Heading2']
    normal = styles['Normal']
    
    Story = []
    
    for line in content.split('\n'):
        line = line.strip()
        if not line:
            Story.append(Spacer(1, 12))
        elif line.startswith('# '):
            Story.append(Paragraph(line[2:], h1))
        elif line.startswith('## '):
            Story.append(Paragraph(line[3:], h2))
        elif line.startswith('- '):
            Story.append(Paragraph("&bull; " + line[2:], normal))
        else:
            Story.append(Paragraph(line, normal))
            
    doc.build(Story)
    print(f"Generated PDF at {pdf_path}")

if __name__ == "__main__":
    main()
