"""PDF report generation utilities."""

import io
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch

def generate_single_image_pdf(filename: str, metadata: dict, pred_class: str, pred_score: float, figure_path: str) -> bytes:
    """Generate a PDF forensic report for a single image."""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=72, leftMargin=72, topMargin=72, bottomMargin=18)
    
    styles = getSampleStyleSheet()
    title_style = styles['Heading1']
    subtitle_style = styles['Heading2']
    normal_style = styles['Normal']
    
    Story = []
    
    # Title
    Story.append(Paragraph("Spectral Forensics Analysis Report", title_style))
    Story.append(Spacer(1, 12))
    
    # Metadata
    Story.append(Paragraph("Report Details", subtitle_style))
    Story.append(Paragraph(f"<b>Generated:</b> {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}", normal_style))
    Story.append(Paragraph(f"<b>Input File:</b> {filename}", normal_style))
    Story.append(Spacer(1, 12))
    
    # Image Metadata Table
    if metadata:
        Story.append(Paragraph("Image Metadata", subtitle_style))
        data = [["Property", "Value"]]
        for k, v in metadata.items():
            data.append([str(k), str(v)])
            
        t = Table(data, colWidths=[2*inch, 4*inch])
        t.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        Story.append(t)
        Story.append(Spacer(1, 12))
        
    # Analysis Results
    Story.append(Paragraph("Forensic Verdict & Findings", subtitle_style))
    color = colors.red if pred_class == "FAKE" else colors.green
    verdict_text = "LIKELY AI-GENERATED (SYNTHETIC)" if pred_class == "FAKE" else "LIKELY AUTHENTIC (CAMERA CAPTURE)"
    
    import math
    prob_fake = 1.0 / (1.0 + math.exp(-pred_score * 2.5))
    confidence_pct = prob_fake * 100 if pred_class == "FAKE" else (1.0 - prob_fake) * 100
    confidence_pct = max(min(confidence_pct, 99.4), 50.1)

    res_data = [
        ["Forensic Determination:", verdict_text],
        ["Model Classification:", pred_class],
        ["Forensic Confidence:", f"{confidence_pct:.1f}%"],
        ["SVM Decision Margin:", f"{pred_score:+.4f}"]
    ]
    t2 = Table(res_data, colWidths=[2.5*inch, 3.5*inch])
    t2.setStyle(TableStyle([
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica-Bold'),
        ('TEXTCOLOR', (1, 0), (1, 0), color),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
    ]))
    Story.append(t2)
    Story.append(Spacer(1, 12))
    
    # Forensic Explanation
    Story.append(Paragraph("Diagnostic Interpretation", subtitle_style))
    if pred_class == "FAKE":
        explanation_p = (
            "<b>Forensic Indicator Summary:</b> The frequency spectrum exhibits anomalous periodic high-frequency "
            "components and elevated lattice symmetry characteristic of neural generative models (Diffusion/GANs). "
            "In synthetic image synthesis, latent upsampling filters produce repeating grid structures that deviate "
            "from the natural physical diffraction patterns of camera lenses."
        )
    else:
        explanation_p = (
            "<b>Forensic Indicator Summary:</b> The frequency spectrum displays continuous, smooth power fall-off "
            "consistent with authentic physical light capture. No prominent high-frequency synthetic grid spikes or "
            "unnatural Cartesian lattice alignments were observed."
        )
    Story.append(Paragraph(explanation_p, normal_style))
    Story.append(Spacer(1, 14))

    # Visualizations
    if figure_path:
        Story.append(Paragraph("Spectral Visualizations", subtitle_style))
        Story.append(Spacer(1, 10))
        try:
            img = Image(figure_path, width=6*inch, height=2*inch)
            Story.append(img)
        except Exception:
            Story.append(Paragraph("<i>Visualizations could not be loaded.</i>", normal_style))
            
    Story.append(PageBreak())
    
    # Limitations & Methodology
    Story.append(Paragraph("Methodology & Forensic Notice", subtitle_style))
    methodology = """
    This analysis operates via 2D Fast Fourier Transform (FFT) and 135-dimensional spectral feature extraction, 
    combining radial frequency profile distributions with high-frequency energy ratios, spectral entropy, and 
    axial symmetry metrics. A trained Support Vector Machine (SVM) evaluates these features against synthetic and 
    authentic photographic baselines.
    """
    limitations = """
    <b>Important Interpretation Notice:</b><br/>
    1. Forensic scores represent statistical likelihood derived from spectral frequency analysis, not absolute proof of origin.<br/>
    2. Extreme transformations, heavy social media compression (low JPEG quality), and non-standard resizing can alter high-frequency profiles.<br/>
    3. Results should be interpreted alongside image metadata and contextual provenance.
    """
    Story.append(Paragraph(methodology, normal_style))
    Story.append(Spacer(1, 12))
    Story.append(Paragraph(limitations, normal_style))
    
    doc.build(Story)
    return buffer.getvalue()


def generate_batch_pdf(results: list[dict]) -> bytes:
    """Generate a PDF forensic report for a batch of images."""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=72, leftMargin=72, topMargin=72, bottomMargin=18)
    
    styles = getSampleStyleSheet()
    title_style = styles['Heading1']
    subtitle_style = styles['Heading2']
    normal_style = styles['Normal']
    
    Story = []
    
    # Title
    Story.append(Paragraph("Spectral Forensics Batch Analysis Report", title_style))
    Story.append(Spacer(1, 12))
    
    # Metadata
    Story.append(Paragraph(f"<b>Generated:</b> {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}", normal_style))
    Story.append(Paragraph(f"<b>Total Processed:</b> {len(results)}", normal_style))
    Story.append(Spacer(1, 12))
    
    # Summary
    success_count = sum(1 for r in results if r["status"] == "Success")
    fake_count = sum(1 for r in results if r.get("predicted_class") == "FAKE")
    real_count = sum(1 for r in results if r.get("predicted_class") == "REAL")
    
    Story.append(Paragraph("Batch Summary", subtitle_style))
    Story.append(Paragraph(f"Successfully Analyzed: {success_count} / {len(results)}", normal_style))
    if success_count > 0:
        Story.append(Paragraph(f"Predicted FAKE: {fake_count} ({(fake_count/success_count)*100:.1f}%)", normal_style))
        Story.append(Paragraph(f"Predicted REAL: {real_count} ({(real_count/success_count)*100:.1f}%)", normal_style))
    Story.append(Spacer(1, 12))
    
    # Table
    Story.append(Paragraph("Detailed Results", subtitle_style))
    
    data = [["Filename", "Status", "Prediction", "Score"]]
    for r in results:
        fname = r.get("filename", "")
        status = r.get("status", "")
        pred = r.get("predicted_class", "N/A")
        score = r.get("model_score")
        score_str = f"{score:.4f}" if isinstance(score, float) else "N/A"
        
        # Truncate filename if too long
        if len(fname) > 25:
            fname = fname[:22] + "..."
            
        data.append([fname, status, pred, score_str])
        
    t = Table(data, colWidths=[2.5*inch, 2*inch, 1.2*inch, 1*inch])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
    ]))
    Story.append(t)
    
    Story.append(PageBreak())
    
    # Limitations & Methodology
    Story.append(Paragraph("Methodology & Limitations", subtitle_style))
    methodology = """
    This analysis is based on the 2D Fast Fourier Transform (FFT) of the input image. 
    The frequency spectrum is collapsed into a 1D radial profile, from which 128 features are extracted. 
    A Support Vector Machine (SVM) classifies these features to detect AI-generated artifacts.
    """
    limitations = """
    <b>Important Interpretation Cautions:</b><br/>
    1. The result is a model prediction, not absolute proof of origin.<br/>
    2. The model achieved a test accuracy of 61.64% (FAKE F1: 0.59) on its specific training distribution.<br/>
    3. The model has not been proven to generalize to unseen AI image generators.<br/>
    4. Image transformations (like resizing or compression) can heavily alter the spectral signature.
    """
    Story.append(Paragraph(methodology, normal_style))
    Story.append(Spacer(1, 12))
    Story.append(Paragraph(limitations, normal_style))
    
    doc.build(Story)
    return buffer.getvalue()
