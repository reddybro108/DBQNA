from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.platypus import Table, TableStyle

def create_pdf(filename):
    # A4 size page
    width, height = A4
    c = canvas.Canvas(filename, pagesize=A4)

    # --- Page 1: Policy Overview ---
    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, height - 50, "Company Refund & Return Policies")

    c.setFont("Helvetica", 12)
    text = c.beginText(50, height - 80)
    text.textLines(
        """
        Our company values transparency and customer satisfaction. 
        This document outlines our refund, return, and cancellation policies 
        for all products and services purchased through our platform.
        
        Refund requests are processed within 7 business days after verification. 
        Partial refunds may be granted in special cases depending on product usage 
        and service consumption levels.
        """
    )
    c.drawText(text)

    # --- Bullet points ---
    text = c.beginText(50, height - 200)
    text.textLines([
        "Key Highlights:",
        "• Refund within 7 business days.",
        "• 24/7 customer support at support@example.com.",
        "• Policy updated quarterly."
    ])
    c.drawText(text)

    c.showPage()  # End of Page 1

    # --- Page 2: Financial Guidelines ---
    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, height - 50, "Financial & Payment Guidelines")

    c.setFont("Helvetica", 12)
    text = c.beginText(50, height - 80)
    text.textLines(
        """
        Payments can be made using credit cards, debit cards, or digital wallets. 
        All transactions are encrypted and securely processed.
        
        Late payment penalties:
        - 1% penalty per day after the due date.
        - Accounts suspended after 30 days of non-payment.
        """
    )
    c.drawText(text)

    # --- Table Example ---
    data = [
        ["Service", "Refund Window", "Penalty After Due Date"],
        ["Basic Plan", "7 days", "1% per day"],
        ["Premium Plan", "14 days", "1% per day"],
        ["Enterprise Plan", "30 days", "Custom contract"]
    ]
    table = Table(data, colWidths=[150, 150, 200])

    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('INNERGRID', (0, 0), (-1, -1), 0.25, colors.black),
        ('BOX', (0, 0), (-1, -1), 0.25, colors.black),
    ]))
    table.wrapOn(c, width, height)
    table.drawOn(c, 50, height - 300)

    c.showPage()  # End of Page 2

    # --- Page 3: Contact Info ---
    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, height - 50, "Contact and Support")

    c.setFont("Helvetica", 12)
    text = c.beginText(50, height - 80)
    text.textLines(
        """
        For any queries or escalations, please contact:
        
        Email: support@example.com
        Phone: +91 98765 43210
        Address: 123 Finance Street, Mumbai, India.
        
        Office Hours: 9 AM – 6 PM, Monday to Friday.
        """
    )
    c.drawText(text)

    c.showPage()
    c.save()

if __name__ == "__main__":
    create_pdf("sample.pdf")
    print("sample.pdf created with multiple pages, tables, and bullet points!")
