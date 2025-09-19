from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.platypus import Table, TableStyle
from reportlab.lib import colors
import random

topics = [
    "Refund & Return Policy",
    "Credit Card Guidelines",
    "Insurance Claims Procedure",
    "Loan Repayment Policy",
    "Account Security Practices",
    "Customer Data Privacy",
    "Investment Advisory",
    "Fraud Detection Measures",
    "Digital Wallet Usage",
    "Customer Support Escalation"
]

def create_pdf(filename, title, index):
    width, height = A4
    c = canvas.Canvas(filename, pagesize=A4)

    # --- Title ---
    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, height - 50, f"{title}")

    # --- Paragraph ---
    c.setFont("Helvetica", 12)
    text = c.beginText(50, height - 80)
    text.textLines(
        f"""
        Document #{index+1} – {title}

        This policy describes the process for {title.lower()}.
        All steps are compliant with the Financial Regulatory Authority (FRA).
        
        Customers must adhere to these policies to ensure smooth transactions.
        Violations may result in penalties, suspension, or legal action.
        """
    )
    c.drawText(text)

    # --- Bullet points ---
    penalties = random.randint(1,5)
    text = c.beginText(50, height - 200)
    text.textLines([
        "Key Highlights:",
        f"• Standard processing time: {random.randint(3,15)} days",
        f"• Penalty after due date: {penalties}% per day",
        "• Dedicated support via support@example.com"
    ])
    c.drawText(text)

    # --- Table ---
    data = [
        ["Service Plan", "Refund Window", "Penalty After Due Date"],
        ["Basic Plan", f"{random.randint(5,10)} days", f"{penalties}% per day"],
        ["Premium Plan", f"{random.randint(10,20)} days", f"{penalties}% per day"],
        ["Enterprise Plan", "Custom contract", "Negotiated terms"]
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
    table.drawOn(c, 50, height - 350)

    # --- Contact Info ---
    c.showPage()
    c.setFont("Helvetica-Bold", 14)
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

    c.save()


if __name__ == "__main__":
    for i, topic in enumerate(topics):
        filename = f"sample_{i+1}.pdf"
        create_pdf(filename, topic, i)
        print(f"{filename} created!")
