from app.generator import generate_answer

contexts = [
    {"text": "Refunds are processed within 7 business days after verification."},
    {"text": "Contact support at support@example.com for refund requests."}
]

answer = generate_answer("When are refunds issued?", contexts)
print("Answer:", answer)
