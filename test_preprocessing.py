from app.preprocessing import preprocess_text


text = """
Machine Learning models are being used for predicting diseases
and improving healthcare outcomes. These models can analyze
large amounts of medical data.
"""


tokens = preprocess_text(text)


print()
print("=" * 60)
print("PREPROCESSING TEST")
print("=" * 60)

print("\nOriginal text:")
print(text)

print("\nProcessed tokens:")
print(tokens)

print("\nNumber of tokens:")
print(len(tokens))