"""
score_headlines.py

This script scores news headlines as Optimistic, Pessimistic, or Neutral
using a pre-trained SVM classifier and sentence embeddings from all-MiniLM-L6-v2.
"""

import sys
import os
from datetime import date
import joblib
from sentence_transformers import SentenceTransformer


def main():
    """Main script logic: load headlines, vectorize, predict, and write output."""

    if len(sys.argv) != 3:
        print("Usage: python score_headlines.py <input_file.txt> <source_name>")
        sys.exit(1)

    input_path = sys.argv[1]
    source_name = sys.argv[2]

    if not os.path.isfile(input_path):
        print(f"Error: input file '{input_path}' not found.")
        sys.exit(1)

    with open(input_path, "r", encoding="utf-8") as input_file_handle:
        headlines = [line.strip() for line in input_file_handle if line.strip()]

    if not headlines:
        print("Error: No headlines found in the input file.")
        sys.exit(1)

    local_model_path = "/opt/huggingface_models/all-MiniLM-L6-v2"
    if os.path.exists(local_model_path):
        model = SentenceTransformer(local_model_path)
    else:
        print("Local model path not found, using downloaded version...")
        model = SentenceTransformer("all-MiniLM-L6-v2")

    try:
        classifier = joblib.load("svm.joblib")
    except FileNotFoundError:
        print("Error: 'svm.joblib' not found. Make sure the model file is in the current directory.")
        sys.exit(1)

    embeddings = model.encode(headlines)
    predictions = classifier.predict(embeddings)

    today = date.today()
    output_filename = (
        f"headline_scores_{source_name}_{today.year}_{today.month:02}_{today.day:02}.txt"
    )

    with open(output_filename, "w", encoding="utf-8") as output_file_handle:
        for label, headline in zip(predictions, headlines):
            output_file_handle.write(f"{label},{headline}\n")

    print(f"Predictions written to {output_filename}")


if __name__ == "__main__":
    main()
