
import os
import re
import csv
import io
import uuid

import cv2
import easyocr
import numpy as np

from flask import (
    Flask, render_template, request,
    jsonify, send_file
)

from PIL import Image
import fitz  # PyMuPDF


app = Flask(__name__)

app.config["MAX_CONTENT_LENGTH"] = 15 * 1024 * 1024

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

ALLOWED_EXTENSIONS = {
    "png", "jpg", "jpeg", "pdf"
}

# OCR languages:
# English, Hindi, Kannada, Tamil, Telugu
reader = easyocr.Reader(
    ["en", "hi", "kn", "ta", "te"],
    gpu=False
)

# Temporary storage for review and CSV export.
# For production, use a database.
documents = {}


def allowed_file(filename):
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower()
        in ALLOWED_EXTENSIONS
    )


def preprocess_image(image):
    """
    Improve scanned document quality.
    """

    image = np.array(image)

    if len(image.shape) == 3:
        gray = cv2.cvtColor(
            image,
            cv2.COLOR_RGB2GRAY
        )
    else:
        gray = image

    # Resize small images for better OCR.
    height, width = gray.shape

    if width < 1200:
        scale = 1200 / width

        gray = cv2.resize(
            gray,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_CUBIC
        )

    # Remove noise.
    gray = cv2.fastNlMeansDenoising(
        gray,
        None,
        10,
        7,
        21
    )

    # Improve contrast.
    gray = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    ).apply(gray)

    # Convert back to RGB.
    return cv2.cvtColor(
        gray,
        cv2.COLOR_GRAY2RGB
    )


def read_pdf(file_path):
    """
    Convert PDF pages into images.
    """

    pdf = fitz.open(file_path)

    images = []

    for page in pdf:
        pix = page.get_pixmap(
            matrix=fitz.Matrix(2, 2),
            alpha=False
        )

        image = Image.frombytes(
            "RGB",
            [pix.width, pix.height],
            pix.samples
        )

        images.append(image)

    return images


def extract_text(images):
    """
    Run OCR on all document pages.
    """

    all_text = []
    all_results = []

    for page_number, image in enumerate(images, start=1):

        processed = preprocess_image(image)

        results = reader.readtext(
            processed,
            detail=1,
            paragraph=False
        )

        for result in results:

            box, text, confidence = result

            all_results.append({
                "page": page_number,
                "text": text,
                "confidence": round(
                    float(confidence) * 100,
                    2
                ),
                "box": box
            })

            all_text.append(text)

    return "\n".join(all_text), all_results


def extract_fields(text, results):
    """
    Extract common document fields.

    These are heuristic rules.
    Users can correct the values manually.
    """

    fields = {}

    patterns = {
        "Email": r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}",

        "Phone": r"(?:\+91[\s-]?)?[6-9]\d{9}",

        "Date": r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b",

        "Amount": r"(?:₹|Rs\.?|INR)\s?[\d,]+(?:\.\d{1,2})?"
    }

    for field_name, pattern in patterns.items():

        match = re.search(
            pattern,
            text,
            re.IGNORECASE
        )

        if match:

            value = match.group(0)

            # Find OCR confidence for text
            # that overlaps the extracted value.
            confidence_values = [
                item["confidence"]
                for item in results
                if value.lower() in item["text"].lower()
                or item["text"].lower() in value.lower()
            ]

            confidence = (
                sum(confidence_values)
                / len(confidence_values)
                if confidence_values
                else 0
            )

            fields[field_name] = {
                "value": value,
                "confidence": round(confidence, 2),
                "review": confidence < 80
            }

    return fields


@app.route("/")
def index():

    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload_document():

    if "file" not in request.files:

        return jsonify({
            "error": "No document uploaded"
        }), 400

    file = request.files["file"]

    if file.filename == "":

        return jsonify({
            "error": "Please select a document"
        }), 400

    if not allowed_file(file.filename):

        return jsonify({
            "error": "Unsupported file type"
        }), 400

    document_id = str(uuid.uuid4())

    extension = file.filename.rsplit(
        ".", 1
    )[1].lower()

    file_path = os.path.join(
        UPLOAD_FOLDER,
        document_id + "." + extension
    )

    file.save(file_path)

    try:

        if extension == "pdf":

            images = read_pdf(file_path)

        else:

            image = Image.open(file_path).convert("RGB")

            images = [image]

        text, results = extract_text(images)

        fields = extract_fields(
            text,
            results
        )

        documents[document_id] = {
            "id": document_id,
            "filename": file.filename,
            "text": text,
            "results": results,
            "fields": fields
        }

        return jsonify({
            "id": document_id,
            "filename": file.filename,
            "text": text,
            "results": results,
            "fields": fields
        })

    except Exception as error:

        return jsonify({
            "error": str(error)
        }), 500


@app.route("/review/<document_id>", methods=["POST"])
def review_document(document_id):

    if document_id not in documents:

        return jsonify({
            "error": "Document not found"
        }), 404

    data = request.get_json()

    documents[document_id]["fields"] = data

    return jsonify({
        "message": "Review saved successfully",
        "fields": data
    })


@app.route("/export/<document_id>")
def export_document(document_id):

    if document_id not in documents:

        return jsonify({
            "error": "Document not found"
        }), 404

    fields = documents[document_id]["fields"]

    output = io.StringIO()

    writer = csv.writer(output)

    writer.writerow([
        "Field",
        "Extracted Value",
        "Confidence",
        "Needs Review"
    ])

    for name, field in fields.items():

        writer.writerow([
            name,
            field["value"],
            field["confidence"],
            field["review"]
        ])

    output.seek(0)

    return send_file(
        io.BytesIO(output.getvalue().encode("utf-8-sig")),
        mimetype="text/csv",
        as_attachment=True,
        download_name="document_data.csv"
    )


if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5000)),
        debug=False
    )
