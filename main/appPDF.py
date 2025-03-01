from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import fitz  # PyMuPDF for PDF text extraction
import torch
from transformers import pipeline
from werkzeug.utils import secure_filename

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for frontend communication

# Set up Upload Folder
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)  # Ensure the uploads directory exists
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Determine device (use GPU if available)
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load Hugging Face models
print("Loading models... This may take a few seconds.")
summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=0 if device == "cuda" else -1)
qa_model = pipeline("question-answering", model="deepset/roberta-base-squad2", device=0 if device == "cuda" else -1)
print("Models loaded successfully!")

# Function to extract text from a PDF file
def extract_text_from_pdf(pdf_path):
    try:
        doc = fitz.open(pdf_path)
        text = "\n".join(page.get_text("text") for page in doc)
        return text if text.strip() else "No text found in PDF."
    except Exception as e:
        return f"Error extracting text: {str(e)}"

# Function to generate a summary
def generate_summary(text):
    try:
        max_input_length = 1024  # Model token limit
        summary_list = []
        for i in range(0, len(text), max_input_length):
            chunk = text[i:i + max_input_length]
            summary = summarizer(chunk, max_length=300, min_length=50, do_sample=False)
            summary_list.append(summary[0]['summary_text'])
        return " ".join(summary_list)
    except Exception as e:
        return f"Error generating summary: {str(e)}"

# Function to perform Q&A
def ask_question(text, question):
    try:
        answer = qa_model(question=question, context=text[:4000])
        return answer['answer']
    except Exception as e:
        return f"Error answering question: {str(e)}"
@app.route("/", methods=["GET"])
def home():
    return "Flask server is running!", 200

# API Endpoint to Upload PDF and Extract Text
@app.route("/upload", methods=["POST"])
def upload_pdf():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Empty file"}), 400

    if not file.filename.lower().endswith(".pdf"):
        return jsonify({"error": "Invalid file type. Only PDFs are allowed."}), 400

    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(file_path)

    try:
        extracted_text = extract_text_from_pdf(file_path)
        return jsonify({"text": extracted_text})
    except Exception as e:
        return jsonify({"error": f"Failed to process PDF: {str(e)}"}), 500
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)

# API Endpoint to Summarize PDF
@app.route("/summarize_pdf", methods=["POST"])
def summarize_pdf():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Empty file"}), 400

    if not file.filename.lower().endswith(".pdf"):
        return jsonify({"error": "Invalid file type. Only PDFs are allowed."}), 400

    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(file_path)

    try:
        extracted_text = extract_text_from_pdf(file_path)
        summary = generate_summary(extracted_text)
        return jsonify({"summary": summary})
    except Exception as e:
        return jsonify({"error": f"Failed to summarize PDF: {str(e)}"}), 500
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)

# API Endpoint to Summarize Extracted Text
@app.route("/summarize", methods=["POST"])
def summarize():
    data = request.get_json()
    if not data or "text" not in data:
        return jsonify({"error": "Missing 'text' field in JSON"}), 400

    text = data["text"]
    try:
        summary = generate_summary(text)
        return jsonify({"summary": summary})
    except Exception as e:
        return jsonify({"error": f"Failed to summarize text: {str(e)}"}), 500

# API Endpoint to Ask Questions on Extracted Text
@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()
    if not data or "text" not in data or "question" not in data:
        return jsonify({"error": "Missing 'text' or 'question' in JSON"}), 400

    text = data["text"]
    question = data["question"]
    try:
        answer = ask_question(text, question)
        return jsonify({"answer": answer})
    except Exception as e:
        return jsonify({"error": f"Failed to answer question: {str(e)}"}), 500

# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)
