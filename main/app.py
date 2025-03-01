from flask import Flask, request, jsonify
import ollama
from flask_cors import CORS

app = Flask(__name__)

CORS(app)  # This will allow requests from any domain


@app.route('/generate', methods=['POST'])
def generate_response():
    try:
        data = request.get_json()
        prompt = data.get("prompt", "")

        if not prompt:
            return jsonify({"error": "Missing prompt"}), 400

        response = ollama.chat(model="llama2", messages=[{"role": "user", "content": prompt}])
        
        return jsonify({"response": response["message"]["content"]})  # ✅ Ensure correct key

    except Exception as e:
        print(f"Server Error: {str(e)}")  # ✅ Print error for debugging
        return jsonify({"error": "Server error occurred"}), 500

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=5000)
