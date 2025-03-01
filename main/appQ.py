from flask import Flask, request, jsonify
import requests
import re

app = Flask(__name__)

# Hugging Face API Key 
API_KEY = "---"

# Hugging Face API URL
API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.3"

# Headers for Hugging Face API
headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

def parse_quiz(quiz_text):
    """Extracts structured quiz questions, options, and correct answers."""
    questions = []
    
    # Improved regex pattern for better parsing
    matches = re.findall(
        r'(\d+\..*?)\n\s*a\)(.*?)\n\s*b\)(.*?)\n\s*c\)(.*?)\n\s*d\)(.*?)\n\s*Correct Answer:\s*([a-dA-D])',
        quiz_text,
        re.DOTALL
    )

    for match in matches:
        question_text = match[0].strip()
        options = [match[1].strip(), match[2].strip(), match[3].strip(), match[4].strip()]
        
        # Convert answer letter (A, B, C, D) to the corresponding option text
        correct_answer_index = ord(match[5].upper()) - ord('A')  # Convert 'A' -> 0, 'B' -> 1, etc.
        correct_answer_text = options[correct_answer_index] if 0 <= correct_answer_index < 4 else None

        questions.append({
            "question": question_text,
            "options": options,
            "answer": correct_answer_text
        })

    return questions

@app.route('/generate_quiz', methods=['POST'])
def generate_quiz():
    try:
        data = request.get_json()
        topic = data.get("topic", "Python programming")  # Default topic
        num_questions = data.get("num_questions", 5)

        # Construct prompt
        prompt = (
            f"Generate a {num_questions}-question multiple-choice quiz on {topic}."
            " Each question should have exactly four options labeled a, b, c, and d."
            " At the end of each question, state 'Correct Answer: ' followed by the correct option letter."
        )

        # Request payload
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": 500,
                "temperature": 0.7,
                "top_p": 0.9,
                "do_sample": True,
                "return_full_text": False
            }
        }

        # Send request to Hugging Face API
        response = requests.post(API_URL, json=payload, headers=headers)

        if response.status_code == 200:
            response_data = response.json()

            if isinstance(response_data, list) and len(response_data) > 0:
                quiz_text = response_data[0].get("generated_text", "No quiz generated.")
                quiz = parse_quiz(quiz_text)  # Convert raw text to structured format
                
                if quiz:
                    return jsonify({"quiz": quiz})
                else:
                    return jsonify({"error": "Failed to parse quiz format."}), 500
            else:
                return jsonify({"error": "Invalid response format from API."}), 500

        else:
            return jsonify({"error": response.json()}), response.status_code

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
