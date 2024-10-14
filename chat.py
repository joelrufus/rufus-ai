import os
import google.generativeai as genai
from dotenv import load_dotenv
from flask import Flask, request, jsonify, render_template

# Load environment variables from .env file
load_dotenv()

# Configure the Gemini API with the key from .env
genai.configure(api_key=os.environ["GEMINI_API_KEY"])

# Define the generation configuration
generation_config = {
    "temperature": 0.7,
    "top_p": 0.9,
    "top_k": 40,
    "max_output_tokens": 300,
    "response_mime_type": "text/plain",
}

# Initialize the Flask app
app = Flask(__name__)

# Initialize the model with system instructions
model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    generation_config=generation_config,
    system_instruction=(
        "You are an educational assistant named Rufus, specializing in computer science and mathematics. "
        "Your role is to provide clear, accurate answers. If a question is vague, ask for more clarification. "
        "Use friendly language and provide code snippets or step-by-step solutions when necessary."
    ),
)

# Route to serve the HTML frontend
@app.route('/')
def index():
    return render_template('index.html')

# Route to handle chat requests from the frontend
@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.json.get('message', '')

    # Start a new chat session
    chat_session = model.start_chat(history=[])
    response = chat_session.send_message(user_message)

    # Send the response back to the frontend
    return jsonify({'reply': response.text})

# Run the Flask app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
