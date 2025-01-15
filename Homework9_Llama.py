from flask import Flask, request, jsonify
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging
import os
import requests

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("tinyllama_backend")
app = Flask(__name__)
device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = os.getenv("MODEL_NAME", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")

try:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    logger.info(f"Model and tokenizer loaded for {model_name} on {device}.")
except Exception as e:
    logger.error(f"Error loading model: {e}")
    raise

def generate_response(prompt: str):
    try:
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        attention_mask = (input_ids != tokenizer.pad_token_id).type(torch.uint8).to(device)

        output_ids = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_length=100,
            num_return_sequences=1,
            temperature=0.9,
            top_p=0.95,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
        response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return response
    except Exception as e:
        logger.error(f"Error during response generation: {e}")
        return "Sorry, the model could not process your request."
@app.route('/llm', methods=['POST'])
def llm_endpoint():
    try:
        data = request.json
        if not data or "message" not in data:
            return jsonify({"error": "Invalid input. 'message' field is required."}), 400

        user_message = data["message"]

        # Log the input for debugging purposes
        logger.info(f"Received message: {user_message}")

        # Generate response
        response = generate_response(user_message)
        return jsonify({"response": response})

    except Exception as e:
        logger.error(f"Error in LLM endpoint: {e}")
        return jsonify({"error": "An error occurred while processing your request."}), 500

@app.route("/")
def home():
    return "Hello, World!"

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok"}), 200

def check():
    try:
        url = "http://127.0.0.1:5000/llm"
        payload = {"message": "Tell me about TinyLlama"}
        response = requests.post(url, json=payload)
        response.raise_for_status()
        print(response.json())
    except requests.exceptions.RequestException as e:
        logger.error(f"Error during health check: {e}")



if __name__ == "__main__":
    # Use a customizable port and enable production-ready features
    app.run(threaded=True)




