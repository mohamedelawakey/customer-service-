import os
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

# Load environment variables from .env file
load_dotenv()

HF_TOKEN = os.getenv("HF_API_KEY")
if HF_TOKEN is None:
    raise ValueError("HF_API_KEY not found. Please set it in .env file.")

client = InferenceClient(model="mistralai/Mistral-7B-Instruct-v0.2", token=HF_TOKEN)

def ask_hf(prompt, max_tokens=256, temperature=0.2):
    response = client.chat_completion(
        messages=[
            {"role": "system", "content": "You are a friendly and concise customer support assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=max_tokens,
        temperature=temperature
    )
    return response.choices[0].message["content"].strip()

# Example usage
print(ask_hf("How can I return a product I purchased from the store?"))
