import streamlit as st
import requests

# Set your API token and model URL
API_TOKEN = "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"  # Use the correct, valid token
MODEL_URL = "https://api-inference.huggingface.co/models/Meta-Llama-3.1-8B-Instruct"  # Try with a public model first

HEADERS = {
    "Authorization": f"Bearer {API_TOKEN}"
}

st.title("Hugging Face Chatbot")

user_input = st.text_input("You: ")

if st.button("Send"):
    if user_input.strip() == "":
        st.write("Please enter a message.")
    else:
        response = requests.post(MODEL_URL, headers=HEADERS, json={"inputs": user_input})

        if response.status_code == 200:
            output = response.json()
            st.write("Bot:", output)
        else:
            st.write("Error:", response.status_code, response.text)
