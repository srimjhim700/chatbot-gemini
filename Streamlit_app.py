from dataclasses import dataclass
from typing import Literal
import streamlit as st
import requests

import google.generativeai as genai
import os


st.title("welcome to Bail Reckoner !")
api_key =st.secrets["api_key"]
api_url = "https://api.gemini.com/v1/completions"
genai.configure(api_key=api_key)
model = genai.GenerativeModel("gemini-1.5-flash")

if "history" not in st.session_state:
     st.session_state.history = []



def  get_gemini_response(prompt):
    
    response = model.generate_content(prompt)
    print(response.text)
    return response.text
# Create a text input for the user prompt
user_input = st.text_input("Ask me anything", value="Hello Gemini!")

# If the submit button is clicked
if st.button("Submit"):
    if user_input:

        st.session_state.history.append(("you: ",user_input))
        # Get the response from Gemini API
        response = get_gemini_response(user_input)
        st.session_state.history.append(("ai: ",response))

        # Display the API response in Streamlit
        for sender,message in st.session_state.history:
            st.write(f"{sender}{message}")          
        
    else:
        st.write("Please enter a prompt.")








