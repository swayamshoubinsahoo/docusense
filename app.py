import streamlit as st
import time
from PIL import Image
from google import genai

# Page configuration
st.set_page_config(page_title='DocuSense', page_icon='medical')
st.title('DocuSense')
st.caption('Multimodal Medical Document Explainer')

# Sidebar for API key input
api_key = st.sidebar.text_input('Gemini API Key', type='password')

# File upload component
file = st.file_uploader('Upload report/prescription', type=['jpg', 'png', 'jpeg'])

# Action button and generation workflow
if file and st.button('Analyze'):
    if not api_key:
        st.warning('Please enter your Gemini API Key in the sidebar.')
    else:
        with st.spinner('Analyzing medical document...'):
            client = genai.Client(api_key=api_key.strip())
            img = Image.open(file)
            prompt = (
                'You are a compassionate medical explainer. '
                'Analyze this document/report image carefully. '
                '1. Extract key readings/biomarkers. '
                '2. Explain the results in simple, reassuring language.'
            )

            # Fallback list of models to handle high load or 503 errors
            models_to_try = [
                'gemini-2.5-flash',
                'gemini-2.0-flash',
                'gemini-1.5-flash'
            ]

            success = False
            last_err = ''

            for model_name in models_to_try:
                try:
                    res = client.models.generate_content(
                        model=model_name,
                        contents=[img, prompt]
                    )
                    if res and res.text:
                        st.markdown(res.text)
                        success = True
                        break  # Stop trying if the model responds successfully
                except Exception as e:
                    last_err = str(e)
                    time.sleep(1)  # Brief pause before falling back to the next model
                    continue

            if not success:
                st.error(f"Service busy across all available models. Please try again shortly. Details: {last_err}")
