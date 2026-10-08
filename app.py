import streamlit as st
import time
from PIL import Image
from google import genai

st.set_page_config(page_title='DocuSense AI', page_icon='medical')
st.title('DocuSense AI')
st.caption('Multimodal Medical Document Explainer')

api_key = st.sidebar.text_input('Gemini API Key', type='password')
file = st.file_uploader('Upload report/prescription', type=['jpg','png','jpeg'])

if file and st.button('Analyze'):
    if not api_key:
        st.warning('Please enter your Gemini API Key in the sidebar.')
    else:
        with st.spinner('Analyzing medical document...'):
            client = genai.Client(api_key=api_key.strip())
            img = Image.open(file)
            prompt = 'You are a compassionate medical explainer. Analyze this document/report image carefully. 1. Extract key readings/biomarkers with normal ranges. 2. Explain all findings in plain, simple everyday language. 3. List 3-4 specific questions the patient should ask their doctor. Include a clear medical disclaimer.'
            
            models = ['gemini-3.8-flash', 'gemini-2.5-flash']
            success = False
            last_err = ''

            for model_name in models:
                for attempt in range(2):
                    try:
                        res = client.models.generate_content(model=model_name, contents=[img, prompt])
                        st.markdown(res.text)
                        success = True
                        break
                    except Exception as e:
                        last_err = str(e)
                        time.sleep(2)
                if success:
                    break

            if not success:
                st.error(f'Google servers are experiencing high load right now. Please wait a moment and click Analyze again. (Details: {last_err})')
