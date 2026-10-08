import streamlit as st
from PIL import Image
from google import genai

st.set_page_config(page_title='DocuSense AI', page_icon='??')
st.title('?? DocuSense AI')
st.caption('Powered by Gemma 4 via Gemini API')

api_key = st.sidebar.text_input('Gemini API Key', type='password')
file = st.file_uploader('Upload report/prescription', type=['jpg','png','jpeg'])
if file and st.button('Analyze'):
    client = genai.Client(api_key=api_key)
    res = client.models.generate_content(model='gemma-4-26b-a4b-it', contents=[Image.open(file), 'Explain this medical document in simple terms with key metrics and doctor questions.'])
    st.markdown(res.text)
