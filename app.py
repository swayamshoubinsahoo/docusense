import streamlit as st
import os
from PIL import Image
from google import genai

st.set_page_config(
    page_title="DocuSense - Multimodal Medical Document Explainer",
    page_icon="📄",
    layout="centered"
)

st.title("DocuSense")
st.subheader("Multimodal Medical Document Explainer")

# 1. API Key Input
api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    try:
        if "GEMINI_API_KEY" in st.secrets:
            api_key = st.secrets["GEMINI_API_KEY"]
    except Exception:
        api_key = None

if not api_key:
    api_key = st.sidebar.text_input("Enter Gemini API Key", type="password")
    if not api_key:
        st.info("Please enter your Gemini API key in the sidebar to proceed.")
        st.stop()

# 2. Client Initialization
try:
    client = genai.Client(api_key=api_key)
except Exception as e:
    st.error(f"Initialization error: {e}")
    st.stop()

# 3. Dynamic Model Picker
@st.cache_resource(show_spinner=False)
def resolve_working_models(_client):
    try:
        models = list(_client.models.list())
        valid_models = []
        for m in models:
            m_name = getattr(m, 'name', '') or str(m)
            actions = getattr(m, 'supported_actions', []) or []
            if "generateContent" in actions or not actions:
                clean = m_name.replace("models/", "")
                valid_models.append(clean)
        
        # Sort so 'flash' models come first
        valid_models.sort(key=lambda x: (0 if "flash" in x.lower() else 1, x))
        return valid_models if valid_models else ["gemini-2.5-flash"]
    except Exception:
        return ["gemini-2.5-flash", "gemini-2.0-flash"]

available_models = resolve_working_models(client)

# Allow manual override or auto-pick first working model
selected_model = st.sidebar.selectbox("Active Model", available_models, index=0)

uploaded_file = st.file_uploader("Upload report/prescription", type=["jpg", "jpeg", "png", "webp"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Document Preview", use_container_width=True)

    if st.button("Analyze"):
        with st.spinner(f"Analyzing with {selected_model}..."):
            prompt = """
            You are an accurate medical document analysis assistant.
            Analyze this medical document or prescription:
            1. Summarize the key findings or diagnosis clearly.
            2. List any prescribed medications, dosages, and instructions mentioned.
            3. Highlight key clinical advice or follow-up recommendations.
            4. Provide standard disclaimers noting that this AI explanation is for informational purposes and should be verified by a qualified physician.
            """

            try:
                res = client.models.generate_content(
                    model=selected_model,
                    contents=[img, prompt]
                )
                if res and res.text:
                    st.markdown(res.text)
                else:
                    st.warning("Model completed the call, but returned empty text.")
            except Exception as e:
                st.error(f"Error with {selected_model}: {e}")
