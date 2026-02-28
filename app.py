import streamlit as st
import os
import cv2
import pytesseract
import numpy as np
from PIL import Image
import pdf2image
import io
import random
from google import genai
from google.genai import types

# ==========================================
# SETUP & INSTALLATION NOTES
# ==========================================
# 1. Install required packages:
#    pip install streamlit opencv-python pytesseract pillow pdf2image
# 2. Tesseract OCR MUST be installed on your system.
#    - Windows: https://github.com/UB-Mannheim/tesseract/wiki
#    - You may need to specify the path below if it's not in your system PATH:
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
# 3. Poppler is required for `pdf2image` on Windows/Mac.
#    - Windows: https://github.com/oschwartz10612/poppler-windows/releases/
#    - Add poppler's bin folder to your system PATH.
# ==========================================

# --- 1. CONFIGURATION & UI SETUP ---
st.set_page_config(page_title="Health Report Simplified", page_icon="🧩", layout="centered")

st.title("Health Report Simplified")
st.subheader("Upload your medical report to receive a clear, structured summary.")

st.warning("⚠️ **NOT medical advice!** This is just a fun helper. Always ask your doctor for real interpretation.")

# --- 2. KEYWORD RULES & FUN PHRASES ---
# Dictionary Mapping: Keyword -> (Positive Phrase, Fun Phrase, Emoji)
# (Replaced by Gemini AI)
# --- 3. HELPER FUNCTIONS ---
def preprocess_image_for_ocr(image):
    """
    OpenCV preprocessing to make the text clearer for Tesseract.
    Converts to grayscale and applies thresholding.
    """
    # Convert PIL Image to OpenCV format (numpy array)
    # Ensure image is in RGB mode to avoid alpha channel issues
    img_cv = cv2.cvtColor(np.array(image.convert('RGB')), cv2.COLOR_RGB2BGR)
    
    # Convert to grayscale
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    
    # Apply thresholding (binarization) to enhance text against background
    # Using Otsu's thresholding
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Optional: You can do more filtering, but for simple scans this is enough
    return thresh

def extract_text(file_obj, file_type):
    """
    Extracts text from image or PDF.
    """
    text = ""
    try:
        if file_type == "pdf":
            # Convert first page of PDF to image
            # Note: poppler must be installed
            images = pdf2image.convert_from_bytes(file_obj.read(), first_page=1, last_page=1)
            if images:
                processed_img = preprocess_image_for_ocr(images[0])
                text = pytesseract.image_to_string(processed_img)
        else:
            # Handle Image directly
            image = Image.open(file_obj)
            processed_img = preprocess_image_for_ocr(image)
            text = pytesseract.image_to_string(processed_img)
    except Exception as e:
        st.error(f"Try clearer photo! 📸 OCR had an issue: {e}")
    
    return text.lower() # Convert to lowercase for easier string matching

# --- 4. MAIN APP LOGIC ---

st.sidebar.header("Gemini Settings")

# Try to get API key from Streamlit secrets first (for deployment)
try:
    api_key = st.secrets["GEMINI_API_KEY"]
except (FileNotFoundError, KeyError):
    # Fallback to sidebar input for local testing
    api_key = st.sidebar.text_input("Enter your Gemini API Key", type="password")

if not api_key:
    st.info("👈 Please enter your Gemini API Key in the sidebar or configure it in secrets to get started!")
    st.stop()

uploaded_file = st.file_uploader("Upload your report (PDF, PNG, JPG)", type=['pdf', 'png', 'jpg', 'jpeg'])

if uploaded_file is not None:
    st.info("Reading your report... 🔍🤓")
    
    file_type = "pdf" if uploaded_file.name.endswith(".pdf") else "image"
    
    with st.spinner("Decoding doctor handwriting... 🩺"):
        extracted_text = extract_text(uploaded_file, file_type)
        
    if not extracted_text.strip():
        st.error("Whoops! Couldn't find any text. Try a clearer photo! 📸")
    else:
        st.success("Extracted text successfully! 🎉")
        
        # --- Layout with Tabs ---
        tab1, tab2 = st.tabs(["Fun Summary ✨", "Raw Text 📄"])
        
        with tab1:
            st.subheader("Here's your Fun Summary! ✨")
            
            with st.spinner("Asking Gemini to explain this simply... 🤖"):
                try:
                    client = genai.Client(api_key=api_key)
                    prompt = f"""
                    You are a 'Fun Report Buddy'. I am giving you the raw text extracted from a medical or lab report via OCR. 
                    
                    Please format your response exactly like this:
                    1. Start with the exact phrase: "Hey, let's figure this out what you got..."
                    2. Give a VERY SHORT and SIMPLE explanation of the core findings to someone who doesn't know any medical terms. Do not make a big complicated summary.
                    3. End with a lighthearted, related joke or humorous comment.
                    
                    Do not provide any real medical advice or diagnoses. 
                    If the text doesn't look like a medical report or is unreadable, just playfully say so!
                    
                    Raw OCR Text:
                    {extracted_text}
                    """
                    response = client.models.generate_content(
                        model='gemini-2.5-flash',
                        contents=prompt,
                    )
                    
                    st.markdown(response.text)
                    
                    # Random fake "Fun Health Score"
                    score = random.randint(75, 100)
                    st.metric(label="Your Fun Health Score (Totally Fake!)", value=f"{score}/100")
                    
                    # Fun visual feedback
                    if score > 90:
                        st.write("### Rating: Superhero level! 🦸‍♂️")
                        st.balloons()
                    elif score > 80:
                        st.write("### Rating: Looking chill like a boss 😎")
                    else:
                        st.write("### Rating: Great job! Always room to level up! 🚀")
                        
                    st.write("---")
                    
                    # Shareable text
                    share_text = "My Fun Report Buddy Summary!\n\n" + response.text + f"\n\nFun Health Score: {score}/100! 🤓"
                    st.download_button(
                        label="📋 Copy Fun Summary (Save Text)",
                        data=share_text,
                        file_name="fun_summary.txt",
                        mime="text/plain"
                    )

                except Exception as e:
                    st.error(f"Failed to connect to Gemini: {e}")
        
        with tab2:
            st.write("Here is the raw text we found:")
            with st.expander("Show/Hide Raw OCR Text"):
                st.text(extracted_text)
