import streamlit as st
from dotenv import load_dotenv
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferWindowMemory
from langchain_groq import ChatGroq
import warnings
from langchain_core.prompts import ChatPromptTemplate
import base64
import cv2
from PIL import Image
import numpy as np
from langchain_google_genai import ChatGoogleGenerativeAI
import os

warnings.filterwarnings("ignore", category=DeprecationWarning)
load_dotenv()


groq_api_key = os.getenv('GROQ_API_KEY')
google_api_key = os.getenv('GOOGLE_API_KEY')

model_llm = ChatGroq(model="llama-3.1-70b-versatile")
model_vision = ChatGroq(model='llama-3.2-11b-vision-preview', temperature=0)
memory_llm = ConversationBufferWindowMemory(k=20)
memory_vision = ConversationBufferWindowMemory(k=10)
conv_chain_llm = ConversationChain(llm=model_llm, memory=memory_llm)
conv_chain_vision = ConversationChain(llm=model_vision, memory=memory_vision)

# Define prompts
prompt_vision = ChatPromptTemplate.from_messages([
    ("system", '''Your task is to classify the given image is related to brain, breast or whether it is a medical prescription. Provide
    only the answer. Ask the user what type of image is this. If user says its related to brain then output 1. If its related to breast output 2'''),
    ("human", "{input} {text}")
])

prompt_llm = ChatPromptTemplate.from_messages([
    ("system", '''You are a medical AI assistant specialized in healthcare topics. If the query
        is related to a medical term, provide a clear, accurate, and concise medical
        explanation. If the term could have non-medical meanings 
        (like technical, programming, or other fields), 
        do not attempt to explain it outside the medical context. 
        Politely inform the user that you only provide medical-related 
        information, and do not cover non-medical subjects. You are capable of accepting medical images like X-rays, mammograms, etc., and
        providing insights on those as well.'''),
    ("human", "{input}"),
])

st.title("Medical AI Assistant")

query = st.text_input("Enter your medical query:")
if query:
    formatted_prompt = prompt_llm.format(input=query)
    result = conv_chain_llm(formatted_prompt)
    st.write(result['response'].strip())

uploaded_file = st.file_uploader("Upload a medical image", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    buffered = BytesIO()
    image.save(buffered, format="PNG")
    encoded_img = base64.b64encode(buffered.getvalue()).decode()
    
    image_type = st.radio("What type of image is this?", ("Brain", "Breast", "Prescription"))
    
    if st.button("Analyze Image"):
        text = f"This is a {image_type.lower()} image."
        formatted_prompt = prompt_vision.format(input=encoded_img, text=text)
        result = conv_chain_vision(formatted_prompt)
        choice = int(result['response'])
        
        if choice == 1:
            st.write("This image is related to the brain.")
            st.write("Performing detailed brain image analysis...")
        elif choice == 2:
            st.write("This image is related to the breast.")
            st.write("Performing detailed breast image analysis...")

        else:
            st.write("This image appears to be a medical prescription.")
            st.write("Analyzing prescription details...")

        st.write("Additional analysis based on the image type can be implemented here.")