from dotenv import load_dotenv
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferWindowMemory
from langchain_groq import ChatGroq
import warnings
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, File, UploadFile
from langchain_core.prompts import ChatPromptTemplate
import base64
import cv2
from PIL import Image
from io import BytesIO
import numpy as np
from langchain_google_genai import ChatGoogleGenerativeAI
from brain_cancer_detection import predict

warnings.filterwarnings("ignore", category=DeprecationWarning)
load_dotenv() 
model_llm = ChatGroq(model="llama-3.1-70b-versatile")
# model_vision = ChatGoogleGenerativeAI(model='gemini-1.5-flash', temperature=0)
model_vision = ChatGroq(model='llama-3.2-11b-vision-preview', temperature=0)
memory_llm = ConversationBufferWindowMemory(k=20)
memory_vision = ConversationBufferWindowMemory(k=10)
conv_chain_llm = ConversationChain(llm=model_llm, memory=memory_llm)
conv_chain_vision = ConversationChain(llm=model_vision, memory=memory_vision)


    
promp_vision = ChatPromptTemplate.from_messages([
    (
        "system", 
    '''Your task is to classify the given image is related to brain, breast or whether it is a medical prescription.Ask the user what type of image is this. If user says its related to brain then output 1 . if its related to brest output 2. Provide
    only the answer'''),
    (
        ("human", "{input} {text}")
    )
])

prompt_llm = ChatPromptTemplate.from_messages([
    (
        "system",
        '''You are a medical AI assistant specialized in healthcare topics. If the query
        is related to a medical term, provide a clear, accurate, and concise medical
        explanation. If the term could have non-medical meanings 
        (like technical, programming, or other fields), 
        do not attempt to explain it outside the medical context. 
        Politely inform the user that you only provide medical-related 
        information, and do not cover non-medical subjects. You are capable of accepting medical images like X-rays, mammograms, etc., and
        providing insights on those as well.''',
    ),
    ("human", "{input}"),
])
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
@app.get('/')
async def get_response(query:str):
    formatted_prompt = prompt_llm.format(input=query)
    result = conv_chain_llm(formatted_prompt)
    return result['response'].strip()

@app.post('/image')
async def get_image_response(file:UploadFile = File(...), text=str):
    file_content = await file.read()
    image = Image.open(BytesIO(file_content))
    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR) # pass to the detection model
    encoded_img = base64.b64encode(file_content).decode('utf-8')
    formated_prompt = promp_vision.format(input=encoded_img, text=text)
    result = conv_chain_vision(formated_prompt)
    choice = (int(result['response']))
    print(choice)
    if choice == 1:
        print(predict(img_cv))