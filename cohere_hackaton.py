import subprocess

# List of required packages
required_packages = ['cohere', 'numpy', 'streamlit_chat', 'PyPDF2','streamlit']

# Function to install packages
def install_packages(packages):
    for package in packages:
        subprocess.run(['pip', 'install', package])

# Install required packages
install_packages(required_packages)
import cohere
import numpy as np
import streamlit as st
from streamlit_chat import message
from amazon import set_background 
from PyPDF2 import PdfReader

API_KEY = 'BXim1rK6O5O3iHccf3YLkdZLN2QBvehqag2FtMEf'
co = cohere.Client(API_KEY)

st.set_page_config(
    page_title="ZORO",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded"
)

set_background('lap_4.png')

st.markdown("""
    <style>
        .sidebar .sidebar-content {
            width: 375px;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1 style='font-family:Courier; color:white; font-size: 70px;text-align: center;'>ZORO</h1>", unsafe_allow_html=True)

if 'generated' not in st.session_state:
    st.session_state['generated'] = []    
if 'past' not in st.session_state:
    st.session_state['past'] = []   
if 'preprocess' not in st.session_state:
    st.session_state['preprocess'] = 0
if 'questions' not in st.session_state:
    st.session_state['questions'] = ''    

uploaded_file = st.file_uploader('Choose your .pdf file', type="pdf")
input_role = st.text_input(label='Give the role you are applying for', key="input_role")

def calculate_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def correctness_check(question, answer, role=input_role):
    text_list = []
    for i in range(3):
        definitions = co.generate(
            model='command-xlarge-nightly',
            prompt=f'{role} Question: {question}?',
            max_tokens=100,
            temperature=0.7,  # Adjust temperature if needed
            k=0,
            p=0.75,
            presence_penalty=1,
            stop_sequences=["\n"]
        )
        text_list.append(definitions.generations[0].text)

    (def_1, def_2, def_3) = co.embed(text_list).embeddings
    answer_emb = (co.embed(list(answer)).embeddings)[0]

    average_similarity = calculate_similarity(def_1, answer_emb) + calculate_similarity(def_2, answer_emb) + calculate_similarity(def_3, answer_emb)
    average_similarity = average_similarity / 3
    return average_similarity

def get_text():
    input_text = st.text_input(label='Answer my questions !:',key="input")
    return input_text

def query():
    response = co.generate(
        model='command-xlarge-nightly',
        prompt=st.session_state['pre_prompt'][-1],
        max_tokens=120,
        temperature=0.5,
        k=0,
        p=0.75,
        presence_penalty=0.50,
        stop_sequences=["Answer:", "Candidate:",'\n']
    )

    bot = response.generations[0].text
    user = get_text()
    prompt = st.session_state['pre_prompt'][-1] + ' ' + bot + 'Candidate: ' + user + '\nTechnical Interviewer:'

    st.session_state.past.append(user)
    st.session_state.generated.append(bot)
    prompt = st.session_state['pre_prompt'][-1] + ' ' + bot + 'Candidate: ' + user + '\nTechnical Interviewer:'
    st.session_state['pre_prompt'].append(prompt)  

    if st.session_state['iterator'] > 2:
        st.session_state['correctness_list'].append(correctness_check(question=bot, answer=user, role=input_role))
    st.session_state['iterator'] += 1

    if user == 'stop':
        score = np.sum(st.session_state['correctness_list']) / i
        if score != 0:
            st.write(score)
        else:
            st.write('Score not available yet ! Try answering more questions')
    return response.generations[0].text

if (uploaded_file is not None) and (input_role != ''):
    if st.session_state['preprocess'] == 0:
        reader = PdfReader(uploaded_file)
        number_of_pages = len(reader.pages)
        page = reader.pages

        response_resume = co.generate(
            model='command-xlarge-nightly',
            prompt=page[0].extract_text()+'\n Extract the following data from above resume - Education details, Work experience, Projects completed, Language:',
            max_tokens=300,
            temperature=0,
            k=0,
            p=0.75,
            presence_penalty=0,
            stop_sequences=[]
        )

        NUM_BEHAVIORAL_QUESTION = 3
        response_behaviour_q = co.generate(
            model='command-xlarge-nightly', 
            prompt=f'Following text is my resume: {response_resume.generations[0].text}\nI am preparing for a behavioral interview for a {input_role} position. Please generate a behavioral question for me:', 
            max_tokens=50, 
            temperature=0.7,
            k=0,
            p=0.75,
            presence_penalty=0.5,
            stop_sequences=[str(NUM_BEHAVIORAL_QUESTION+1)]
        )

        NUM_TECHNICAL_QUESTION = 7
        response_technical_q = co.generate(
            model='command-xlarge-nightly',
            prompt=f'Data Science: Explain the difference between Linear Regression and Logistic Resression.\nSystem Design: Design a distributed file system on top of commodity hardware.\nDatabase: How do you design a database schema to store user activity?\nNetworking: How does the TCP handshake work?\nLinux: What is the difference between hard links and symbolic links?\nC: What are some limitations of malloc()/free()?\nPython: What is the difference between .__getitem__() and .[]?\nCSS: How do you vertically center an element using CSS?\nR: What is the RStudio IDE?\nSalesforce: What is the difference between Workflow, Process Builder, and Apex?\nMongoDB: How would you model a blog in MongoDB?\nSDE: How would you design a REST API for a social media app?\nQA: How do you write a test case for Google Maps?\nDigital Marketing: How do you measure the success of a digital marketing campaign?\nJAVA: What is the difference between checked and unchecked exceptions?\nDevOps: How do you configure a production environment on AWS?\nGit: How do you resolve merge conflicts in Git?\nFollowing the above format, write {str(NUM_TECHNICAL_QUESTION)} of most diverse and logical technical {input_role} interview questions without answers from Geeks for Geeks:', 
            max_tokens=300, 
            temperature=0.7,
            k=0,
            p=0.75,
            presence_penalty=0.76,
            stop_sequences=[str(NUM_TECHNICAL_QUESTION+1)]
        )

        questions = response_behaviour_q.generations[0].text + response_technical_q.generations[0].text
        st.session_state['questions'] = questions

        st.session_state['preprocess'] = 1

        prompt = f'''Below is a series of chats between Technical Interviewer and Candidate. In this chat, the Technical Interviewer is conducting a technical interview for a job position. The Technical Interviewer asks the Candidate technical questions related to the job requirements, assesses their technical knowledge, and evaluates their problem-solving skills. The Technical Interviewer speaks professionally and objectively, providing clear and concise feedback. The Technical Interviewer doesn't stop asking questions unless the Candidate explicitly asks to stop the interview. The Technical Interviewer never repeats the same questions twice.\n{questions}\nAsk questions from the above given questions to a candidate in an interview form:\nTechnical Interviewer:'''

        if 'pre_prompt' not in st.session_state:   
            st.session_state['pre_prompt'] = [prompt]

        if 'iterator' not in st.session_state:   
            st.session_state['iterator'] = 0

        if 'correctness_list' not in st.session_state:   
            st.session_state['correctness_list'] = []

    output = query()

    if st.session_state['generated']:
        for i in range(len(st.session_state['generated'])-1, -1, -1):
            message(st.session_state["generated"][i], key=str(i))
            message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')
