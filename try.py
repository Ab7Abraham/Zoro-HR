import streamlit as st
import cohere
from PyPDF2 import PdfReader

uploaded_file = st.file_uploader('Choose your .pdf file', type="pdf")
input_text = st.text_input(label='Give the role you are applying for',key="input")
print('-----------------',input_text,'--------------')
print(type(input_text))
if input_text=='': 
     print('+++++++++++',1)
     if uploaded_file is not None:
        reader = PdfReader("Abraham DA.pdf")
        number_of_pages = len(reader.pages)
        page = reader.pages
        print(page[0].extract_text())
API_KEY='Kk5Drx88gQ1JurDXRxmIhYfPYtREtE4X6n0KN7lL'
co = cohere.Client(API_KEY)
response_resume = co.generate(
      model='command-xlarge-nightly',
      prompt='Hello:',
      max_tokens=10,
      temperature=0,
      k=0,
      p=0.75,
      # frequency_penalty=0,
      presence_penalty=0.5,
      stop_sequences=[],
      return_likelihoods='NONE')
print('response_resume:',response_resume.generations[0].text)

# import streamlit as st
# import cohere
# from PyPDF2 import PdfReader

# uploaded_file = st.file_uploader('Choose your .pdf file', type="pdf")
# input_role = st.text_input(label='Give the role you are applying for',key="input_role")

# if uploaded_file is not None and input_role:
#     reader = PdfReader(uploaded_file)
#     number_of_pages = len(reader.pages)
#     page = reader.pages

#     response_resume = co.generate(
#       model='command-xlarge-nightly',
#       prompt=page[0].extract_text() + '\n Extract the following data from above resume - Education details, Work experience, Projects completed, Language:',
#       max_tokens=300,
#       temperature=0,
#       k=0,
#       p=0.75,
#       stop_sequences=[],
#       return_likelihoods='NONE'
#     )

#     response_behavioral_q = co.generate(
#       model='command-xlarge-nightly',
#       prompt=f'Following text is my resume: {response_resume.generations[0].text}\nI am preparing for a behavioral interview for a {input_role} position. Please generate a behavioral question for me:',
#       max_tokens=50,
#       temperature=0.7,
#       k=0,
#       p=0.75,
#       stop_sequences=['\n'],
#       return_likelihoods='NONE'
#     )

#     st.write(response_behavioral_q.generations[0].text)