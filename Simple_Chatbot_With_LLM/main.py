# Q&A Chatbot
import os
import streamlit as st
from langchain.llms import OpenAI
from constants import openai_key

os.environ["OPENAI_API_KEY"] = openai_key
## Function to load OpenAI model and get respones

def get_openai_response(question):
    llm=OpenAI(openai_api_key = os.environ["OPENAI_API_KEY"],model_name="text-davinci-003",temperature=0.5)
    response=llm(question)
    return response

##initialize our streamlit app
st.set_page_config(page_title="Q&A Demo")
st.header("Langchain Application")


input=st.text_input("Input: ",key="input")
response=get_openai_response(input)

submit=st.button("Ask the question")

## If ask button is clicked
if submit:
    st.subheader("The Response is")
    st.write(response)
