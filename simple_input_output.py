## Integrate the code OpenAI API
import os
from constants import openai_key
from langchain.llms import OpenAI
import streamlit as st

## Initialise streamlit framework
st.title("Langchain Demo With OpenAi API")
input_text = st.text_input("Search the topic you want.")

## Initialise the LLM model here(OpenAi)
os.environ["OPENAI_API_KEY"]=openai_key
llm = OpenAI(temperature = 0.8) #  Controls randomness in responses (higher = more creative, lower = more deterministic).

## Returen the response from API call
if input_text:
    st.write(llm(input_text))