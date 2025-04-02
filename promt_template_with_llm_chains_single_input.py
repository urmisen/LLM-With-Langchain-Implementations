## Integrate the code OpenAI API
import os
from constants import openai_key
from langchain.llms import OpenAI
from langchain import PromptTemplate
from langchain.chains import LLMChain
import streamlit as st

## Initialise streamlit framework
st.title("Langchain Demo With OpenAi API [Celebrity Search Results]")
input_text = st.text_input("Search the topic you want.")

## Initialise the LLM model here(OpenAi)
os.environ["OPENAI_API_KEY"]=openai_key
llm = OpenAI(temperature = 0.8) #  Controls randomness in responses (higher = more creative, lower = more deterministic).

## Prompt Templates
first_input_prompt=PromptTemplate(
    input_variables=['name'],
    template="Tell me about celebrity {name}"
)

## Initialise Chain
chain = LLMChain(llm = llm, prompt = first_input_prompt, verbose = True)

## Returen the response from API call
if input_text:
    st.write(chain.run(input_text))