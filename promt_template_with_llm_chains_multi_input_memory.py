## Integrate the code OpenAI API
import os
from constants import openai_key
from langchain.llms import OpenAI
from langchain import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SequentialChain
from langchain.memory import ConversationBufferMemory
import streamlit as st

## Initialise streamlit framework
st.title("Langchain Demo With OpenAi API [Celebrity Search Results]")
input_text = st.text_input("Search the topic you want.")

## Initialise the LLM model here(OpenAi)
os.environ["OPENAI_API_KEY"]=openai_key
llm = OpenAI(temperature = 0.8) #  Controls randomness in responses (higher = more creative, lower = more deterministic).

## 1st Prompt Templates
first_input_prompt=PromptTemplate(
    input_variables=['name'],
    template="Tell me about celebrity {name}"
)
# Memory
person_memory = ConversationBufferMemory(input_key='name', memory_key='chat_history')
## Initialise Chain-1
chain1 = LLMChain(llm = llm, prompt = first_input_prompt, output_key='person', verbose = True)

## 2nd Prompt Templates
second_input_prompt=PromptTemplate(
    input_variables=['person'],
    template="when was {person} born"
)
# Memory
dob_memory = ConversationBufferMemory(input_key='person', memory_key='chat_history')
## Initialise Chain-2
chain2 = LLMChain(llm = llm, prompt = second_input_prompt, output_key='dob', verbose = True)

## 3rd Prompt Templates
third_input_prompt=PromptTemplate(
    input_variables=['dob'],
    template="Mention 5 major events happened around {d0b} in the world"
)
# Memory
descr_memory = ConversationBufferMemory(input_key='dob', memory_key='description_history')
## Initialise Chain-3
chain3 = LLMChain(llm = llm, prompt = third_input_prompt, output_key='description', verbose = True)


parent_chain = SequentialChain(chains = [chain1, chain2, chain3], input_variables = ['name'], output_variables = ['person', 'dob', 'description'], verbose = True)
## Returen the response from API call
if input_text:
    st.write(parent_chain({'name':input_text}))
    with st.expander('Person Name'): 
        st.info(person_memory.buffer)

    with st.expander('Major Events'): 
        st.info(descr_memory.buffer)