from langchain.agents import create_agent
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from tools import web_search,web_scrap
from dotenv import load_dotenv
load_dotenv()

# Model Setup
llm = ChatMistralAI(model='mistral-small-2506',temperature=0)

parser = StrOutputParser()

# 1st Agent 

def build_search_agent():
    return create_agent(
        model=llm,
        tools=[web_search]
    )


# 2nd Agent 

def build_reader_agent():
    return create_agent(
        model=llm,
        tools=[web_scrap]
    )


# Writer Chain

writer_prompt = ChatPromptTemplate.from_messages([
    ('system','You are an expert research writer. write clear,structured and insightful reports'),
    ('human','''Write a detailed research report on topic below.
    Topic:{topic}
    Research gathered:{research}
    Structured the report as :
    - Introduction
    - Key Findings (minimum 3 well-explained points)
    - Conclusion
    -Sources (list all URLs found in the research)
     Be Detailed ,factual and professional     
     ''')
])

writer_chain = writer_prompt | llm | parser

# Critic Chain

critic_prompt = ChatPromptTemplate.from_messages([
    ('system','You are a Sharp and constructive research critic .Be honest and specific.'),
    ('human','''Review the research report below and evaluate it strictly.
     Report:{report}
     respond in this exact format:
     Score:X/10
     Strengths:
     - ...
     - ...

     Area to Improve:
     - ...
     - ...
     
     One line verdict:
     - ...
     - ...     
     ''')
])

critic_chain = critic_prompt | llm | parser 