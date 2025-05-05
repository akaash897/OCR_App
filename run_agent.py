# run_agent.py

from langchain.agents import initialize_agent
from langchain.agents.agent_types import AgentType
from langchain.llms import OpenAI
from langchain.tools import Tool
from langchain_tool import preprocessing_tool

llm = OpenAI(temperature=0)
tools = [preprocessing_tool]

agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# Example run:
response = agent.run("Preprocess the file exam_copy.pdf for OCR.")
print(response)
