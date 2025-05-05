# run_agent.py

from langchain.agents import initialize_agent
from langchain.agents.agent_types import AgentType
from langchain_community.llms import Ollama
from langchain_tool import preprocessing_tool

# Initialize Ollama with LLaMA 3
llm = Ollama(model="llama3.2:latest")

# Define tools
tools = [preprocessing_tool]

# Create the agent
agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# Example run (file should be in the same directory or provide path)
response = agent.run("Preprocess the file exam_copy.pdf for OCR.")
print(response)
