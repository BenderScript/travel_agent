import logging

from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent
from langchain.agents.output_parsers import ReActJsonSingleInputOutputParser
from langchain.tools.render import render_text_description_and_args
from langchain_openai import ChatOpenAI
from langchain_community.agent_toolkits.amadeus.toolkit import AmadeusToolkit
from dotenv import load_dotenv, find_dotenv

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# https://python.langchain.com/v0.2/docs/integrations/toolkits/amadeus/
# https://medium.com/@shanakachathuranga/utilizing-the-amadeus-toolkit-with-langchain-agents-for-seamless-flight-queries-e5e407292125


# Load the .env file
path = find_dotenv()
logging.info(path)
load_dotenv(override=True, verbose=True)

llm = ChatOpenAI(model="gpt-4")
toolkit = AmadeusToolkit(llm=llm)
tools = toolkit.get_tools()

llm = ChatOpenAI(temperature=0, model='gpt-4')

prompt = hub.pull("hwchase17/react-json")
agent = create_react_agent(
    llm,
    tools,
    prompt,
    tools_renderer=render_text_description_and_args,
    output_parser=ReActJsonSingleInputOutputParser(),
)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True
)

while True:
    user_input = input("User: ")
    if user_input.lower() in ["quit", "exit", "q"]:
        print("Goodbye!")
        break
    answer = agent_executor.invoke({"input": user_input})
    print("Assistant:", answer)
