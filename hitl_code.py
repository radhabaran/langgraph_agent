from langchain_core.tools import tool
from typing import TypedDict, Annotated, List, Union
from langchain_experimental.utilities import PythonREPL
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_openai import OpenAI
from langchain_core.agents import AgentAction, AgentFinish
from langchain.agents import create_json_chat_agent
from langgraph.prebuilt.tool_executor import ToolExecutor
from langgraph.graph import END, StateGraph
from langchain_core.prompts import (
     ChatPromptTemplate,
     SystemMessagePromptTemplate,
     PromptTemplate, 
     MessagesPlaceholder, 
     HumanMessagePromptTemplate )
from langchain_core.messages import BaseMessage
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    ChatMessage,
    SystemMessage,
    FunctionMessage,
    ToolMessage
)
import json
import os
import operator

# **************************************************************************************************
# We define a class named AgentState that tracks the conversation's current status. This state
# includes the latest user message, the entire chat history, what the agent plans to do next 
# (like searching for something), and whether the conversation is ending. It also keeps a log of 
# the agent's actions throughout the conversation to understand its decision process.
# **************************************************************************************************

class AgentState(TypedDict):
   input: str
   chat_history: list[BaseMessage]
   agent_outcome: Union[AgentAction, AgentFinish, None]
   intermediate_steps: Annotated[list[tuple[AgentAction, str]], operator.add]


tavily_api_key = os.environ['tavily_key']           
os.environ["TAVILY_API_KEY"] = tavily_api_key

search = TavilySearchResults(max_results=1)

repl = PythonREPL()

api_key = os.environ['OA_API']           
os.environ['OPENAI_API_KEY'] = api_key

chat_model = OpenAI(openai_api_key=api_key)


# **************************************************************************************************
# We define a function called extract_json, which takes a single parameter, text. 
# it searches for the index where the JSON part of the text begins by locating the substring “AI:”.
#
# If “AI:” is found, the function proceeds to pull the JSON portion of the text, stripping any 
# leading or trailing whitespace.
#
# Following this, it tries to parse the extracted JSON string into a Python object using the 
# JSON.loads() function.
# **************************************************************************************************

def extract_json(text):
    try:
        # Find the index where the JSON part starts (after "AI:")
        start_index = text.find("AI:")
        if start_index != -1:
            # Extract the JSON part of the text
            json_str = text[start_index + len("AI:"):].strip()
            # Parse the JSON object
            json_obj = json.loads(json_str)
            return json_obj
        else:
            # If "AI:" prefix not found, return None
            return None
    except json.JSONDecodeError as e:
        # Handle JSON decoding errors
        print(f"Error decoding JSON: {e}")
        return None
    except Exception as e:
        # Handle other unexpected errors
        print(f"Unexpected error: {e}")
        return None


# **************************************************************************************************
# Then we set up Tools. 
# One is Tavily, a search engine, where the agent can find information.to ensure only the top 
# search result is retrieved from Tavily.
#
# we set up a function named python_repl Imagine you have a tool as a mini-computer inside 
# the chatbot. You can give this mini-computer Python code, and it will run the code for you.
# **************************************************************************************************


@tool
def python_repl(
    code: Annotated[str, "The python code to execute to generate your chart."]
):
    """Use this to execute python code. If you want to see the output of a value,
    you should print it out with `print(...)`. This is visible to the user."""
    try:
        result = repl.run(code)
    except BaseException as e:
        return f"Failed to execute. Error: {repr(e)}"
    return f"Succesfully executed:\n```python\n{code}\n```\nStdout: {result}"

tools = [search, python_repl]


# **************************************************************************************************
# Defines a ChatPromptTemplate called ‘prompt2’ with input variables 
# ‘agent_scratchpad’, ‘input’, ‘tool_names’, and ‘tools’.
# It specifies the input types for ‘chat_history’ and ‘agent_scratchpad’ as lists of various 
# message types (AIMessage, HumanMessage, ChatMessage, SystemMessage, FunctionMessage, ToolMessage).
#
# The template contains multiple messages: a SystemMessagePromptTemplate, 
# a MessagesPlaceholder for ‘chat_history’, a HumanMessagePromptTemplate, 
# and another MessagesPlaceholder for ‘agent_scratchpad’.
#
# The SystemMessagePromptTemplate describes the assistant’s capabilities as a large language model, 
# capable of assisting with various tasks and constantly learning and improving.
#
# The HumanMessagePromptTemplate includes instructions for the response format, which can be 
# either a JSON code snippet indicating the use of a tool or a final answer in JSON format.
#
# The template also includes placeholders for the user’s input and the available tools.
#
# After defining the prompt template, the code creates a JSON chat agent using the 
# ‘create_json_chat_agent’ function from the langchain library, passing the chat model, 
# tools, and the defined ‘prompt2’ template.
#
# Finally, a ToolExecutor instance is created, passing the available tools.
# **************************************************************************************************
prompt2 = ChatPromptTemplate(
    input_variables=["agent_scratchpad", "input", "tool_names", "tools"],
    input_types={
        "chat_history": List[
            Union[
                AIMessage,
                HumanMessage,
                ChatMessage,
                SystemMessage,
                FunctionMessage,
                ToolMessage,
            ]
        ],
        "agent_scratchpad": List[
            Union[
                AIMessage,
                HumanMessage,
                ChatMessage,
                SystemMessage,
                FunctionMessage,
                ToolMessage,
            ]
        ],
    },
    messages=[
        SystemMessagePromptTemplate(
            prompt=PromptTemplate(
                input_variables=[],
                template="""Assistant is a large language model trained by OpenAI.\n\nAssistant is designed to be able to assist with a wide range 
                         of tasks, from answering simple questions to providing in-depth explanations and discussions on a wide range of topics. 
                         As a language model, Assistant is able to generate human-like text based on the input it receives, allowing it to engage 
                         in natural-sounding conversations and provide responses that are coherent and relevant to the topic at hand.
                         \n\nAssistant is constantly learning and improving, and its capabilities are constantly evolving. It is able to process 
                         and understand large amounts of text, and can use this knowledge to provide accurate and informative responses to a wide 
                         range of questions. Additionally, Assistant is able to generate its own text based on the input it receives, allowing it 
                         to engage in discussions and provide explanations and descriptions on a wide range of topics.
                         \n\nOverall, Assistant is a powerful system that can help with a wide range of tasks and provide valuable insights and 
                         information on a wide range of topics. Whether you need help with a specific question or just want to have a conversation
                         about a particular topic, Assistant is here to assist.""",
            )
        ),
        MessagesPlaceholder(variable_name="chat_history", optional=True),
        HumanMessagePromptTemplate(
            prompt=PromptTemplate(
                input_variables=["input", "tool_names", "tools"],
                template='TOOLS\n------\nAssistant can ask the user to use tools to look up information that may be helpful in answering the users original question. 
The tools the human can use are:\n\n{tools}\n\nRESPONSE FORMAT INSTRUCTIONS\n----------------------------\n\nWhen responding to me, please output a response in one of two formats:\n\n**Option 1:**\nUse this if you want the human to use a tool.\nMarkdown code snippet formatted in the following schema:\n\n```json\n{{\n    "action": string, \\ The action to take. Must be one of {tool_names}\n    "action_input": string \\ The input to the action\n}}\n```\n\n**Option #2:**\nUse this if you can respond directly to the human after tool execution. Markdown code snippet formatted in the following schema:\n\n```json\n{{\n    "action": "Final Answer",\n    "action_input": string \\ You should put what you want to return to use here\n}}\n```\n\nUSER\'S INPUT\n--------------------\nHere is the user\'s input (remember to respond with a markdown code snippet of a json blob with a single action, and NOTHING else):\n\n{input}',

            )
        ),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ],
)

agent_runnable = create_json_chat_agent(chat_model, tools, prompt2)

tool_executor = ToolExecutor(tools)


# **************************************************************************************************
# The run_agent function takes data as an argument and invokes the agent_runnable with the given 
# data.
# It returns a dictionary containing the agent_outcome from the invocation.
# **************************************************************************************************

def run_agent(data):
    agent_outcome = agent_runnable.invoke(data)
    return {"agent_outcome": agent_outcome}


# **************************************************************************************************
# The execute_tools function also takes data as an argument. It extracts the agent_action and 
# tool_input from the agent_outcome in the data.
# It then prompts the user with a confirmation message, asking whether to continue with the 
# specified tool and tool_input. If the user responds with "n" (no), a ValueError is raised.
# 
# If the user confirms (or if the confirmation step is skipped), it invokes the tool_executor 
# (created earlier) with the agent_action. The output of the tool execution is appended to the 
# intermediate_steps list in the data dictionary.
# **************************************************************************************************

def execute_tools(data):
    agent_action = data['agent_outcome']
    tool = agent_action.tool
    tool_input = agent_action.tool_input

    # Human in the Loop: Confirm tool execution with the user
    response = input(f"[y/n] continue with: {tool} with {tool_input}?")
    if response == "n":
        raise ValueError

    output = tool_executor.invoke(agent_action)
    data["intermediate_steps"].append((agent_action, str(output)))
    return data

# **************************************************************************************************
# The should_continue function is defined to determine the next edge to follow in conditional 
# branching. It takes data as an argument.
#
# If the agent_outcome in the data is an instance of AgentFinish, the function returns the string 
# "end", indicating that the branching should end.
#
# Otherwise, if the agent_outcome is not an instance of AgentFinish, the function returns the 
# string "continue", indicating that the branching should continue.
# **************************************************************************************************

def should_continue(data):
    if isinstance(data['agent_outcome'], AgentFinish):
        return "end"
    else:
        return "continue"
    

# **************************************************************************************************
# Sets up a state graph workflow with two nodes (‘agent’ and ‘action’), defines conditional 
# branching based on the ‘should_continue’ function, compiles the graph into an executable 
# application, provides an input dictionary with a query, invokes the application with the input, 
# and prints the final output from the agent’s outcome.
# **************************************************************************************************

# Define a new graph
workflow = StateGraph(AgentState)

# Define two nodes for looping
workflow.add_node("agent", run_agent)
workflow.add_node("action", execute_tools)

# Set entry point
workflow.set_entry_point("agent")

# Add conditional edges
workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        "continue": "action",
        "end": END,
    },
)

workflow.add_edge("action", "agent")

app = workflow.compile()

inputs = {"input": "what is the weather in taiwan", "chat_history": []}
result = app.invoke(inputs)

print(result["agent_outcome"].return_values["output"])