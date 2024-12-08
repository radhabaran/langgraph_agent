# Enhancing the Chatbot with Tools

# To handle queries our chatbot can't answer "from memory", we'll integrate a web search tool. 
# Our bot can use this tool to find relevant information and provide better responses.

from typing import Annotated, Literal
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import ToolMessage
from IPython.display import Image, display
import os
import json

from dotenv import load_dotenv

load_dotenv()

api_key = os.environ['OA_API']           
os.environ['OPENAI_API_KEY'] = api_key

tavily_api_key = os.environ['TAVILY_KEY']           
os.environ["TAVILY_API_KEY"] = tavily_api_key

# Define the tool
tool = TavilySearchResults(max_results=2)
tools = [tool]


class State(TypedDict):
    messages : Annotated[list, add_messages]


graph_builder = StateGraph(State)

llm = ChatOpenAI(model='gpt-4o-mini')
# Modification: tell the LLM which tools it can call

llm_with_tools = llm.bind_tools(tools)


def chatbot(state: State):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}


graph_builder.add_node("chatbot", chatbot)

# Next we need to create a function to actually run the tools if they are called. 
# We'll do this by adding the tools to a new node.
#
# Below, we implement a BasicToolNode that checks the most recent message in the state and 
# calls tools if the message contains tool_calls.
# It relies on the LLM's tool_calling support, which is available in 
# Anthropic, OpenAI, Google Gemini, and a number of other LLM providers.

class BasicToolNode:
    """A node that runs the tools requested in the last AIMessage"""

    def __init__(self, tools: list) -> None:
        self.tools_by_name = {tool.name: tool for tool in tools}


    def __call__(self, inputs: dict):
        if messages := inputs.get("messages", []):
            message = messages[-1]
        else:
            raise ValueError("No message found in input")
        outputs = []
        for tool_call in message.tool_calls:
            tool_result = self.tools_by_name[tool_call["name"]].invoke(
                tool_call["args"]
            )
            outputs.append(
                ToolMessage(
                    content=json.dumps(tool_result),
                    name=tool_call["name"],
                    tool_call_id=tool_call["id"],
                )
            )
        return {"messages": outputs}


tool_node = BasicToolNode(tools=[tool])
graph_builder.add_node("tools", tool_node)


# with the tool added, we can define the coditional_edges.
# The edges route the control flow from one node to the next. 
# Conditional edges usually contain "if" statements to route to different nodes depnending on the current
# graph state. These functions receive the current graph state and return a string or list of strings 
# indicating which node(s) to call next.
#
# Define a router function called "route_tools", that checks for tool_calls in the chatbot's output. 
# Provide this function to the graph by calling add_conditional_edges, which tells the graph that whenever
# the chatbot node completes to check this function to see where to go next.
# The condition will route to tools if tool calls are present and END if not.


def route_tools(
    state: State,
):
    """
    Use in the conditional_edge to route to the ToolNode if the last message
    has tool calls. Otherwise, route to the end.
    """
    if isinstance(state, list):
        ai_message = state[-1]
    elif messages := state.get("messages", []):
        ai_message = messages[-1]
    else:
        raise ValueError(f"No messages found in input state to tool_edge: {state}")
    if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
        return "tools"
    return END


# The `tools_condition` function returns "tools" if the chatbot asks to use a tool, and "END" if
# it is fine directly responding. This conditional routing defines the main agent loop.
graph_builder.add_conditional_edges(
    "chatbot",
    route_tools,
    # The following dictionary lets you tell the graph to interpret the condition's outputs as a specific node
    # It defaults to the identity function, but if you
    # want to use a node named something else apart from "tools",
    # You can update the value of the dictionary to something else
    # e.g., "tools": "my_tools"
    {"tools": "tools", END: END},
)


# Any time a tool is called, we return to the chatbot to decide the next step
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge(START, "chatbot")
graph = graph_builder.compile()


# Notice that conditional edges start from a single node. This tells the graph "any time 
# the 'chatbot' node runs, either go to 'tools' if it calls a tool, or end the loop if it responds directly.

# You can visualize the graph using the get_graph method and one of the "draw" methods, 
# like draw_ascii or draw_png

try:
    png_data = graph.get_graph().draw_mermaid_png()
    with open("bot_tools_graph.png", "wb") as f:
        f.write(png_data)
except Exception:
    pass


# Now let's run the chatbot!
def stream_graph_updates(user_input: str):
    for event in graph.stream({"messages": [("user", user_input)]}):
        for value in event.values():
            print("Assistant:", value["messages"][-1].content)

            
while True:
    try:
        user_input = input("User: ")
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break

        stream_graph_updates(user_input)
    except:
        # fallback if input() is not available
        user_input = "What do you know about LangGraph?"
        print("User: " + user_input)
        stream_graph_updates(user_input)
        break
