# Build a basic chatbot using LangGraph

from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from IPython.display import Image, display
import os

from dotenv import load_dotenv



load_dotenv()


# ********************************************************************************************************
# When defining a graph, the first step is to define its State. The State includes the graph's schema and
# reducer functions that handle state updates. 
# 
# In our example, State is a TypedDict with one key: messages. 
# The add_messages reducer function is used to append new messages to the list instead of 
# overwriting it. Keys without a reducer annotation will overwrite previous values. 
# 
# ********************************************************************************************************

class State(TypedDict):
    messages : Annotated[list, add_messages]


graph_builder = StateGraph(State)

# ********************************************************************************************************
# Our graph can now handle two key tasks:
#
# 1. Each node can receive the current State as input and output an update to the state.
# 2. Updates to messages will be appended to the existing list rather than overwriting it, 
#    thanks to the prebuilt add_messages function used with the Annotated syntax.
# ********************************************************************************************************

# Next, add a "chatbot" node. Nodes represent units of work. They are typically regular python functions.
api_key = os.environ['OA_API']           
os.environ['OPENAI_API_KEY'] = api_key

llm = ChatOpenAI(model='gpt-4o-mini')


def chatbot(state: State):
    return {'messages': [llm.invoke(state['messages'])]}


# Add the node
# Nodes represent units of work. They are typically regular python functions.
# The first argument is the unique node name and the second argument is the function or object
# that will be called whenever the node is used..

graph_builder.add_node("chatbot", chatbot)

# Next, add an entry point. This tells our graph where to start its work each time we run it.
graph_builder.add_edge(START, "chatbot")

# Similarly, set a finish point. This instructs the graph "any time this node is run, you can exit."
graph_builder.add_edge("chatbot", END)

# Finally, we'll want to be able to run our graph. To do so, call "compile()" on the graph builder.
# This creates a "CompiledGraph" we can use invoke on our state.
graph = graph_builder.compile()

# You can visualize the graph using the get_graph method and one of the "draw" methods, 
# like draw_ascii or draw_png

try:
    png_data = graph.get_graph().draw_mermaid_png()
    with open("basic_bot_graph.png", "wb") as f:
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





