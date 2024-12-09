# ***********************************************************************************************
# Human-in-the-loop
# Agents can be unreliable and may need human input to successfully accomplish tasks. 
# Similarly, for some actions, you may want to require human approval before running to 
# ensure that everything is running as intended. 
# 
# LangGraph supports human-in-the-loop workflows in a number of ways. In this section, 
# we will use LangGraph's interrupt_before functionality to always break the tool node.
# ***********************************************************************************************
from typing import Annotated, Literal
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import BaseMessage, ToolMessage
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
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


memory = MemorySaver()
graph_builder = StateGraph(State)

llm = ChatOpenAI(model='gpt-4o-mini')
# Modification: tell the LLM which tools it can call

llm_with_tools = llm.bind_tools(tools)


def chatbot(state: State):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}


graph_builder.add_node("chatbot", chatbot)

tool_node = ToolNode(tools=[tool])
graph_builder.add_node("tools", tool_node)


# The `tools_condition` function returns "tools" if the chatbot asks to use a tool, and "END" if
# it is fine directly responding.
graph_builder.add_conditional_edges(
    "chatbot",
    tools_condition,
)

# Any time a tool is called, we return to the chatbot to decide the next step
graph_builder.add_edge("tools", "chatbot")
# graph_builder.set_entry_point("chatbot")
graph_builder.add_edge(START, "chatbot")

# Now, compile the graph, specifying to interrupt_before the tools node.
graph = graph_builder.compile(checkpointer=memory, interrupt_before=["tools"])


# Notice that conditional edges start from a single node. This tells the graph "any time 
# the 'chatbot' node runs, either go to 'tools' if it calls a tool, or end the loop if it responds directly.

# You can visualize the graph using the get_graph method and one of the "draw" methods, 
# like draw_ascii or draw_png

try:
    png_data = graph.get_graph().draw_mermaid_png()
    with open("bot_memory_graph.png", "wb") as f:
        f.write(png_data)
except Exception:
    pass


# Now we can interact with your bot! First, pick a thread to use as the key for this conversation.
config = {"configurable": {"thread_id": "1"}}

def stream_graph_updates(user_input: str):
    events = graph.stream(
        {"messages": [("user", user_input)]}, 
        config, 
        stream_mode="values"
    )
    for event in events:
        if "messages" in event:
            event["messages"][-1].pretty_print()

print("Chat with the AI (type 'quit', 'exit', or 'q' to end)")


while True:
    try:
        user_input = input("\nUser: ").strip()
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break
        if not user_input:
            continue
            
        stream_graph_updates(user_input)

    
        # Continue the conversation if there are pending actions
        events = graph.stream(None, config, stream_mode="values")
        for event in events:
            if "messages" in event:
                event["messages"][-1].pretty_print()
                
    except KeyboardInterrupt:
        print("\nGoodbye!")
        break
    except Exception as e:
        print(f"An error occurred: {str(e)}")

# user_input = "I'm learning LangGraph. Could you do some research on it for me?"
# events = graph.stream(
#     {"messages": [("user", user_input)]}, config, stream_mode="values"
# )

# for event in events:
#     if "messages" in event:
#         event["messages"][-1].pretty_print()

# snapshot = graph.get_state(config)
# print(snapshot.next)

# existing_message = snapshot.values["messages"][-1]
# print(existing_message.tool_calls)

# events = graph.stream(None, config, stream_mode="values")
# for event in events:
#     if "messages" in event:
#         event["messages"][-1].pretty_print()

