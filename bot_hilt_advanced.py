# ***********************************************************************************************
# Customizing State
# If you want to define complex behavior without relying on the message list, you can add 
# additional fields to the state. In this program, we will extend our chat bot with a new node
# to illustrate this.
# 
# One way to do this is to create a passthrough "human" node, before which the graph will 
# always stop. We will only execute this node if the LLM invokes a "human" tool. 
# For our convenience, we will include an "ask_human" flag in our graph state that we will 
# flip if the LLM calls this tool.
# ***********************************************************************************************
from typing import Annotated, Literal
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import BaseMessage, ToolMessage, AIMessage
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel
from IPython.display import Image, display
import os
import json

from dotenv import load_dotenv

load_dotenv()

api_key = os.environ['OA_API']           
os.environ['OPENAI_API_KEY'] = api_key

anthro_api_key = os.environ['ANTHRO_KEY']           
os.environ['ANTHROPIC_API_KEY'] = anthro_api_key

tavily_api_key = os.environ['TAVILY_KEY']           
os.environ["TAVILY_API_KEY"] = tavily_api_key

# define a schema to show the model to let it decide to request assistance.
class RequestAssistance(BaseModel):
    """Escalate the conversation to an expert. Use this if you are unable to assist directly or /
    if the user requires support beyond your permissions.

    To use this function, relay the user's 'request' so the expert can provide the right guidance.
    """

    request: str


# Define the tool
tool = TavilySearchResults(max_results=2)
tools = [tool]


class State(TypedDict):
    messages: Annotated[list, add_messages]
    ask_human: bool


memory = MemorySaver()

llm = ChatAnthropic(model="claude-3-5-sonnet-20240620")
# llm = ChatOpenAI(model='gpt-3.5-turbo')

# Modification: tell the LLM which tools it can call
# We can bind the llm to a tool definition, a pydantic model, or a json schema

llm_with_tools = llm.bind_tools(tools + [RequestAssistance])

# Next, define the chatbot node. The primary modification here is flip the ask_human flag 
# if we see that the chat bot has invoked the RequestAssistance flag.


def chatbot(state: State):
    response = llm_with_tools.invoke(state["messages"])
    ask_human = False
    if (
        response.tool_calls
        and response.tool_calls[0]["name"] == RequestAssistance.__name__
    ):
        ask_human = True
    return {"messages": [response], "ask_human": ask_human}


graph_builder = StateGraph(State)
graph_builder.add_node("chatbot", chatbot)

tool_node = ToolNode(tools=[tool])
graph_builder.add_node("tools", tool_node)

# Next, create the "human" node. This node function is mostly a placeholder in our graph that will
# trigger an interrupt. If the human does not manually update the state during the interrupt, it 
# inserts a tool message so the LLM knows the user was requested but didn't respond. This node also 
# unsets the ask_human flag so the graph knows not to revisit the node unless further requests are made.


def create_response(response: str, ai_message: AIMessage):
    return ToolMessage(
        content=response,
        tool_call_id=ai_message.tool_calls[0]["id"],
    )


def human_node(state: State):
    new_messages = []
    if not isinstance(state["messages"][-1], ToolMessage):
        # Typically, the user will have updated the state during the interrupt.
        # If they choose not to, we will include a placeholder ToolMessage to
        # let the LLM continue.
        new_messages.append(
            create_response("No response from human.", state["messages"][-1])
        )
    return {
        # Append the new messages
        "messages": new_messages,
        # Unset the flag
        "ask_human": False,
    }


graph_builder.add_node("human", human_node)

def select_next_node(state: State):
    if state["ask_human"]:
        return "human"
    # Otherwise, we can route as before
    return tools_condition(state)


graph_builder.add_conditional_edges(
    "chatbot",
    select_next_node,
    {"human": "human", "tools": "tools", END: END},
)


# The `tools_condition` function returns "tools" if the chatbot asks to use a tool, and "END" if
# it is fine directly responding.
graph_builder.add_conditional_edges(
    "chatbot",
    tools_condition,
)

# Any time a tool is called, we return to the chatbot to decide the next step
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge("human", "chatbot")
graph_builder.add_edge(START, "chatbot")

memory = MemorySaver()

# Now, compile the graph, specifying to interrupt_before the tools node.
graph = graph_builder.compile(checkpointer=memory, interrupt_before=["human"])


# Notice that conditional edges start from a single node. This tells the graph "any time 
# the 'chatbot' node runs, either go to 'tools' if it calls a tool, or end the loop if it responds directly.

# You can visualize the graph using the get_graph method and one of the "draw" methods, 
# like draw_ascii or draw_png

try:
    png_data = graph.get_graph().draw_mermaid_png()
    with open("bot_hilt_advanced_graph.png", "wb") as f:
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
        snapshot = graph.get_state(config)

        ai_message = snapshot.values["messages"][-1]
        # human_response = (
        #     "We, the experts are here to help! We'd recommend you check out LangGraph to build your agent."
        #     " It's much more reliable and extensible than simple autonomous agents."
        # )

        # Check if user is satisfied with AI response
        print("\nAre you satisfied with the response? (yes/no):")
        satisfaction = input().strip().lower()

        if satisfaction in ['no', 'n']:
            print("\nConsulting experts...")
            print("Expert (type your response):")
            human_response = input().strip()

            # If no response is provided, use a default message
            if not human_response:
                human_response = "No response from human expert."

            # tool_message = create_response(human_response, ai_message)
            # graph.update_state(config, {"messages": [tool_message]})
    
            # Update state with the expert response
            graph.update_state(config, {"messages": [("user", f"Expert Response: {human_response}")]})

            # Continue the conversation if there are pending actions
            # events = graph.stream(None, config, stream_mode="values")

            # Continue conversation with updated state
            events = graph.stream(
                {"messages": graph.get_state(config).values["messages"]},
                    config,
                    stream_mode="values"
            )

            for event in events:
                if "messages" in event:
                    event["messages"][-1].pretty_print()
        else:
            print("Great! Let's continue with your next question.")
                
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

