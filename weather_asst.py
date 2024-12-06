# Weather checking assistant
# Let’s build a simple weather-checking assistant using LangGraph. Our assistant will:
#
# 1. Greet the user
#
# 2. Ask for their location
#
# 3. Check the weather (simulated for this example)
#
# 4. Provide a weather report
# **********************************************************************************************************
from langchain_core.tools import tool
from typing import TypedDict, Annotated, List
from langgraph.graph import END, StateGraph
import os
import operator
from dotenv import load_dotenv


load_dotenv()

# We define a class named AgentState that tracks the conversation's current status.
class AgentState(TypedDict):
    messages: Annotated[list, operator.add]
    location: str
    weather: str

# Create our graph
workflow = StateGraph(AgentState)


# Define our nodes
def greet(AgentState):
    return {"messages": [("ai", "Hello! I'm your weather assistant. Where are you located?")]}


def get_location(AgentState):
    return {"location": AgentState["messages"][-1][1]}


def check_weather(AgentState):
    # In a real app, we'd call a weather API here
    weather = "sunny" if "new york" in AgentState["location"].lower() else "rainy"
    return {"weather": weather}


def report_weather(AgentState):
    return {"messages": [
        ("ai", f"The weather in {AgentState['location']} is {AgentState['weather']}. Can I help you with anything else?")]}


# Add nodes to our graph
workflow.add_node("greet", greet)
workflow.add_node("get_location", get_location)
workflow.add_node("check_weather", check_weather)
workflow.add_node("report_weather", report_weather)

# Connect our nodes
workflow.set_entry_point("greet")
workflow.add_edge("greet", "get_location")
workflow.add_edge("get_location", "check_weather")
workflow.add_edge("check_weather", "report_weather")
workflow.add_edge("report_weather", END)


# Compile our graph
app = workflow.compile()


# Run our app
inputs = {"messages": [("human", "Hi, I'd like to check the weather.")]}
for output in app.stream(inputs):
    for key, value in output.items():
        print(f"{key}: {value}")