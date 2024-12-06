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
    location: str | None
    weather: str | None

def greet(state: AgentState) -> dict:
    print("Executing greet...")  # Debug print
    return {"messages": [("ai", "Hello! I'm your weather assistant. Where are you located?")]}

def get_location(state: AgentState) -> dict:
    print("Executing get_location...")  # Debug print
    last_message = state['messages'][-1][1]
    return {"location": last_message}

def check_weather(state: AgentState) -> dict:
    print("Executing check_weather...")  # Debug print
    location = state.get("location", "").lower()
    weather = "sunny" if "new york" in location else "rainy"
    return {"weather": weather}

def report_weather(state: AgentState) -> dict:
    print("Executing report_weather...")  # Debug print
    return {
        "messages": [
            ("ai", f"The weather in {state['location']} is {state['weather']}. Can I help you with anything else?")
        ]
    }

# Create and configure workflow
workflow = StateGraph(AgentState)

workflow.add_node("greet", greet)
workflow.add_node("get_location", get_location)
workflow.add_node("check_weather", check_weather)
workflow.add_node("report_weather", report_weather)

workflow.set_entry_point("greet")
workflow.add_edge("greet", "get_location")
workflow.add_edge("get_location", "check_weather")
workflow.add_edge("check_weather", "report_weather")
workflow.add_edge("report_weather", END)

# Compile the app
app = workflow.compile()

# Run directly without the function wrapper
initial_state = {
    "messages": [("human", "Hi, I'd like to check the weather.")],
    "location": None,
    "weather": None
}

print("Starting the weather assistant...")
for output in app.stream(initial_state):
    print(f"Output received: {output}")  # Debug print
    if "messages" in output and output["messages"]:
        for sender, message in output["messages"]:
            print(f"{sender}: {message}")

location = input("You: ")

new_state = {
    "messages": [("human", location)],
    "location": None,
    "weather": None
}

for output in app.stream(new_state):
    print(f"Output received: {output}")  # Debug print
    if "messages" in output and output["messages"]:
        for sender, message in output["messages"]:
            print(f"{sender}: {message}")