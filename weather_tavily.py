from typing import Annotated, Literal, TypedDict, List
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import BaseMessage
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    ChatMessage,
    SystemMessage,
    FunctionMessage,
    ToolMessage
)
from langgraph.graph import END, START, StateGraph, MessagesState
from langgraph.prebuilt import ToolNode
import gradio as gr

import os
import operator
from dotenv import load_dotenv
import uuid


def create_chat_interface():
    load_dotenv()

    # anthro_api_key = os.environ['ANTHRO_KEY']           
    # os.environ["ANTHROPIC_API_KEY"] = anthro_api_key

    tavily_api_key = os.environ['TAVILY_KEY']           
    os.environ["TAVILY_API_KEY"] = tavily_api_key

    api_key = os.environ['OA_API']           
    os.environ['OPENAI_API_KEY'] = api_key

    # We define a class named AgentState that tracks the conversation's current status.
    # class AgentState(TypedDict):
    #     messages: Annotated[list[BaseMessage], operator.add]
    #     # chat_history: list[BaseMessage]
    #     chat_history: Annotated[list[BaseMessage], operator.add]


    # Define the tools for the agent to use
    @tool
    def search(query: str):
        """Call to surf the web."""
        search = TavilySearchResults(max_results=1)
        results = search.invoke(query)
        return results[0]['content'] if results else "No results found."

    tools = [search]
    tool_node = ToolNode(tools)

    # model = ChatAnthropic(model="claude-3-5-sonnet-20240620", temperature=0).bind_tools(tools)
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0).bind_tools(tools)


    # Define the function that determines whether to continue or not
    def should_continue(state: MessagesState) -> Literal["tools", END]:
        messages = state['messages']
        last_message = messages[-1]

        print("\n[should_continue] Current messages:")
        for msg in messages:
            print(f"- {msg.content}")

        # If the LLM makes a tool call, then we route to the "tools" node
        if last_message.tool_calls:
            return "tools"
        # Otherwise, we stop (reply to the user)
        return END


    # Define the function that calls the model
    def call_model(state: MessagesState):
        messages = state['messages']
    
        print("\n[call_model] Input messages:")
        for msg in messages:
            print(f"- {msg.content}")

        response = model.invoke(messages)

        # We return a list, because this will get added to the existing list
        return {"messages": [response]}


    # Define a new graph
    workflow = StateGraph(MessagesState)

    # Define the two nodes we will cycle between
    workflow.add_node("agent", call_model)
    workflow.add_node("tools", tool_node)

    # Set the entrypoint as `agent`
    # This means that this node is the first one called
    workflow.add_edge(START, "agent")

    # We now add a conditional edge
    workflow.add_conditional_edges(
        # First, we define the start node. We use `agent`.
        # This means these are the edges taken after the `agent` node is called.
        "agent",
        # Next, we pass in the function that will determine which node is called next.
        should_continue,
    )

    # We now add a normal edge from `tools` to `agent`.
    # This means that after `tools` is called, `agent` node is called next.
    workflow.add_edge("tools", 'agent')

    # Initialize memory to persist state between graph runs
    checkpointer = MemorySaver()

    # Finally, we compile it!
    # This compiles it into a LangChain Runnable,
    # meaning you can use it as you would any other runnable.
    # Note that we're (optionally) passing the memory when compiling the graph
    app = workflow.compile(checkpointer=checkpointer)


    def generate_thread_id():
        return str(uuid.uuid4())


    def process_message(message, history, thread_id):
        if not message:
            return "", history, thread_id
        
        if not thread_id:
            thread_id = generate_thread_id()

        # Use the Runnable
        final_state = app.invoke(
            {
                "messages": [HumanMessage(content= message)]
            },
            config={"configurable": {"thread_id": thread_id}}
        )

        # Let's add prints to see the message flow
        print("\nMessage Flow:")
        print(f"Input message: {message}")
        print(f"Final state messages: {[msg.content for msg in final_state['messages']]}")

        response = final_state["messages"][-1].content
        history.append((message, response))
        # print("history", history)
        
        return "", history, thread_id


    with gr.Blocks() as demo:
        thread_id = gr.State("")
        chatbot = gr.Chatbot(height=400)
        
        with gr.Row():
            txt = gr.Textbox(
                show_label=False,
                placeholder="Enter your message here...",
                container=False
            )
            submit_btn = gr.Button("Send")
        
        txt.submit(process_message, [txt, chatbot, thread_id], [txt, chatbot, thread_id])
        submit_btn.click(process_message, [txt, chatbot, thread_id], [txt, chatbot, thread_id])

    return demo


if __name__ == "__main__":
    demo = create_chat_interface()
    demo.launch(share=True)

