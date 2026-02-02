import os
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph.message import add_messages
from  langgraph.checkpoint.memory import InMemorySaver
from dotenv import load_dotenv
load_dotenv()
from langchain_groq import ChatGroq

groq_llm = ChatGroq(
    model="llama-3.1-8b-instant", 
    temperature=0.7, 
    api_key=os.environ.get("GROQ_API_KEY"),
    base_url="https://api.groq.com",
)
class ChatState(TypedDict):
    messages:Annotated[list[BaseMessage],add_messages]
def Chat_node(state:ChatState):
    messages=state['messages']
    response=groq_llm.invoke(messages)
    return{'messages':[response]}

checkpointer=InMemorySaver()
graph=StateGraph(ChatState)
graph.add_node('Chat_node',Chat_node)
graph.add_edge(START,'Chat_node')
graph.add_edge('Chat_node',END)
chatbot=graph.compile(checkpointer=checkpointer)