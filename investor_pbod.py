import os
from dotenv import load_dotenv
from typing import TypedDict

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, END

load_dotenv()
llm = ChatOpenAI(model="gpt-4.1", temperature=0.7)

class InvestorState(TypedDict):
    user_input: str
    strategist_response: str
    technical_response: str
    risk_response: str
    platform_response: str
    discussion: str
    synthesizer_response: str

def make_agent(name, system_prompt):
    def agent(state: InvestorState):
        context = state.get("discussion", "")
        messages = [
            SystemMessage(content=system_prompt),
            AIMessage(content=f"Discussion so far: {context}"),
            HumanMessage(content=state["user_input"])
        ]
        response = llm.invoke(messages)
        return {
            f"{name}_response": response.content,
            "discussion": context + f"\n{name.capitalize()}: {response.content}"
        }
    return agent

# Agents with more conversational roles
strategist = make_agent("strategist", "You're a Market Strategist. Discuss macroeconomic trends with the team and explore sectoral opportunities collaboratively.")
technical = make_agent("technical", "You're a Technical Analyst. Join the conversation by analyzing patterns, and comment on others' views where appropriate.")
risk = make_agent("risk", "You're a Risk Manager. Discuss risk exposure, portfolio balance, and counterpoints with other team members.")
platform = make_agent(
    "platform",
    "You're a Platform Specialist. Join the conversation and suggest suitable instruments and strategies available via modern online investing platforms."
)

def synthesizer(state: InvestorState):
    messages = [
        SystemMessage(content="You're the Chairperson of the Investment Board. Based on the discussion and everyone's input, summarize the consensus and give a final recommendation."),
        AIMessage(content=state["discussion"]),
        HumanMessage(content=state["user_input"])
    ]
    response = llm.invoke(messages)
    return {"synthesizer_response": response.content}

# Build the LangGraph
graph = StateGraph(state_schema=InvestorState)
graph.add_node("Strategist", strategist)
graph.add_node("Technical", technical)
graph.add_node("Risk", risk)
graph.add_node("Platform", platform)
graph.add_node("Synthesizer", synthesizer)

graph.set_entry_point("Strategist")
graph.add_edge("Strategist", "Technical")
graph.add_edge("Technical", "Risk")
graph.add_edge("Risk", "Platform")
graph.add_edge("Platform", "Synthesizer")
graph.add_edge("Synthesizer", END)

board = graph.compile()

if __name__ == "__main__":
    print("\U0001F4C8 Welcome to your Personal Investment Board Meeting (AI Edition)")
    user_input = input("User: What’s your investment decision/question today?\n> ")

    result = board.invoke({"user_input": user_input, "discussion": ""})

    print("\n--- Investment Board Report ---")
    print("\nDiscussion Transcript\n----------------------")
    print(result.get("discussion", ""))

    print("\nFinal Recommendation\n--------------------")
    print(result.get("synthesizer_response", "No conclusion reached."))