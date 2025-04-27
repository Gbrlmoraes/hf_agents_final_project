### Imports

# Agent Building
from langgraph.graph import START, END, StateGraph, MessagesState
from langgraph.prebuilt import ToolNode
from langchain_core.messages import HumanMessage, SystemMessage

# AI Models
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI

# Tools
from langchain_tavily import TavilySearch
from tools import YoutubeQueryTool, ReadDocumentsTool

# Misc
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import Literal

load_dotenv()

### Agent Building


# Structured Output
class FinalResponseOutput(BaseModel):
    """Respond to the user with this"""

    final_reponse: str = Field(
        description="your answer with the following template: FINAL ANSWER: [YOUR FINAL ANSWER]. YOUR FINAL ANSWER should be a number OR as few words as possible OR a comma separated list of numbers and/or strings."
    )
    reasoning_trace: str = Field("The different steps by which you reached answer")


# Agent State
class AgentState(MessagesState):
    final_response: FinalResponseOutput


# Agent Class
class GAIAAgent:
    def __init__(self):
        # Define the tools
        self.tools = [TavilySearch(max_results=5), YoutubeQueryTool, ReadDocumentsTool]

        # Define LLM Models
        llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)
        # llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0)
        llm_with_tools = llm.bind_tools(self.tools)
        llm_with_structured_output = llm.with_structured_output(
            FinalResponseOutput, method="function_calling"
        )

        # Define the system message
        sys_msg = SystemMessage(
            content="You are a general AI assistant. I will ask you a question. Report your thoughts, and finish your answer with the following template: FINAL ANSWER: [YOUR FINAL ANSWER]. YOUR FINAL ANSWER should be a number OR as few words as possible OR a comma separated list of numbers and/or strings. If you are asked for a number, don't use comma to write your number neither use units such as $ or percent sign unless specified otherwise. If you are asked for a string, don't use articles, neither abbreviations (e.g. for cities), and write the digits in plain text unless specified otherwise. If you are asked for a comma separated list, apply the above rules depending of whether the element to be put in the list is a number or a string."
        )

        ### Nodes definition

        # Main function of the agent
        def call_model(state: AgentState):
            response = llm_with_tools.invoke(state["messages"])
            # We return a list, because this will get added to the existing list
            return {"messages": [response]}

        # Define the function that responds to the user
        def respond(state: AgentState):
            response = llm_with_structured_output.invoke([sys_msg] + state["messages"])
            # We return the final answer
            return {"final_response": response}

        ### Edges definition

        # Define the function that determines whether to continue or not
        def should_continue(state: AgentState) -> Literal["tools", "respond"]:
            messages = state["messages"]
            last_message = messages[-1]
            # If there is no function call, then we respond to the user
            if not last_message.tool_calls:
                return "respond"
            # Otherwise if there is, we continue
            else:
                return "tools"

        # Define a new graph
        workflow = StateGraph(AgentState)

        ### Nodes
        workflow.add_node("agent", call_model)
        workflow.add_node("respond", respond)
        workflow.add_node("tools", ToolNode(self.tools))

        ### Edges
        workflow.add_edge(START, "agent")

        # We now add a conditional edge
        workflow.add_conditional_edges(
            "agent",
            should_continue,
        )

        workflow.add_edge("tools", "agent")
        workflow.add_edge("respond", END)

        # Compile the graph and assign it to the instance
        self.graph = workflow.compile()
        # print("AI Agent initialized.")

    def __call__(self, question: str) -> str:
        # print(f"Agent received question (first 100 chars): {question[:100]}...")
        # Prepare the initial state for the graph
        initial_state = {"messages": [HumanMessage(content=question)]}
        # Invoke the graph
        final_state = self.graph.invoke(initial_state)
        # Extract the final response
        final_answer = final_state["final_response"].final_reponse
        # print(f"Agent returning final answer: {final_answer}")

        return final_answer


### Testing the agent
if __name__ == "__main__":
    question = "How many studio albums were published by Mercedes Sosa between 2000 and 2009 (included)? You can use the latest 2022 version of english wikipedia."

    # Run the agent with the test case question
    gaia_agent = GAIAAgent()
    response = gaia_agent(question)
    print(f"Final Answer: {response}\n")
