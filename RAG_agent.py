from langgraph.graph import StateGraph
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import create_react_agent
from langchain_tavily import TavilySearch
import regex as re
import json
import os
from operator import add as add_messages
import operator
from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, ToolMessage, AIMessage
from langchain_community.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.chat_models import ChatOllama  # ✅ Use Ollama LLM
from langchain.schema.runnable import Runnable
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.vectorstores.base import VectorStoreRetriever
from langchain_core.tools import tool

# Step 1: Initialize ChromaDB + retriever
embedding_model_name = "all-MiniLM-L6-v2"
persist_directory = "./my_company_chroma"
collection_name = "my_company_collection"

embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)
vectorstore = Chroma(
    collection_name=collection_name,
    embedding_function=embeddings,
    persist_directory=persist_directory
)
retriever: VectorStoreRetriever = vectorstore.as_retriever()

# ✅ Replace OpenAI with Ollama model

llm = ChatOllama(model="llama3.1:8b", temperature=0)

class AgentState(TypedDict):
    query: str
    summary: str
    company_url: str
    messages: Annotated[Sequence[BaseMessage], operator.add]


def agent(state: AgentState) -> str:
    """
    The RAG agent will answer the query based on the knolwedge provided.
    """

    prompt_template = """
    Use the following context to answer the question.
    If you don't know the answer, say you don't know.

    Context:
    {context}

    Question:
    {question}

    Answer:
    """
    prompt = PromptTemplate(input_variables=["context", "question"], template=prompt_template)

    rag_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff",
    chain_type_kwargs={"prompt": prompt}
    )

    answer = rag_chain.run(state["query"])

    return {
        "summary": state["messages"] + [answer]
    }


search_tool = TavilySearch(
    max_results=5,
    topic="general",
    tavily_api_key="<api_key>",
)

news_tool = TavilySearch(
    max_results=5,
    topic="news",
    tavily_api_key="<api_key>",
)

def search_agent(state: AgentState) -> AgentState:
    summary = state["summary"]
    messages = state["messages"]

    # === Step 1: Ask the model to generate search questions ===
    prompt_1 = f"""
You are a research assistant helping to identify potential competitors based on the company summary below.

Company Summary:
\"\"\"
{summary}
\"\"\"

Step 1: Think about what kind of companies would be competitors.
Step 2: Suggest 2-3 specific search queries that could help identify those competitors.

Return ONLY the questions as a list. Do NOT explain them.
"""
    
    # queries_response = model.generate_content(prompt_1)
    # print("\n[DEBUG] Competitors found:\n", queries_response.text)

    queries_response = llm.invoke(prompt_1)
    print("\n[DEBUG] Queries generated:\n", queries_response.content)

    # Extract search queries from model response
    queries = [line.strip("-• ").strip() for line in queries_response.content.strip().splitlines() if line.strip()]
    # queries = [line.strip("-• ").strip() for line in queries_response.text.strip().splitlines() if line.strip()]
    if not queries:
        queries = [summary]  # fallback query

    # === Step 2: Run Tavily search for each query ===
    all_results = []
    for query in queries:
        result = search_tool.invoke({"query": query})
        # print("\n[DEBUG] Raw search result:\n", type(result), type(result["results"]))
        # all_results.extend(result["results"])
        if isinstance(result, dict) and "results" in result and isinstance(result["results"], list):
            all_results.extend(result["results"])
        else:
            print(f"[WARNING] Unexpected result format for query '{query}':", result)
            

    print(f"\n[DEBUG] Collected {len(all_results)} search results from Tavily.\n")

    # === Step 3: Ask model to extract competitors from the search results ===
    prompt_2 = f"""
Based on the following search results:

{all_results}

Extract at least 3 companies that are potential competitors. For each one, return:
- Company Name
- Company Website

Format your answer as:

[
  {{
    "company_name_1": "Example Co",
    "company_website_1": "https://example.com"
  }},
  ...
]
Only include real companies with websites.
"""
    
    # competitors_response = model.generate_content(prompt_2)
    # print("\n[DEBUG] Competitors found:\n", competitors_response.text)

    competitors_response = llm.invoke(prompt_2)
    print("\n[DEBUG] Competitors found:\n", competitors_response.content)

    # === Step 4: Save result in state ===
    new_message = AIMessage(content=str(competitors_response.content))
    # new_message = AIMessage(content=str(competitors_response.text))
    return {
        "messages": messages + [new_message],
    }

def extract_json_array(text):
    """
    Extracts the first JSON array found in the text.
    """
    match = re.search(r"\[\s*{.*?}\s*\]", text, re.DOTALL)
    if match:
        return match.group(0)
    return None


def news_agent(state: AgentState) -> AgentState:
    messages = state["messages"]
    if not messages:
        raise ValueError("No previous messages found with competitors.")

    last_message = messages[-1].content.strip()

    json_part = extract_json_array(last_message)
    if not json_part:
        raise ValueError("No JSON array found in the last message.")

    try:
        competitors = json.loads(json_part)
        if not isinstance(competitors, list):
            raise ValueError("Expected a list of dictionaries for competitors.")
    except json.JSONDecodeError:
        raise ValueError("Last message is not valid JSON.")

    # ✅ Extract company names from list of dicts
    company_names = []
    for entry in competitors:
        if isinstance(entry, dict):
            for key, value in entry.items():
                if key.startswith("company_name"):
                    company_names.append(value)

    if not company_names:
        raise ValueError("No company names found in the competitor list.")

    print("\n[DEBUG] Extracted company names:\n", company_names)

    # ✅ Generate news-related queries
    queries = [f"Latest product launches, features, or brand updates from {name}" for name in company_names]

    # ✅ Search Tavily for news articles
    all_news_results = []
    for query in queries:
        result = news_tool.invoke({"query": query})
        all_news_results.extend(result["results"])

    print(f"\n[DEBUG] Retrieved {len(all_news_results)} news articles.\n")

    # ✅ Ask LLM to summarize all results, grouped by company
    prompt = f"""
Below are web search results related to recent news about these companies:

{json.dumps(all_news_results, indent=2)}

Please extract and summarize the most recent relevant updates grouped by company. Focus on:
- Product or feature launches
- New courses, services, or content
- Partnerships or deals
- Strategic moves or rebrands

Make the output clean and organized by company name.
"""
    # summary = model.generate_content(prompt)

    summary = llm.invoke(prompt)

    # ✅ Save to state
    new_message = AIMessage(content=summary.content)
    # new_message = AIMessage(content=summary.text)

    return {
        "messages": messages + [new_message],
    }

def generate_answer(state: AgentState):
    prompt = """

You are an AI assitant that helps business to grow and suggest some improvements to business based on their rival company data and current
trends that one needs to follow.
There is one current company who is asking for some improvements or suggestions so that they will keep standing the market.
Now, you have current company summary, all current company's rival company names and all the news about rival company regarding their improvements.
Based on on the data, you need to suggest current company some improvements or suggestions so that they increase their revenue and customer engagement.
"""

    all_messages = state["messages"] + [prompt]

    # response = model.generate_content(all_messages)
    response = llm.invoke(all_messages)

    return {
        "messages" : all_messages + [response.content]
    }



graph = StateGraph(AgentState)
# graph.add_node("agent", agent)
# graph.add_node("search_company", search_agent)
# graph.add_node("generate_answer", generate_answer)
# graph.add_node("news_agent", news_agent)
# graph.set_entry_point("agent")
# graph.add_edge("agent", "search_company")
# graph.add_edge("search_company", "news_agent")
# graph.add_edge("news_agent", "generate_answer")
# graph.add_edge("generate_answer", END)

# graph.add_node("agent", agent)
graph.add_node("search_company", search_agent)
graph.add_node("generate_answer", generate_answer)
graph.add_node("news_agent", news_agent)
graph.set_entry_point("search_company")
graph.add_edge("search_company", "news_agent")
graph.add_edge("news_agent", "generate_answer")
graph.add_edge("generate_answer", END)


full_agent = graph.compile()


def running_agent():
    print("\n=== RAG AGENT===")
    
    # while True:
    #     user_input = input("\nWhat is your question: ")
    #     if user_input.lower() in ['exit', 'quit']:
    #         break

        # query = HumanMessage(content=user_input)
            
        # messages = [HumanMessage(content=user_input)] # converts back to a HumanMessage type

        # company_list = """[
        # {
        #     "company_name": "Pendo",
        #     "company_website": "https://www.pendo.io/"
        # },
        # {
        #     "company_name": "Udemy",
        #     "company_website": "https://www.udemy.com/"
        # },
        # {
        #     "company_name": "Mind the Product",
        #     "company_website": "https://mindtheproduct.com/"
        # }
        # ]"""

    query = "What is the goal of the company?"

    input_state = {
        "company_url": "https://theproductspace.in/sitemap.xml",
        "query": query,
        "messages" : [],
        "summary" : "The name of the company is Product Space.We help user to Master core Product Management Skills and AI workflows to stay relevant in the next wave of Product Careers."
    }

    # result = full_agent.invoke({"query": user_input, "messages": [HumanMessage(content=user_input)]})

    result = full_agent.invoke(input_state)
    
    # print("\n=== ANSWER ===")
    # print(result)
    print("=====Trimmed Answer=========")
    print(result['messages'][-1])

# running_agent()
# print(rag_agent)
