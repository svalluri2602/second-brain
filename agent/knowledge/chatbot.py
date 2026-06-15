"""
Hybrid RAG Chatbot — natural language queries over Neo4j + Qdrant.

Neo4j  → structured graph (jobs, skills, companies, applications)
Qdrant → semantic search (CV chunks, notes, articles)
Claude → synthesizes both into a natural answer

Usage:
  python3 -m agent.knowledge.chatbot
  python3 -m agent.knowledge.chatbot "what jobs match my LangGraph skills?"
"""

import os
import sys
from typing import TypedDict, Optional
from pathlib import Path

from dotenv import load_dotenv
from neo4j import GraphDatabase
from langgraph.graph import StateGraph, END
from anthropic import Anthropic

from agent.knowledge.vector_store import search_relevant

load_dotenv()

client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

NEO4J_URI  = os.getenv("NEO4J_URI",  "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASS = os.getenv("NEO4J_PASS", "secondbrain")


# ── Neo4j helper ──────────────────────────────────────────────────────────────

def neo4j_query(cypher: str, **params) -> list[dict]:
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))
    with driver.session() as s:
        result = s.run(cypher, **params).data()
    driver.close()
    return result


# ── State ─────────────────────────────────────────────────────────────────────

class ChatState(TypedDict):
    question:      str
    cypher:        Optional[str]
    graph_results: list
    vector_results: list
    answer:        str
    history:       list   # list of {"role": ..., "content": ...}


# ── Nodes ─────────────────────────────────────────────────────────────────────

SCHEMA_CONTEXT = """
Neo4j graph schema:
- (Company {name}) -[:POSTED]-> (Role {url, title, location, date_seen, score, archetype})
- (Role) -[:SURFACED_BY]-> (Portal {name})
- (Role) -[:REQUIRES]-> (Skill {name})
- (Application {num, date, company, role, score, status, notes})
- (Codebase {url, name, description, languages, explanation, resume_bullets}) -[:DEMONSTRATES]-> (Skill)
- (Codebase) -[:HAS_COMMIT]-> (GitCommit {sha, message, date})
- (Skill) -[:RELEVANT_TO]-> (Application)

Canonical application statuses: Evaluated, Applied, Responded, Interview, Offer, Rejected, Discarded, SKIP
"""


def generate_cypher_node(state: ChatState) -> dict:
    """Convert natural language question to Cypher query."""
    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=400,
        messages=[{
            "role": "user",
            "content": f"""{SCHEMA_CONTEXT}

Convert this question to a Cypher query. Return ONLY the Cypher query, nothing else.
If the question cannot be answered from the graph, return: SKIP

Question: {state['question']}

Rules:
- Use MATCH, WHERE, RETURN
- Always LIMIT to 20 results max
- For skills use toLower() for matching
- Return readable field names"""
        }]
    )
    cypher = response.content[0].text.strip()
    if cypher.startswith("```"):
        cypher = cypher.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    return {"cypher": cypher}


def neo4j_search_node(state: ChatState) -> dict:
    """Run the Cypher query against Neo4j."""
    if not state["cypher"] or state["cypher"] == "SKIP":
        return {"graph_results": []}
    try:
        results = neo4j_query(state["cypher"])
        return {"graph_results": results[:20]}
    except Exception as e:
        return {"graph_results": [{"error": str(e)}]}


def qdrant_search_node(state: ChatState) -> dict:
    """Semantic search on CV + stored content."""
    try:
        results = search_relevant(state["question"], limit=3)
        return {"vector_results": results}
    except Exception:
        return {"vector_results": []}


def synthesize_node(state: ChatState) -> dict:
    """Combine graph + vector results and generate final answer."""
    graph_str  = str(state["graph_results"])  if state["graph_results"]  else "No graph results."
    vector_str = "\n---\n".join(state["vector_results"]) if state["vector_results"] else "No CV context."

    history = state.get("history", [])
    messages = history + [{
        "role": "user",
        "content": f"""You are a personal career AI assistant for Satwik Valluri, CS senior at Virginia Tech graduating Dec 2026.
You have access to his full career graph and CV.

Question: {state['question']}

Graph data (Neo4j):
{graph_str}

CV context (semantic search):
{vector_str}

Answer naturally and helpfully. Be specific — use real data from above.
If graph data is empty or irrelevant, rely on CV context.
Keep it concise unless they ask for detail."""
    }]

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=800,
        messages=messages
    )

    answer = response.content[0].text.strip()

    updated_history = history + [
        {"role": "user",      "content": state["question"]},
        {"role": "assistant", "content": answer}
    ]

    return {"answer": answer, "history": updated_history}


# ── Build pipeline ────────────────────────────────────────────────────────────

def build_chatbot():
    g = StateGraph(ChatState)
    g.add_node("generate_cypher", generate_cypher_node)
    g.add_node("neo4j_search",    neo4j_search_node)
    g.add_node("qdrant_search",   qdrant_search_node)
    g.add_node("synthesize",      synthesize_node)

    g.set_entry_point("generate_cypher")
    g.add_edge("generate_cypher", "neo4j_search")
    g.add_edge("generate_cypher", "qdrant_search")
    g.add_edge("neo4j_search",    "synthesize")
    g.add_edge("qdrant_search",   "synthesize")
    g.add_edge("synthesize",      END)
    return g.compile()


# ── REPL ─────────────────────────────────────────────────────────────────────

def run_chat():
    print("Career AI — ask me anything about your pipeline, skills, or jobs.")
    print("Type 'quit' to exit.\n")

    bot     = build_chatbot()
    history = []

    while True:
        try:
            question = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nBye.")
            break

        if not question or question.lower() in ("quit", "exit", "q"):
            print("Bye.")
            break

        state = bot.invoke({
            "question":       question,
            "cypher":         None,
            "graph_results":  [],
            "vector_results": [],
            "answer":         "",
            "history":        history,
        })

        print(f"\nAI: {state['answer']}\n")
        history = state["history"]


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
        bot = build_chatbot()
        state = bot.invoke({
            "question": question, "cypher": None,
            "graph_results": [], "vector_results": [],
            "answer": "", "history": [],
        })
        print(state["answer"])
    else:
        run_chat()
