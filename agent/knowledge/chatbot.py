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

Canonical application statuses: Evaluated, Applied, Responded, Interview, Offer, Rejected, Discarded, SKIP

Example queries:
- All applications:
  MATCH (a:Application) RETURN a.num, a.company, a.role, a.score, a.status ORDER BY a.score DESC

- Skills my GitHub demonstrates:
  MATCH (cb:Codebase)-[:DEMONSTRATES]->(s:Skill) RETURN cb.name AS repo, collect(s.name) AS skills

- Gap detection (my GitHub skills vs job requirements — do NOT add Application joins):
  MATCH (cb:Codebase)-[:DEMONSTRATES]->(ds:Skill)
  MATCH (rs:Skill)<-[:REQUIRES]-(r:Role)
  WHERE toLower(rs.name) CONTAINS toLower(ds.name) OR toLower(ds.name) CONTAINS toLower(rs.name)
  RETURN rs.name AS skill, count(DISTINCT r) AS jobs_needing_it, collect(DISTINCT cb.name)[0..2] AS your_repos
  ORDER BY jobs_needing_it DESC LIMIT 10

- Companies by portal:
  MATCH (c:Company)-[:POSTED]->(r:Role)-[:SURFACED_BY]->(p:Portal {name: 'greenhouse'})
  RETURN c.name, r.title LIMIT 10

- My codebases:
  MATCH (cb:Codebase) RETURN cb.name, cb.explanation, cb.languages, cb.resume_bullets
"""


PREBUILT_QUERIES = {
    "gap": """
        MATCH (cb:Codebase)-[:DEMONSTRATES]->(s:Skill)<-[:REQUIRES]-(r:Role)
        RETURN s.name AS skill, count(DISTINCT r) AS jobs_needing_it,
               collect(DISTINCT cb.name)[0..2] AS your_repos
        ORDER BY jobs_needing_it DESC LIMIT 15
    """,
    "applications": """
        MATCH (a:Application)
        RETURN a.num AS num, a.company AS company, a.role AS role,
               a.score AS score, a.status AS status, a.notes AS notes
        ORDER BY a.score DESC
    """,
    "codebases": """
        MATCH (cb:Codebase)
        RETURN cb.name AS name, cb.explanation AS explanation,
               cb.languages AS languages, cb.resume_bullets AS bullets,
               cb.complexity AS complexity
    """,
    "skills": """
        MATCH (cb:Codebase)-[:DEMONSTRATES]->(s:Skill)
        RETURN cb.name AS repo, collect(s.name) AS skills
    """,
    "companies": """
        MATCH (c:Company)-[:POSTED]->(r:Role)
        RETURN c.name AS company, count(r) AS roles
        ORDER BY roles DESC LIMIT 20
    """,
}

def _route_prebuilt(question: str) -> str | None:
    q = question.lower()
    if any(w in q for w in ["gap", "match", "require", "need", "missing", "qualify"]):
        return PREBUILT_QUERIES["gap"]
    if any(w in q for w in ["application", "applied", "pipeline", "status", "score"]):
        return PREBUILT_QUERIES["applications"]
    if any(w in q for w in ["built", "project", "codebase", "repo", "github"]):
        return PREBUILT_QUERIES["codebases"]
    if any(w in q for w in ["skill", "technology", "stack", "know", "experience"]):
        return PREBUILT_QUERIES["skills"]
    if any(w in q for w in ["compan", "employer", "who is hiring"]):
        return PREBUILT_QUERIES["companies"]
    return None


def generate_cypher_node(state: ChatState) -> dict:
    """Route to pre-built Cypher or generate dynamically."""
    prebuilt = _route_prebuilt(state["question"])
    if prebuilt:
        return {"cypher": prebuilt}

    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=400,
        messages=[{
            "role": "user",
            "content": f"""{SCHEMA_CONTEXT}

Convert this question to a Cypher query. Return ONLY the Cypher, nothing else.
If unanswerable from the graph, return: SKIP

Question: {state['question']}

Rules:
- ONLY use relationships defined in the schema above
- LIMIT to 20 max
- toLower() for string matching"""
        }]
    )
    cypher = response.content[0].text.strip()
    if "```" in cypher:
        cypher = cypher.split("```")[1]
        if cypher.lower().startswith("cypher"):
            cypher = cypher[6:]
        cypher = cypher.strip()
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
