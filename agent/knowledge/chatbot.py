"""
Hybrid RAG Chatbot — intelligent graph traversal over Neo4j + Qdrant.

Design principles:
- Pre-built traversal queries for every common question type (never fail)
- Multi-hop graph traversal (Codebase → Skill → Role → Company)
- Error tracking stored in graph (ErrorLog nodes)
- Fall back to Qdrant semantic search when graph has no data
- Claude synthesizes graph + vector results into grounded answers

Usage:
  python3 -m agent.knowledge.chatbot
  python3 -m agent.knowledge.chatbot "what jobs match my LangGraph skills?"
"""

import os
import re
import sys
from datetime import date
from typing import TypedDict, Optional

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


# ── Neo4j helpers ─────────────────────────────────────────────────────────────

def neo4j_query(cypher: str, **params) -> list[dict]:
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))
    with driver.session() as s:
        result = s.run(cypher, **params).data()
    driver.close()
    return result


def log_error(project: str, error_msg: str, fix: str = "", skill: str = ""):
    """Store an error/fix in the graph for future learning."""
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))
    with driver.session() as s:
        s.run("""
            MERGE (e:ErrorLog {error: $error, project: $project})
            SET e.fix = $fix, e.date = $today, e.skill = $skill
            WITH e
            MATCH (cb:Codebase {name: $project})
            MERGE (cb)-[:HAS_ERROR]->(e)
        """, error=error_msg[:300], project=project,
             fix=fix[:500], today=str(date.today()), skill=skill)
    driver.close()


# ── State ─────────────────────────────────────────────────────────────────────

class ChatState(TypedDict):
    question:       str
    query_type:     str
    cypher:         Optional[str]
    graph_results:  list
    vector_results: list
    answer:         str
    history:        list


# ── Pre-built traversal queries ───────────────────────────────────────────────
# Each is a proven multi-hop Cypher query. Never generates bad Cypher.

TRAVERSALS = {

    # Your GitHub skills → jobs that need them (multi-hop)
    "gap": """
        MATCH (cb:Codebase)-[:DEMONSTRATES]->(s:Skill)<-[:REQUIRES]-(r:Role)<-[:POSTED]-(c:Company)
        RETURN s.name AS skill,
               count(DISTINCT r) AS jobs_needing_it,
               collect(DISTINCT cb.name)[0..2] AS your_repos,
               collect(DISTINCT c.name)[0..3] AS companies
        ORDER BY jobs_needing_it DESC LIMIT 15
    """,

    # Full application pipeline with scores
    "applications": """
        MATCH (a:Application)
        RETURN a.num AS num, a.company AS company, a.role AS role,
               a.score AS score, a.status AS status, a.notes AS notes
        ORDER BY a.score DESC
    """,

    # Your GitHub projects with full context
    "codebases": """
        MATCH (cb:Codebase)
        OPTIONAL MATCH (cb)-[:DEMONSTRATES]->(s:Skill)
        OPTIONAL MATCH (cb)-[:HAS_COMMIT]->(gc:GitCommit)
        RETURN cb.name AS name, cb.explanation AS explanation,
               cb.technical_summary AS technical_summary,
               cb.languages AS languages,
               cb.resume_bullets AS bullets,
               cb.complexity AS complexity,
               cb.project_type AS type,
               collect(DISTINCT s.name) AS skills,
               count(DISTINCT gc) AS commit_count
    """,

    # All skills demonstrated across repos
    "my_skills": """
        MATCH (cb:Codebase)-[:DEMONSTRATES]->(s:Skill)
        WITH s.name AS skill, collect(DISTINCT cb.name) AS repos
        RETURN skill, repos
        ORDER BY size(repos) DESC, skill ASC
    """,

    # What skills do jobs in my pipeline require most
    "job_skill_demand": """
        MATCH (r:Role)-[:REQUIRES]->(s:Skill)
        RETURN s.name AS skill, count(DISTINCT r) AS demand
        ORDER BY demand DESC LIMIT 20
    """,

    # Multi-hop: path from my repos → matching companies
    "matching_companies": """
        MATCH (cb:Codebase)-[:DEMONSTRATES]->(s:Skill)<-[:REQUIRES]-(r:Role)<-[:POSTED]-(c:Company)
        WITH c.name AS company, collect(DISTINCT s.name) AS matched_skills,
             count(DISTINCT r) AS roles
        RETURN company, matched_skills, roles
        ORDER BY size(matched_skills) DESC LIMIT 15
    """,

    # Recent commits across all my repos
    "recent_commits": """
        MATCH (cb:Codebase)-[:HAS_COMMIT]->(gc:GitCommit)
        RETURN cb.name AS repo, gc.message AS commit, gc.date AS date
        ORDER BY gc.date DESC LIMIT 20
    """,

    # Errors logged on my projects
    "errors": """
        MATCH (cb:Codebase)-[:HAS_ERROR]->(e:ErrorLog)
        RETURN cb.name AS project, e.error AS error,
               e.fix AS fix, e.date AS date, e.skill AS skill
        ORDER BY e.date DESC LIMIT 20
    """,

    # What's stale in my pipeline
    "stale": """
        MATCH (a:Application)
        WHERE a.status IN ['Evaluated', 'Applied', 'Responded']
        RETURN a.num AS num, a.company AS company, a.role AS role,
               a.status AS status, a.date AS date, a.score AS score
        ORDER BY a.date ASC
    """,

    # Skills I'm missing — in job demand but NOT in my repos
    "missing_skills": """
        MATCH (r:Role)-[:REQUIRES]->(s:Skill)
        WHERE NOT EXISTS {
            MATCH (cb:Codebase)-[:DEMONSTRATES]->(s)
        }
        RETURN s.name AS missing_skill, count(DISTINCT r) AS jobs_needing_it
        ORDER BY jobs_needing_it DESC LIMIT 15
    """,

    # Full graph summary — what's in the graph
    "summary": """
        MATCH (c:Company) WITH count(c) AS companies
        MATCH (r:Role)    WITH companies, count(r) AS roles
        MATCH (a:Application) WITH companies, roles, count(a) AS apps
        MATCH (s:Skill)   WITH companies, roles, apps, count(s) AS skills
        MATCH (cb:Codebase) WITH companies, roles, apps, skills, count(cb) AS codebases
        RETURN companies, roles, apps, skills, codebases
    """,
}


# ── Query router ──────────────────────────────────────────────────────────────

ROUTE_MAP = [
    (["gap", "match", "qualify", "fit for", "what jobs", "missing skill"],           "gap"),
    (["application", "applied", "pipeline", "status", "score", "offer", "rejected"], "applications"),
    (["stale", "follow up", "no reply", "cold", "overdue"],                          "stale"),
    (["built", "project", "codebase", "repo", "github", "explain my"],               "codebases"),
    (["my skill", "what i know", "what can i do", "my stack", "technologies i"],     "my_skills"),
    (["missing", "don't have", "lack", "should learn", "need to learn"],             "missing_skills"),
    (["compan", "who is hiring", "which companies", "employers"],                     "matching_companies"),
    (["commit", "recent work", "what have i worked"],                                 "recent_commits"),
    (["error", "bug", "fix", "problem", "issue", "mistake"],                         "errors"),
    (["demand", "most wanted", "popular skill", "trending"],                         "job_skill_demand"),
    (["summary", "overview", "how many", "stats", "total"],                          "summary"),
]


def route_query(question: str) -> tuple[str, str]:
    """Return (query_type, cypher). Falls back to LLM generation."""
    q = question.lower()
    for keywords, qtype in ROUTE_MAP:
        if any(kw in q for kw in keywords):
            return qtype, TRAVERSALS[qtype]
    return "generated", ""


# ── LangGraph nodes ───────────────────────────────────────────────────────────

SCHEMA_HINT = """
Neo4j schema (ONLY use these nodes and relationships):
Nodes: Company{name}, Role{url,title,location,date_seen}, Portal{name},
       Skill{name}, Application{num,date,company,role,score,status,notes},
       Codebase{url,name,description,languages,explanation,resume_bullets},
       GitCommit{sha,message,date}, ErrorLog{error,fix,date,project,skill}

Relationships:
(Company)-[:POSTED]->(Role)
(Role)-[:SURFACED_BY]->(Portal)
(Role)-[:REQUIRES]->(Skill)
(Codebase)-[:DEMONSTRATES]->(Skill)
(Codebase)-[:HAS_COMMIT]->(GitCommit)
(Codebase)-[:HAS_ERROR]->(ErrorLog)

Rules: ONLY these relationships. Always LIMIT 20. Use toLower() for text matching.
"""


def generate_cypher_node(state: ChatState) -> dict:
    qtype, cypher = route_query(state["question"])

    if cypher:
        return {"query_type": qtype, "cypher": cypher}

    # LLM fallback for unknown questions
    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=300,
        messages=[{"role": "user", "content": f"""{SCHEMA_HINT}
Write a Cypher query for: {state['question']}
Return ONLY the Cypher. If unanswerable from schema, return: SKIP"""}]
    )
    raw = response.content[0].text.strip()
    if "```" in raw:
        raw = raw.split("```")[1]
        raw = re.sub(r"^(cypher|sql)\s*", "", raw, flags=re.IGNORECASE).strip()
    return {"query_type": "generated", "cypher": raw}


def neo4j_search_node(state: ChatState) -> dict:
    if not state.get("cypher") or state["cypher"].strip().upper() == "SKIP":
        return {"graph_results": []}
    try:
        results = neo4j_query(state["cypher"])
        return {"graph_results": results[:20]}
    except Exception as e:
        return {"graph_results": [{"_query_error": str(e)[:200]}]}


def qdrant_search_node(state: ChatState) -> dict:
    try:
        return {"vector_results": search_relevant(state["question"], limit=3)}
    except Exception:
        return {"vector_results": []}


def synthesize_node(state: ChatState) -> dict:
    graph_data   = state["graph_results"]
    vector_data  = state["vector_results"]
    query_type   = state.get("query_type", "")

    has_graph    = graph_data and "_query_error" not in str(graph_data)
    has_vector   = bool(vector_data)

    graph_str    = str(graph_data) if has_graph else "No graph data returned."
    vector_str   = "\n---\n".join(vector_data) if has_vector else "No CV context."

    system_prompt = """You are Satwik Valluri's personal career AI with full access to his graph.
Satwik is a CS senior at Virginia Tech graduating December 2026. He's targeting AI agent engineering internships and full-time roles.

When the graph returns data: use it directly, be specific, cite companies/skills/scores by name.
When the graph is empty: say so clearly but still help using CV context.
Never make up graph data. Never be generic when you have specifics.
Format with markdown. Be concise but complete."""

    messages = state.get("history", []) + [{
        "role": "user",
        "content": f"""Question: {state['question']}

Query type: {query_type}

Graph data (Neo4j traversal):
{graph_str}

CV/semantic context (Qdrant):
{vector_str}

Answer grounded in the data above."""
    }]

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1000,
        system=system_prompt,
        messages=messages
    )

    answer = response.content[0].text.strip()
    updated_history = state.get("history", []) + [
        {"role": "user",      "content": state["question"]},
        {"role": "assistant", "content": answer},
    ]
    return {"answer": answer, "history": updated_history}


# ── Build pipeline ────────────────────────────────────────────────────────────

def build_chatbot():
    g = StateGraph(ChatState)
    g.add_node("route",     generate_cypher_node)
    g.add_node("graph",     neo4j_search_node)
    g.add_node("vector",    qdrant_search_node)
    g.add_node("synthesize", synthesize_node)

    g.set_entry_point("route")
    g.add_edge("route",   "graph")
    g.add_edge("route",   "vector")
    g.add_edge("graph",   "synthesize")
    g.add_edge("vector",  "synthesize")
    g.add_edge("synthesize", END)
    return g.compile()


# ── Public API ────────────────────────────────────────────────────────────────

def ask(question: str, history: list = None) -> str:
    """Single-shot query. Returns answer string."""
    bot = build_chatbot()
    state = bot.invoke({
        "question": question, "query_type": "",
        "cypher": None, "graph_results": [],
        "vector_results": [], "answer": "",
        "history": history or [],
    })
    return state["answer"]


# ── REPL ─────────────────────────────────────────────────────────────────────

def run_chat():
    print("Career AI — ask me anything about your graph, skills, jobs, or code.")
    print("Commands: 'errors' to log an error, 'quit' to exit.\n")

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
            "question": question, "query_type": "",
            "cypher": None, "graph_results": [],
            "vector_results": [], "answer": "",
            "history": history,
        })

        print(f"\nAI: {state['answer']}\n")
        history = state["history"]


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) > 1:
        print(ask(" ".join(sys.argv[1:])))
    else:
        run_chat()
