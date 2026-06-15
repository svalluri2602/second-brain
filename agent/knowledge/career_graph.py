import csv
import re
import yaml
import os
import uuid
from pathlib import Path
from datetime import date
from typing import TypedDict, Optional

from neo4j import GraphDatabase
from langgraph.graph import StateGraph, END
from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()

client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

ROOT       = Path(__file__).parent.parent.parent
CAREER_OPS = ROOT / "career-ops"

NEO4J_URI  = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASS = os.getenv("NEO4J_PASS", "secondbrain")


# ── Driver ─────────────────────────────────────────────────────────────────────

def get_driver():
    return GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))


def run(driver, query: str, **params):
    with driver.session() as session:
        return session.run(query, **params).data()


# ── Schema (constraints + indexes) ────────────────────────────────────────────

def init_schema(driver):
    constraints = [
        "CREATE CONSTRAINT IF NOT EXISTS FOR (c:Company)     REQUIRE c.name       IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (r:Role)        REQUIRE r.url        IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (a:Application) REQUIRE a.num        IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (p:Portal)      REQUIRE p.name       IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (o:Outreach)    REQUIRE o.id         IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (s:Skill)       REQUIRE s.name       IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (e:ErrorLog)    REQUIRE e.error      IS UNIQUE",
    ]
    for c in constraints:
        run(driver, c)
    print("Schema ready.")


# ── Ingestion ──────────────────────────────────────────────────────────────────

def ingest_scan_history(driver):
    tsv_path = CAREER_OPS / "data" / "scan-history.tsv"
    if not tsv_path.exists():
        print("scan-history.tsv not found, skipping")
        return

    added = 0
    with open(tsv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            url      = row.get("url", "").strip()
            title    = row.get("title", "").strip()
            company  = row.get("company", "").strip()
            portal   = row.get("portal", "").strip()
            date_seen = row.get("first_seen", "").strip()
            location = row.get("location", "").strip()

            if not url or not company:
                continue

            run(driver, """
                MERGE (c:Company {name: $company})
                ON CREATE SET c.stage = '', c.location = $location
            """, company=company, location=location)

            run(driver, """
                MERGE (r:Role {url: $url})
                ON CREATE SET r.title = $title, r.archetype = '',
                              r.location = $location, r.date_seen = $date_seen,
                              r.source = $portal
            """, url=url, title=title, location=location,
                 date_seen=date_seen, portal=portal)

            run(driver, "MERGE (p:Portal {name: $portal})", portal=portal)

            run(driver, """
                MATCH (c:Company {name: $company}), (r:Role {url: $url})
                MERGE (c)-[:POSTED]->(r)
            """, company=company, url=url)

            run(driver, """
                MATCH (r:Role {url: $url}), (p:Portal {name: $portal})
                MERGE (r)-[:SURFACED_BY]->(p)
            """, url=url, portal=portal)

            added += 1

    print(f"scan-history ingested — {added} roles → Neo4j")


def ingest_applications(driver):
    apps_path = CAREER_OPS / "data" / "applications.md"
    if not apps_path.exists():
        print("applications.md not found, skipping")
        return

    added = 0
    for line in open(apps_path, encoding="utf-8"):
        line = line.strip()
        if not line.startswith("|") or line.startswith("| #") or line.startswith("|---"):
            continue

        parts = [p.strip() for p in line.strip("|").split("|")]
        if len(parts) < 7:
            continue

        try:
            num       = int(parts[0])
            date_val  = parts[1]
            company   = parts[2]
            role      = parts[3]
            score     = float(parts[4].replace("/5", "").strip() or 0)
            status    = parts[5]
            pdf       = parts[6].strip() == "✅"
            report    = re.sub(r'\[.*?\]\((.*?)\)', r'\1', parts[7]) if len(parts) > 7 else ""
            notes     = parts[8] if len(parts) > 8 else ""
        except (ValueError, IndexError):
            continue

        run(driver, "MERGE (c:Company {name: $company}) ON CREATE SET c.stage='', c.location=''",
            company=company)

        run(driver, """
            MERGE (a:Application {num: $num})
            ON CREATE SET a.date = $date, a.score = $score, a.status = $status,
                          a.pdf = $pdf, a.report_path = $report, a.notes = $notes,
                          a.company = $company, a.role = $role
            ON MATCH SET  a.status = $status, a.score = $score
        """, num=num, date=date_val, score=score, status=status,
             pdf=pdf, report=report, notes=notes, company=company, role=role)

        added += 1

    print(f"applications.md ingested — {added} applications → Neo4j")


def ingest_reports(driver):
    reports_dir = CAREER_OPS / "reports"
    if not reports_dir.exists():
        return

    for report_file in sorted(reports_dir.glob("*.md")):
        content = report_file.read_text(encoding="utf-8")
        match = re.search(r"```yaml\n(.*?)```", content, re.DOTALL)
        if not match:
            continue
        try:
            summary = yaml.safe_load(match.group(1))
        except yaml.YAMLError:
            continue

        url       = str(summary.get("url", ""))
        archetype = str(summary.get("archetype", ""))
        score     = float(summary.get("score", 0))
        status    = str(summary.get("status", ""))

        if url:
            run(driver, """
                MERGE (r:Role {url: $url})
                ON CREATE SET r.title='', r.archetype=$archetype, r.location='',
                              r.date_seen='', r.source=''
                ON MATCH SET  r.archetype = $archetype
            """, url=url, archetype=archetype)

        print(f"  report ingested: {report_file.name}")


# ── LangGraph: outreach pipeline ───────────────────────────────────────────────

class OutreachState(TypedDict):
    min_score:         float
    jobs:              list
    cv_text:           str
    outreach_messages: list


def load_top_jobs_node(state: OutreachState) -> dict:
    driver = get_driver()
    results = run(driver, """
        MATCH (a:Application)
        WHERE a.score >= $min_score AND a.status <> 'Discarded'
        RETURN a.num AS num, a.score AS score, a.status AS status,
               a.notes AS notes, a.report_path AS report_path, a.role AS role
        ORDER BY a.score DESC
        LIMIT 5
    """, min_score=state["min_score"])
    driver.close()
    print(f"Found {len(results)} jobs scoring >= {state['min_score']}")
    return {"jobs": results}


def load_cv_node(state: OutreachState) -> dict:
    cv_path = CAREER_OPS / "cv.md"
    return {"cv_text": cv_path.read_text(encoding="utf-8") if cv_path.exists() else ""}


def generate_outreach_node(state: OutreachState) -> dict:
    messages = []
    for job in state["jobs"]:
        report_path = CAREER_OPS / (job.get("report_path") or "").lstrip("../")
        report_text = report_path.read_text(encoding="utf-8")[:2000] if report_path.exists() else ""

        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=400,
            messages=[{
                "role": "user",
                "content": f"""Write a personalized LinkedIn outreach message.

Candidate CV:
{state['cv_text'][:1500]}

Job evaluation:
{report_text}

Rules:
- Under 300 characters
- Reference 1 specific technical match
- Natural, not salesy
- Sender: Satwik Valluri, CS senior at Virginia Tech graduating Dec 2026"""
            }]
        )
        msg = response.content[0].text.strip()
        messages.append({"job_num": job["num"], "score": job["score"], "message": msg})
        print(f"\nJob #{job['num']} — {job.get('role', '')} (score {job['score']}):\n{msg}")

    return {"outreach_messages": messages}


def store_outreach_node(state: OutreachState) -> dict:
    driver = get_driver()
    today  = str(date.today())

    for item in state["outreach_messages"]:
        oid = str(uuid.uuid4())[:8]
        msg = item["message"]
        run(driver, """
            CREATE (o:Outreach {id: $id, date: $date, message: $message})
        """, id=oid, date=today, message=msg)

        run(driver, """
            MATCH (o:Outreach {id: $id}), (a:Application {num: $num})
            CREATE (o)-[:SENT_FOR]->(a)
        """, id=oid, num=item["job_num"])

    driver.close()
    return {}


def build_outreach_pipeline():
    g = StateGraph(OutreachState)
    g.add_node("load_top_jobs",      load_top_jobs_node)
    g.add_node("load_cv",            load_cv_node)
    g.add_node("generate_outreach",  generate_outreach_node)
    g.add_node("store_outreach",     store_outreach_node)

    g.set_entry_point("load_top_jobs")
    g.add_edge("load_top_jobs",     "load_cv")
    g.add_edge("load_cv",           "generate_outreach")
    g.add_edge("generate_outreach", "store_outreach")
    g.add_edge("store_outreach",    END)
    return g.compile()


# ── Queries ────────────────────────────────────────────────────────────────────

def query_top_applications(min_score: float = 4.0):
    driver  = get_driver()
    results = run(driver, """
        MATCH (a:Application)
        WHERE a.score >= $min_score
        RETURN a.num, a.score, a.status, a.role, a.company
        ORDER BY a.score DESC
    """, min_score=min_score)
    driver.close()
    return results


def query_companies_by_portal(portal_name: str):
    driver  = get_driver()
    results = run(driver, """
        MATCH (c:Company)-[:POSTED]->(r:Role)-[:SURFACED_BY]->(p:Portal {name: $portal})
        RETURN DISTINCT c.name AS company, r.title AS role, r.date_seen AS date
        ORDER BY r.date_seen DESC
        LIMIT 20
    """, portal=portal_name)
    driver.close()
    return results


# ── Skill extraction ──────────────────────────────────────────────────────────

# Common tech skills to detect in job titles — zero API cost, instant
SKILL_KEYWORDS = [
    "Python", "JavaScript", "TypeScript", "Java", "Go", "Rust", "C++", "Ruby", "Scala",
    "React", "Next.js", "Vue", "Angular", "Node.js", "FastAPI", "Django", "Flask",
    "LangChain", "LangGraph", "LangSmith", "LlamaIndex", "Haystack",
    "OpenAI", "Anthropic", "Claude", "GPT", "Gemini",
    "RAG", "GraphRAG", "Embeddings", "Fine-tuning", "RLHF", "Prompt Engineering",
    "AI Agent", "Agentic", "Multi-agent", "Autonomous Agent",
    "Neo4j", "PostgreSQL", "MySQL", "MongoDB", "Redis", "SQLite", "Pinecone", "Qdrant",
    "AWS", "GCP", "Azure", "Docker", "Kubernetes", "Terraform",
    "Machine Learning", "Deep Learning", "NLP", "Computer Vision",
    "PyTorch", "TensorFlow", "Hugging Face", "scikit-learn",
    "Playwright", "Selenium", "Web Scraping",
    "GraphQL", "REST", "API", "Microservices",
    "Git", "CI/CD", "GitHub Actions",
    "SQL", "ETL", "Data Pipeline", "Spark",
    "LLM", "Vector Database", "Knowledge Graph",
]


def extract_skills_from_titles(driver):
    """Scan all Role titles for known skills, create Skill nodes + REQUIRES edges."""
    print("Extracting skills from job titles...")

    roles = run(driver, "MATCH (r:Role) RETURN r.url AS url, r.title AS title")

    linked = 0
    for role in roles:
        title = (role.get("title") or "").lower()
        url   = role.get("url", "")
        if not url:
            continue

        for skill in SKILL_KEYWORDS:
            if skill.lower() in title:
                run(driver, """
                    MERGE (s:Skill {name: $skill})
                    WITH s
                    MATCH (r:Role {url: $url})
                    MERGE (r)-[:REQUIRES]->(s)
                """, skill=skill, url=url)
                linked += 1

    print(f"  Skill extraction done — {linked} Role→Skill links created")


def show_skill_gap(driver):
    """Compare skills your GitHub demonstrates vs skills jobs require."""
    results = run(driver, """
        MATCH (cb:Codebase)-[:DEMONSTRATES]->(s:Skill)<-[:REQUIRES]-(r:Role)
        RETURN s.name AS skill,
               count(DISTINCT r) AS jobs_requiring,
               collect(DISTINCT cb.name)[0..3] AS your_repos
        ORDER BY jobs_requiring DESC
        LIMIT 20
    """)

    if not results:
        print("  No skill matches yet — run github_connector on your repos first")
        return

    print("\n── Skill Gap Report ──────────────────────────────────────────")
    print(f"  {'Skill':<25} {'Jobs requiring':<15} {'Your repos'}")
    print(f"  {'─'*25} {'─'*15} {'─'*30}")
    for r in results:
        repos = ", ".join(r["your_repos"])
        print(f"  {r['skill']:<25} {r['jobs_requiring']:<15} {repos}")


# ── Main ───────────────────────────────────────────────────────────────────────

def ingest_all():
    driver = get_driver()
    init_schema(driver)
    ingest_scan_history(driver)
    ingest_applications(driver)
    ingest_reports(driver)
    extract_skills_from_titles(driver)
    show_skill_gap(driver)
    driver.close()
    print("\nIngestion complete.")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "outreach":
        min_score = float(sys.argv[2]) if len(sys.argv) > 2 else 4.0
        build_outreach_pipeline().invoke({
            "min_score": min_score, "jobs": [],
            "cv_text": "", "outreach_messages": []
        })
    else:
        ingest_all()
        print("\nTop applications (score >= 4.0):")
        for row in query_top_applications(4.0):
            print(f"  #{row['a.num']} | {row['a.score']}/5 | {row['a.status']} | {row['a.role']}")
