"""
GitHub Connector — pulls a repo into the Neo4j graph.

What it does:
  1. Fetches repo metadata, languages, README, recent commits, file tree via GitHub API
  2. Uses Claude to extract skills + generate plain-English explanation + resume bullets
  3. Stores Codebase, GitCommit, Skill nodes in Neo4j
  4. Links Skill → Role nodes already in graph (gap detection)

Usage:
  python3 -m agent.knowledge.github_connector https://github.com/user/repo
  python3 -m agent.knowledge.github_connector https://github.com/user/repo --owner "Your Name"
"""

import os
import re
import sys
import json
import urllib.request
import urllib.error
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from neo4j import GraphDatabase
from anthropic import Anthropic
from agent.knowledge.career_graph import SKILL_KEYWORDS

load_dotenv()

client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

NEO4J_URI  = os.getenv("NEO4J_URI",  "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASS = os.getenv("NEO4J_PASS", "secondbrain")
GH_TOKEN   = os.getenv("GITHUB_TOKEN", "")


# ── Skill normalization ───────────────────────────────────────────────────────

def _word_tokens(text: str) -> set[str]:
    """Split text into lowercase word tokens on common delimiters."""
    return set(re.split(r"[\s\-/_.,()+]+", text.lower()))


def normalize_skills(raw_skills: list, languages: list) -> list:
    """
    Match Claude's free-form skill names against SKILL_KEYWORDS using
    word-boundary matching to avoid 'Go' matching 'Django', etc.
    GitHub-detected languages are matched by exact case-insensitive comparison.
    """
    normalized = set()

    # GitHub API languages are exact — match directly
    for lang in languages:
        for kw in SKILL_KEYWORDS:
            if kw.lower() == lang.lower():
                normalized.add(kw)
                break

    # For Claude's free-form skills use word-token matching
    for raw in raw_skills:
        raw_tokens = _word_tokens(raw)
        for kw in SKILL_KEYWORDS:
            kw_tokens = _word_tokens(kw)
            # All tokens of the keyword must appear in the raw skill tokens
            if kw_tokens and kw_tokens.issubset(raw_tokens):
                normalized.add(kw)
            # Or all tokens of raw appear in keyword tokens (reverse)
            elif raw_tokens and raw_tokens.issubset(kw_tokens):
                normalized.add(kw)

    return sorted(normalized)


# ── GitHub API helpers ────────────────────────────────────────────────────────

def _gh_get(path: str) -> dict | list:
    url = f"https://api.github.com{path}"
    req = urllib.request.Request(url)
    req.add_header("Accept", "application/vnd.github+json")
    req.add_header("User-Agent", "second-brain-connector")
    if GH_TOKEN:
        req.add_header("Authorization", f"Bearer {GH_TOKEN}")
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return {}
        raise


def parse_repo_url(url: str) -> tuple[str, str]:
    url = url.rstrip("/")
    match = re.search(r"github\.com/([^/]+)/([^/]+)", url)
    if not match:
        raise ValueError(f"Not a valid GitHub URL: {url}")
    return match.group(1), match.group(2).removesuffix(".git")


def fetch_repo_data(owner: str, repo: str) -> dict:
    print(f"Fetching {owner}/{repo} from GitHub...")

    meta      = _gh_get(f"/repos/{owner}/{repo}")
    languages = _gh_get(f"/repos/{owner}/{repo}/languages")
    commits   = _gh_get(f"/repos/{owner}/{repo}/commits?per_page=30")
    tree_resp = _gh_get(f"/repos/{owner}/{repo}/git/trees/HEAD?recursive=0")
    readme    = _gh_get(f"/repos/{owner}/{repo}/readme")

    readme_text = ""
    if readme and readme.get("content"):
        import base64
        try:
            readme_text = base64.b64decode(readme["content"]).decode("utf-8", errors="ignore")[:3000]
        except Exception:
            pass

    file_tree = []
    if tree_resp and tree_resp.get("tree"):
        file_tree = [
            item["path"] for item in tree_resp["tree"]
            if item.get("type") == "blob" and "/" not in item["path"]
        ][:30]

    commit_list = []
    if isinstance(commits, list):
        for c in commits[:30]:
            msg  = c.get("commit", {}).get("message", "").split("\n")[0]
            date = c.get("commit", {}).get("author", {}).get("date", "")[:10]
            sha  = c.get("sha", "")[:8]
            commit_list.append({"sha": sha, "message": msg, "date": date})

    return {
        "name":        meta.get("name", repo),
        "description": meta.get("description") or "",
        "url":         meta.get("html_url", f"https://github.com/{owner}/{repo}"),
        "stars":       meta.get("stargazers_count", 0),
        "topics":      meta.get("topics", []),
        "languages":   list(languages.keys()) if isinstance(languages, dict) else [],
        "readme":      readme_text,
        "commits":     commit_list,
        "file_tree":   file_tree,
        "owner":       owner,
    }


# ── Claude analysis ───────────────────────────────────────────────────────────

def analyze_repo(data: dict) -> dict:
    print("Analyzing with Claude...")

    commit_sample = "\n".join(
        f"  [{c['date']}] {c['message']}" for c in data["commits"][:15]
    )
    file_sample = ", ".join(data["file_tree"][:20])
    lang_str    = ", ".join(data["languages"]) or "unknown"

    prompt = f"""You are analyzing a GitHub repository to extract career-relevant insights.

Repo: {data['name']}
Description: {data['description']}
Languages: {lang_str}
Topics: {', '.join(data['topics'])}
Top files: {file_sample}

README (first 2000 chars):
{data['readme'][:2000]}

Recent commits:
{commit_sample}

Return ONLY valid JSON (no markdown, no backticks):
{{
  "skills": ["list of specific technical skills demonstrated — languages, frameworks, tools, concepts"],
  "explanation": "3-4 sentence plain English explanation of what this project does and why it matters. Suitable for a non-technical recruiter.",
  "technical_summary": "2-3 sentence technical explanation for a senior engineer.",
  "resume_bullets": [
    "Built X using Y that achieved Z — bullet 1 (lead with action verb, include specific tech)",
    "Built X using Y that achieved Z — bullet 2",
    "Built X using Y that achieved Z — bullet 3"
  ],
  "project_type": "one of: web-app, api, ml-model, ai-agent, data-pipeline, cli-tool, library, mobile-app, other",
  "complexity": "one of: beginner, intermediate, advanced"
}}"""

    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=1000,
        messages=[{"role": "user", "content": prompt}]
    )

    raw = response.content[0].text.strip()
    raw = re.sub(r"^```(?:json)?", "", raw).strip()
    raw = re.sub(r"```$", "", raw).strip()

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        print("  Warning: Claude returned invalid JSON, using defaults")
        return {
            "skills": data["languages"],
            "explanation": data["description"],
            "technical_summary": "",
            "resume_bullets": [],
            "project_type": "other",
            "complexity": "intermediate",
        }


# ── Neo4j storage ─────────────────────────────────────────────────────────────

def get_driver():
    return GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))


def run(driver, query: str, **params):
    with driver.session() as s:
        return s.run(query, **params).data()


def ensure_schema(driver):
    constraints = [
        "CREATE CONSTRAINT IF NOT EXISTS FOR (cb:Codebase) REQUIRE cb.url IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (gc:GitCommit) REQUIRE gc.sha IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (s:Skill)     REQUIRE s.name IS UNIQUE",
    ]
    for c in constraints:
        run(driver, c)


def store_in_graph(driver, repo_data: dict, analysis: dict, repo_owner_name: str):
    print("Storing in Neo4j...")

    # Codebase node
    run(driver, """
        MERGE (cb:Codebase {url: $url})
        SET cb.name              = $name,
            cb.description       = $description,
            cb.stars             = $stars,
            cb.languages         = $languages,
            cb.topics            = $topics,
            cb.explanation       = $explanation,
            cb.technical_summary = $technical_summary,
            cb.resume_bullets    = $bullets,
            cb.project_type      = $project_type,
            cb.complexity        = $complexity,
            cb.owner             = $owner
    """,
        url=repo_data["url"],
        name=repo_data["name"],
        description=repo_data["description"],
        stars=repo_data["stars"],
        languages=repo_data["languages"],
        topics=repo_data["topics"],
        explanation=analysis.get("explanation", ""),
        technical_summary=analysis.get("technical_summary", ""),
        bullets=analysis.get("resume_bullets", []),
        project_type=analysis.get("project_type", "other"),
        complexity=analysis.get("complexity", "intermediate"),
        owner=repo_owner_name,
    )

    # GitCommit nodes + relationships
    for c in repo_data["commits"]:
        run(driver, """
            MERGE (gc:GitCommit {sha: $sha})
            SET gc.message = $message, gc.date = $date
            WITH gc
            MATCH (cb:Codebase {url: $url})
            MERGE (cb)-[:HAS_COMMIT]->(gc)
        """, sha=c["sha"], message=c["message"], date=c["date"], url=repo_data["url"])

    # Normalize Claude's free-form skills to canonical SKILL_KEYWORDS names
    canonical_skills = normalize_skills(
        analysis.get("skills", []),
        repo_data.get("languages", [])
    )

    # Skill nodes + DEMONSTRATES relationships
    skills_added = 0
    for skill in canonical_skills:
        skill = skill.strip()
        if not skill:
            continue
        run(driver, """
            MERGE (s:Skill {name: $skill})
            WITH s
            MATCH (cb:Codebase {url: $url})
            MERGE (cb)-[:DEMONSTRATES]->(s)
        """, skill=skill, url=repo_data["url"])
        skills_added += 1

    # Link skills to matching roles already in graph (gap detection)
    matched = run(driver, """
        MATCH (cb:Codebase {url: $url})-[:DEMONSTRATES]->(s:Skill)
        MATCH (a:Application)
        WHERE toLower(a.notes) CONTAINS toLower(s.name)
           OR toLower(a.role)  CONTAINS toLower(s.name)
        MERGE (s)-[:RELEVANT_TO]->(a)
        RETURN count(DISTINCT a) AS matched_apps
    """, url=repo_data["url"])

    print(f"  Stored: {len(repo_data['commits'])} commits, {skills_added} skills")
    if matched and matched[0]["matched_apps"]:
        print(f"  Gap detection: {matched[0]['matched_apps']} applications linked to your skills")

    return skills_added


# ── Gap report ────────────────────────────────────────────────────────────────

def print_gap_report(driver, repo_url: str):
    results = run(driver, """
        MATCH (cb:Codebase {url: $url})-[:DEMONSTRATES]->(s:Skill)
        RETURN collect(s.name) AS your_skills
    """, url=repo_url)

    if not results:
        return

    your_skills = results[0]["your_skills"]

    jobs = run(driver, """
        MATCH (a:Application)
        WHERE a.status <> 'Discarded'
        RETURN a.num AS num, a.role AS role, a.company AS company,
               a.score AS score, a.notes AS notes
        ORDER BY a.score DESC
    """)

    if not jobs:
        return

    print("\n── Gap Detection ─────────────────────────────────────────")
    for job in jobs:
        notes = (job.get("notes") or "").lower()
        role  = (job.get("role")  or "").lower()
        hits  = [s for s in your_skills if s.lower() in notes or s.lower() in role]
        if hits:
            print(f"  #{job['num']} {job['company']} — {job['role']}")
            print(f"    Skills from your GitHub that match: {', '.join(hits)}")


# ── Main ─────────────────────────────────────────────────────────────────────

def ingest_repo(repo_url: str, repo_owner_name: str = "Satwik Valluri") -> dict:
    owner, repo = parse_repo_url(repo_url)

    repo_data = fetch_repo_data(owner, repo)
    analysis  = analyze_repo(repo_data)

    driver = get_driver()
    ensure_schema(driver)
    store_in_graph(driver, repo_data, analysis, repo_owner_name)
    print_gap_report(driver, repo_data["url"])
    driver.close()

    print(f"\n✓ {repo_data['name']} ingested into graph")
    print(f"\nExplanation:\n{analysis.get('explanation', '')}")
    print(f"\nResume bullets:")
    for b in analysis.get("resume_bullets", []):
        print(f"  • {b}")

    return {"repo": repo_data["name"], "skills": analysis.get("skills", []), "url": repo_data["url"]}


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 -m agent.knowledge.github_connector <github-url> [--owner 'Your Name']")
        sys.exit(1)

    url   = sys.argv[1]
    owner = "Satwik Valluri"
    if "--owner" in sys.argv:
        idx   = sys.argv.index("--owner")
        owner = sys.argv[idx + 1]

    ingest_repo(url, owner)
