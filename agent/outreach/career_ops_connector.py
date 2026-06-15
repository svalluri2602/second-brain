import re
from pathlib import Path
from agent.knowledge.vector_store import search_relevant, generate_outreach

CAREER_OPS = Path(__file__).parent.parent.parent / "career-ops"
SENDER = "Satwik Valluri"


def get_top_jobs(min_score: float = 4.0) -> list:
    top_jobs = []
    for report_path in sorted((CAREER_OPS / "reports").glob("*.md")):
        content = report_path.read_text(encoding="utf-8")
        match = re.search(r'\*\*Score:\*\*\s*([\d.]+)', content)
        if not match:
            continue
        try:
            score = float(match.group(1))
        except ValueError:
            continue
        if score >= min_score:
            company_match = re.search(r'# Evaluation: (.+?) —', content)
            company = company_match.group(1).strip() if company_match else report_path.stem
            top_jobs.append({
                "file": str(report_path),
                "company": company,
                "score": score,
                "content": content,
            })
    return sorted(top_jobs, key=lambda x: x["score"], reverse=True)


def generate_outreach_for_top_jobs(min_score: float = 4.0):
    jobs = get_top_jobs(min_score)
    if not jobs:
        print("No high-scoring jobs found. Run /career-ops to evaluate roles first.")
        return

    for job in jobs[:3]:
        print(f"\n{'─'*50}")
        print(f"Company: {job['company']}  |  Score: {job['score']}/5")
        relevant = search_relevant(job["content"][:1000])
        message = generate_outreach(relevant, job["content"][:600], "Hiring Manager", SENDER)
        print(f"Message:\n{message}")


if __name__ == "__main__":
    generate_outreach_for_top_jobs()