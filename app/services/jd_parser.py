import json
from config import client, MODEL


def parse_jd_with_llm(jd_text: str) -> dict:
    prompt = f"""You are a technical recruiter assistant. Extract structured hiring requirements from the job description below.

JOB DESCRIPTION:
{jd_text[:6000]}

Return ONLY a raw JSON object — no markdown, no code fences, no extra text.
Use exactly this structure:

{{
    "job_title": "exact job title from the JD, or 'Not specified' if absent",
    "required_skills": ["skill1", "skill2"],
    "preferred_skills": ["skill1", "skill2"],
    "experience": {{"min": <int years or 0 if not stated>, "max": <int years or 10 if not stated>}},
    "education": ["degree1", "degree2"]
}}

Rules:
- required_skills: skills explicitly marked as required/must-have/mandatory
- preferred_skills: skills marked as nice-to-have/preferred/bonus, or listed separately
- If the JD doesn't distinguish required vs preferred, put all technical skills in required_skills and leave preferred_skills as []
- experience min/max must be integers (years). Use 0 and 10 as defaults if not mentioned.
- education: list degree names only (e.g. "B.Tech", "MBA", "B.Sc Computer Science")
"""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=1024,
    )

    content = response.choices[0].message.content.strip()
    content = content.removeprefix("```json").removeprefix("```").removesuffix("```").strip()

    parsed = json.loads(content)

    parsed.setdefault("job_title", "Not specified")
    parsed.setdefault("required_skills", [])
    parsed.setdefault("preferred_skills", [])
    parsed.setdefault("experience", {"min": 0, "max": 10})
    parsed.setdefault("education", [])

    return parsed


