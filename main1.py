from dotenv import load_dotenv
from groq import Groq
from fastapi import FastAPI, UploadFile, File, HTTPException
import fitz
import json
import os
import re

load_dotenv()

app = FastAPI()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))


# ─────────────────────────────────────────────
# IN-MEMORY JD STORE
# ─────────────────────────────────────────────
current_jd: dict | None = None


# ─────────────────────────────────────────────
# RESUME VALIDATION
# ─────────────────────────────────────────────
RESUME_KEYWORDS = [
    "education", "experience", "skills", "projects",
    "work experience", "certifications", "internship", "summary",
]


def is_resume(resume_text: str) -> bool:
    text_lower = resume_text.lower()
    matches = sum(1 for word in RESUME_KEYWORDS if word in text_lower)
    return matches >= 2


# ─────────────────────────────────────────────
# PDF EXTRACTION
# ─────────────────────────────────────────────
def extract_text_from_pdf(pdf_path: str) -> str:
    doc = fitz.open(pdf_path)
    return "".join(page.get_text() for page in doc)


# ─────────────────────────────────────────────
# JD PARSING (LLM)
# ─────────────────────────────────────────────
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
        model="llama-3.3-70b-versatile",
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


# ─────────────────────────────────────────────
# RULE-BASED SCORING (0–100, fully deterministic)
# ─────────────────────────────────────────────
def calculate_score(role_data: dict, resume_text: str) -> dict:
    score = 0
    resume_lower = resume_text.lower()

    # Required skills — 40 pts
    required_skills = role_data["required_skills"]
    if required_skills:
        points_per_required = 40 / len(required_skills)
        for skill in required_skills:
            if skill.lower() in resume_lower:
                score += points_per_required

    # Preferred skills — 10 pts
    preferred_skills = role_data["preferred_skills"]
    if preferred_skills:
        points_per_preferred = 10 / len(preferred_skills)
        for skill in preferred_skills:
            if skill.lower() in resume_lower:
                score += points_per_preferred

    # Experience evidence — 20 pts
    experience_keywords = ["years", "experience", "worked", "developer", "engineer"]
    experience_matches = sum(1 for w in experience_keywords if w in resume_lower)
    score += min(experience_matches * 4, 20)

    # Project evidence — 20 pts
    project_keywords = ["project", "developed", "built", "implemented", "designed", "api"]
    project_matches = sum(1 for w in project_keywords if w in resume_lower)
    score += min(project_matches * 3, 20)

    # Education — 10 pts
    for edu in role_data["education"]:
        if edu.lower() in resume_lower:
            score += 10
            break

    score = min(int(score), 100)

    if score >= 88:
        category = "Excellent Match"
    elif score >= 75:
        category = "Best Match"
    elif score >= 60:
        category = "Good Match"
    elif score >= 45:
        category = "Average Match"
    else:
        category = "Weak Match"

    return {"score": score, "category": category}


# ─────────────────────────────────────────────
# SKILL CONFIDENCE (0–10 per skill)
# ─────────────────────────────────────────────
def _get_ai_skill_scores(skills: list, resume_text: str) -> dict:
    """
    Asks Groq to score project usage (0 or 3) and experience depth (0 or 2)
    for each skill in one single API call.
    """
    skill_list = ", ".join(skills)

    prompt = f"""
You are evaluating a resume for skill evidence.

For each skill below, return a JSON object with two keys:
- "project": 3 if the resume shows the candidate actively built or implemented something with this skill, else 0
- "experience": 2 if the resume shows professional experience or proficiency with this skill, else 0

Be generous with paraphrasing — "spearheaded", "owned", "leveraged" all count.

Skills: {skill_list}

Resume:
{resume_text[:5000]}

Respond ONLY with valid JSON. Example format:
{{
  "React": {{"project": 3, "experience": 2}},
  "GraphQL": {{"project": 0, "experience": 2}}
}}
"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=1024,
    )

    content = response.choices[0].message.content.strip()
    content = content.removeprefix("```json").removeprefix("```").removesuffix("```").strip()

    try:
        return json.loads(content)
    except json.JSONDecodeError:
        return {}  # fallback: all skills get 0 for project/experience


def calculate_skill_confidence(skills: list, resume_text: str) -> dict:
    resume_lower = resume_text.lower()
    confidence = {}

    # AI call: project usage + experience phrases (one shot for all skills)
    ai_scores = _get_ai_skill_scores(skills, resume_text)

    for skill in skills:
        s = skill.lower()
        score = 0

        # bare mention
        if s in resume_lower:
            score += 2

        # repeat mentions (depth signal)
        if resume_lower.count(s) >= 3:
            score += 2

        # years of experience
        if re.search(rf'\d+\+?\s+years?\s+of\s+(?:\w+\s+)?{re.escape(s)}', resume_lower):
            score += 4

        # AI-judged project usage + experience
        score += ai_scores.get(skill, {}).get("project", 0)
        score += ai_scores.get(skill, {}).get("experience", 0)

        confidence[skill] = min(score, 10)

    return confidence


# ─────────────────────────────────────────────
# AI NARRATIVE ANALYSIS
# ─────────────────────────────────────────────
def generate_ai_analysis(role_data: dict, resume_text: str) -> dict:
    prompt = f"""You are a senior technical recruiter writing a concise, professional evaluation of a candidate's resume.

JOB TITLE: {role_data.get('job_title', 'Not specified')}

JOB REQUIREMENTS:
- Required Skills : {role_data['required_skills']}
- Preferred Skills: {role_data['preferred_skills']}
- Experience Range: {role_data['experience']['min']}–{role_data['experience']['max']} years
- Education       : {role_data['education']}

RESUME:
{resume_text[:5000]}

Write a fluid, human-quality evaluation. Do NOT produce robotic, template-style sentences such as
"X has experience with Y as listed in their technical stack." or "X's experience with Y is not explicitly backed by a project."
Instead write the way a thoughtful senior recruiter would — referencing specific projects, roles,
measurable achievements, and real context pulled from the resume.

OUTPUT RULES — follow exactly, return only a raw JSON object (no markdown, no code fences):

1. "summary"
   2 sentences max. Sentence 1: who they are, years of experience, current role and company,
   and core specialisation. Sentence 2: one standout fact or achievement that defines their profile.
   Do NOT use the word "candidate". No filler adjectives like "seasoned", "dynamic", or "passionate".

2. "strengths"
   Exactly 3 to 4 items. Each is ONE tight sentence — max 20 words — grounded in a specific
   project, metric, or achievement from the resume. Lead with the impact or skill, not the person's
   name. No filler phrases like "strong background", "proven track record", or "passionate about".

3. "weaknesses"
   Exactly 2 to 3 items. Each is ONE tight sentence — max 18 words — naming a concrete gap:
   a required skill that is absent, unverified, or not backed by any real project or job.
   Be direct and specific. No hedging.

4. "recommendation"
   Exactly one of: Excellent Match | Best Match | Good Match | Average Match | Weak Match

{{
    "summary": "",
    "strengths": [],
    "weaknesses": [],
    "recommendation": ""
}}"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=1024,
    )

    content = response.choices[0].message.content.strip()
    content = content.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    return json.loads(content)


# ─────────────────────────────────────────────
# ENDPOINTS
# ─────────────────────────────────────────────
@app.post("/upload-jd")
async def upload_jd(jd: UploadFile = File(...)):
    global current_jd

    if not jd.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted for the JD.")

    os.makedirs("uploads", exist_ok=True)
    jd_path = f"uploads/jd_{jd.filename}"
    with open(jd_path, "wb") as f:
        f.write(await jd.read())

    jd_text = extract_text_from_pdf(jd_path)

    if len(jd_text.strip()) < 100:
        raise HTTPException(
            status_code=400,
            detail="JD PDF appears empty or too short to parse.",
        )

    try:
        parsed = parse_jd_with_llm(jd_text)
    except (json.JSONDecodeError, KeyError) as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to parse JD structure from LLM output: {str(e)}",
        )

    current_jd = parsed

    return {
        "message"         : "JD uploaded and parsed successfully.",
        "job_title"       : parsed["job_title"],
        "required_skills" : parsed["required_skills"],
        "preferred_skills": parsed["preferred_skills"],
        "experience"      : parsed["experience"],
        "education"       : parsed["education"],
    }


@app.get("/jd-status")
async def jd_status():
    if current_jd is None:
        return {"loaded": False, "message": "No JD uploaded yet. POST to /upload-jd first."}
    return {
        "loaded"          : True,
        "job_title"       : current_jd["job_title"],
        "required_skills" : current_jd["required_skills"],
        "preferred_skills": current_jd["preferred_skills"],
        "experience"      : current_jd["experience"],
        "education"       : current_jd["education"],
    }


@app.post("/evaluate")
async def evaluate_resume(resume: UploadFile = File(...)):
    if current_jd is None:
        raise HTTPException(
            status_code=400,
            detail="No JD loaded. Upload a job description first via POST /upload-jd.",
        )

    os.makedirs("uploads", exist_ok=True)
    file_path = f"uploads/{resume.filename}"
    with open(file_path, "wb") as f:
        f.write(await resume.read())

    resume_text = extract_text_from_pdf(file_path)

    if len(resume_text.strip()) < 200:
        raise HTTPException(
            status_code=400,
            detail="Resume appears empty or too short to evaluate.",
        )

    if not is_resume(resume_text):
        raise HTTPException(
            status_code=400,
            detail="Uploaded file does not appear to be a resume.",
        )

    # Rule-based score (deterministic)
    score_result = calculate_score(current_jd, resume_text)

    # Skill confidence out of 10
    all_skills = current_jd["required_skills"] + current_jd["preferred_skills"]
    skill_confidence = calculate_skill_confidence(all_skills, resume_text)

    # AI narrative
    try:
        ai_result = generate_ai_analysis(current_jd, resume_text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI analysis failed: {str(e)}")

    return {
        "job_title"       : current_jd["job_title"],
        "score"           : score_result["score"],
        "category"        : score_result["category"],
        "summary"         : ai_result.get("summary"),
        "strengths"       : ai_result.get("strengths"),
        "weaknesses"      : ai_result.get("weaknesses"),
        "recommendation"  : ai_result.get("recommendation"),
        "skill_confidence": skill_confidence,
    }