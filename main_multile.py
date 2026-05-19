from dotenv import load_dotenv
from groq import Groq
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
import fitz
import json
import os
import uuid
from datetime import datetime
from typing import Optional

load_dotenv()

app = FastAPI()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# ─────────────────────────────────────────────
# DIRECTORIES
# ─────────────────────────────────────────────

UPLOADS_DIR = "uploads"
RESULTS_DIR = "results"

os.makedirs(UPLOADS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


# ─────────────────────────────────────────────
# JOB DESCRIPTIONS
# ─────────────────────────────────────────────

JOB_DESCRIPTIONS = {
    "backend": {
        "required_skills": ["Python", "FastAPI", "SQL", "REST API", "Git"],
        "preferred_skills": ["Docker", "AWS"],
        "experience": {"min": 2, "max": 6},
        "education": ["MBA", "B.Tech", "MCA"],
    },
    "frontend": {
        "required_skills": ["React", "JavaScript", "HTML", "CSS"],
        "preferred_skills": ["Tailwind CSS", "TypeScript"],
        "experience": {"min": 1, "max": 5},
        "education": ["B.Tech", "MCA"],
    },
}


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
# RULE-BASED SCORING  (0–100, fully deterministic)
# ─────────────────────────────────────────────

def calculate_score(role_data: dict, resume_text: str) -> dict:
    score = 0
    resume_lower = resume_text.lower()

    # Required skills — 40 pts
    required_skills = role_data["required_skills"]
    points_per_required = 40 / len(required_skills)
    for skill in required_skills:
        if skill.lower() in resume_lower:
            score += points_per_required

    # Preferred skills — 10 pts 
    preferred_skills = role_data["preferred_skills"]
    points_per_preferred = 10 / max(len(preferred_skills), 1)
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
# SKILL CONFIDENCE  (0–10 per skill, fully deterministic)
# ─────────────────────────────────────────────

def calculate_skill_confidence(skills: list, resume_text: str) -> dict:
    confidence = {}
    resume_lower = resume_text.lower()

    for skill in skills:
        skill_lower = skill.lower()
        score = 0

        if skill_lower in resume_lower:
            score += 3

        project_patterns = [
            f"built {skill_lower}",
            f"developed {skill_lower}",
            f"using {skill_lower}",
            f"{skill_lower} project",
        ]
        if any(p in resume_lower for p in project_patterns):
            score += 4

        experience_patterns = [
            f"experience with {skill_lower}",
            f"worked on {skill_lower}",
        ]
        if any(p in resume_lower for p in experience_patterns):
            score += 3

        confidence[skill] = min(score, 10)

    return confidence


# ─────────────────────────────────────────────
# AI NARRATIVE ANALYSIS
# ─────────────────────────────────────────────

def generate_ai_analysis(role_data: dict, resume_text: str) -> dict:
    prompt = f"""You are a senior technical recruiter writing a concise, professional evaluation of a candidate's resume.

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
# STORAGE HELPERS
# ─────────────────────────────────────────────

def save_result(result: dict) -> None:
    path = os.path.join(RESULTS_DIR, f"{result['resume_id']}.json")
    with open(path, "w") as f:
        json.dump(result, f, indent=2)


def load_result(resume_id: str) -> dict | None:
    path = os.path.join(RESULTS_DIR, f"{resume_id}.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def load_all_results() -> list[dict]:
    results = []
    for fname in os.listdir(RESULTS_DIR):
        if fname.endswith(".json"):
            with open(os.path.join(RESULTS_DIR, fname)) as f:
                results.append(json.load(f))
    results.sort(key=lambda r: r.get("evaluated_at", ""), reverse=True)
    return results


def delete_result(resume_id: str) -> bool:
    path = os.path.join(RESULTS_DIR, f"{resume_id}.json")
    if not os.path.exists(path):
        return False
    os.remove(path)
    return True


# ─────────────────────────────────────────────
# CLEAN RESPONSE HELPER
# ─────────────────────────────────────────────

def clean_response(result: dict) -> dict:
    """Strip internal fields (resume_id) from API responses."""
    exclude = {"resume_id"}
    return {k: v for k, v in result.items() if k not in exclude}


# ─────────────────────────────────────────────
# CORE EVALUATION PIPELINE
# ─────────────────────────────────────────────

async def evaluate_single(role: str, role_data: dict, resume_file: UploadFile) -> dict:
    resume_id = str(uuid.uuid4())
    filename  = resume_file.filename or f"{resume_id}.pdf"
    file_path = os.path.join(UPLOADS_DIR, f"{resume_id}_{filename}")

    with open(file_path, "wb") as f:
        f.write(await resume_file.read())

    resume_text = extract_text_from_pdf(file_path)

    # Guard: too short
    if len(resume_text.strip()) < 200:
        return {
            "resume_id"   : resume_id,
            "filename"    : filename,
            "role"        : role,
            "status"      : "failed",
            "error"       : "Resume appears empty or too short to evaluate.",
            "evaluated_at": datetime.utcnow().isoformat(),
        }

    # Guard: not a resume
    if not is_resume(resume_text):
        return {
            "resume_id"   : resume_id,
            "filename"    : filename,
            "role"        : role,
            "status"      : "failed",
            "error"       : "Uploaded file does not appear to be a resume.",
            "evaluated_at": datetime.utcnow().isoformat(),
        }

    # Guard: resume has no relevant skills for the role
    resume_lower = resume_text.lower()
    matched_required = sum(1 for s in role_data["required_skills"] if s.lower() in resume_lower)
    if matched_required == 0:
        return {
            "resume_id"   : resume_id,
            "filename"    : filename,
            "role"        : role,
            "status"      : "failed",
            "error"       : f"Resume has no relevant skills for the '{role}' role.",
            "evaluated_at": datetime.utcnow().isoformat(),
        }

    # Scoring
    score_result     = calculate_score(role_data, resume_text)
    all_skills       = role_data["required_skills"] + role_data["preferred_skills"]
    skill_confidence = calculate_skill_confidence(all_skills, resume_text)

    # AI narrative
    try:
        ai_result = generate_ai_analysis(role_data, resume_text)
    except Exception as e:
        return {
            "resume_id"   : resume_id,
            "filename"    : filename,
            "role"        : role,
            "status"      : "failed",
            "error"       : f"AI analysis failed: {str(e)}",
            "evaluated_at": datetime.utcnow().isoformat(),
        }

    return {
        "resume_id"       : resume_id,
        "filename"        : filename,
        "role"            : role,
        "status"          : "success",
        "evaluated_at"    : datetime.utcnow().isoformat(),
        "score"           : score_result["score"],
        "category"        : score_result["category"],
        "summary"         : ai_result.get("summary"),
        "strengths"       : ai_result.get("strengths"),
        "weaknesses"      : ai_result.get("weaknesses"),
        "recommendation"  : ai_result.get("recommendation"),
        "skill_confidence": skill_confidence,
    }


# ─────────────────────────────────────────────
# ENDPOINTS
# ─────────────────────────────────────────────

# ── 1. Single resume ──
@app.post("/evaluate")
async def evaluate_resume(
    role: str = Form(...),
    resume: UploadFile = File(...),
):
    role_data = JOB_DESCRIPTIONS.get(role)
    if not role_data:
        raise HTTPException(status_code=400, detail=f"Invalid role '{role}'. Choose from: {list(JOB_DESCRIPTIONS.keys())}")

    result = await evaluate_single(role, role_data, resume)

    if result["status"] == "failed":
        raise HTTPException(status_code=400, detail=result["error"])

    save_result(result)
    return clean_response(result)


# ── 2. Batch evaluate (up to 5 resumes) ──
@app.post("/evaluate/batch")
async def evaluate_batch(
    role: str = Form(...),
    resumes: list[UploadFile] = File(...),
):
    if len(resumes) > 5:
        raise HTTPException(status_code=400, detail="Maximum 5 resumes at a time.")

    role_data = JOB_DESCRIPTIONS.get(role)
    if not role_data:
        raise HTTPException(status_code=400, detail=f"Invalid role '{role}'. Choose from: {list(JOB_DESCRIPTIONS.keys())}")

    results, succeeded, failed = [], 0, 0

    for resume_file in resumes:
        result = await evaluate_single(role, role_data, resume_file)
        save_result(result)
        results.append(result)
        if result["status"] == "success":
            succeeded += 1
        else:
            failed += 1

    # Sort: successful first, then by score desc
    results.sort(key=lambda r: (r["status"] == "success", r.get("score", 0)), reverse=True)

    return {
        "total"    : len(resumes),
        "succeeded": succeeded,
        "failed"   : failed,
        "results"  : [clean_response(r) for r in results],
    }


# ── 3. List all stored results ──
@app.get("/results")
async def list_results(
    role: Optional[str] = None,
    status: Optional[str] = None,
    min_score: Optional[int] = None,
):
    all_results = load_all_results()

    if role:
        all_results = [r for r in all_results if r.get("role") == role]
    if status:
        all_results = [r for r in all_results if r.get("status") == status]
    if min_score is not None:
        all_results = [r for r in all_results if r.get("score", 0) >= min_score]

    summary = [
        {
            "filename"      : r["filename"],
            "role"          : r["role"],
            "status"        : r["status"],
            "score"         : r.get("score"),
            "category"      : r.get("category"),
            "recommendation": r.get("recommendation"),
            "evaluated_at"  : r["evaluated_at"],
            "error"         : r.get("error"),
        }
        for r in all_results
    ]

    return {"total": len(summary), "results": summary}


# ── 4. Full detail of one resume ──
@app.get("/results/{resume_id}")
async def get_result(resume_id: str):
    result = load_result(resume_id)
    if result is None:
        raise HTTPException(status_code=404, detail=f"No result found for '{resume_id}'.")
    return clean_response(result)


# ── 5. Delete a result ──
@app.delete("/results/{resume_id}")
async def delete_resume_result(resume_id: str):
    if not delete_result(resume_id):
        raise HTTPException(status_code=404, detail=f"No result found for '{resume_id}'.")
    return {"message": f"Result '{resume_id}' deleted successfully."}


# ── 6. Compare / ranked leaderboard ──
@app.get("/compare")
async def compare_results(role: str):
    role_data = JOB_DESCRIPTIONS.get(role)
    if not role_data:
        raise HTTPException(status_code=400, detail=f"Invalid role '{role}'. Choose from: {list(JOB_DESCRIPTIONS.keys())}")

    all_results = [
        r for r in load_all_results()
        if r.get("role") == role and r.get("status") == "success"
    ]

    if not all_results:
        return {"message": f"No successful evaluations found for role '{role}'.", "rankings": []}

    all_results.sort(key=lambda r: r.get("score", 0), reverse=True)

    rankings = [
        {
            "rank"            : i + 1,
            "filename"        : r["filename"],
            "score"           : r["score"],
            "category"        : r["category"],
            "recommendation"  : r["recommendation"],
            "summary"         : r["summary"],
            "skill_confidence": r["skill_confidence"],
            "evaluated_at"    : r["evaluated_at"],
        }
        for i, r in enumerate(all_results)
    ]

    return {"role": role, "total": len(rankings), "rankings": rankings}