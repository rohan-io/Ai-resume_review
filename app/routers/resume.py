import os
import json

from fastapi import APIRouter, UploadFile, File, HTTPException
from typing import List
import state
from utils.pdf import extract_text_from_pdf
from utils.resume_validator import is_resume
from services.jd_parser import parse_jd_with_llm
from services.scorer import calculate_score
from services.skill_confidence import calculate_skill_confidence
from services.ai_analysis import generate_ai_analysis

router = APIRouter()


@router.post("/upload-jd")
async def upload_jd(jd: UploadFile = File(...)):
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

    state.set_jd(parsed)

    return {
        "message"         : "JD uploaded and parsed successfully.",
        "job_title"       : parsed["job_title"],
        "required_skills" : parsed["required_skills"],
        "preferred_skills": parsed["preferred_skills"],
        "experience"      : parsed["experience"],
        "education"       : parsed["education"],
    }


@router.get("/jd-status")
async def jd_status():
    jd = state.get_jd()
    if jd is None:
        return {"loaded": False, "message": "No JD uploaded yet. POST to /upload-jd first."}
    return {
        "loaded"          : True,
        "job_title"       : jd["job_title"],
        "required_skills" : jd["required_skills"],
        "preferred_skills": jd["preferred_skills"],
        "experience"      : jd["experience"],
        "education"       : jd["education"],
    }


@router.post("/evaluate")
async def evaluate_resume(resume: UploadFile = File(...)):
    jd = state.get_jd()
    if jd is None:
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
    score_result = calculate_score(jd, resume_text)

    # Skill confidence out of 10
    all_skills = jd["required_skills"] + jd["preferred_skills"]
    skill_confidence = calculate_skill_confidence(all_skills, resume_text)

    # AI narrative
    try:
        ai_result = generate_ai_analysis(jd, resume_text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI analysis failed: {str(e)}")

    return {
        "job_title"       : jd["job_title"],
        "score"           : score_result["score"],
        "category"        : score_result["category"],
        "summary"         : ai_result.get("summary"),
        "strengths"       : ai_result.get("strengths"),
        "weaknesses"      : ai_result.get("weaknesses"),
        "recommendation"  : ai_result.get("recommendation"),
        "skill_confidence": skill_confidence,
        # 'skill_confidence': {k: v for k, v in skill_confidence.items() if v > 0},
    }


    # skill_result = calculate_skill_confidence(all_skills, resume_text)

    # return {
    #     "job_title"       : jd["job_title"],
    #     "score"           : score_result["score"],
    #     "category"        : score_result["category"],
    #     "summary"         : ai_result.get("summary"),
    #     "strengths"       : ai_result.get("strengths"),
    #     "weaknesses"      : ai_result.get("weaknesses"),
    #     "recommendation"  : ai_result.get("recommendation"),
    #     "matched_skills"  : skill_result["matched"],
    #     "skill_gaps"      : skill_result["gaps"],
    #     # 'skill_confidence': {k: v for k, v in skill_confidence.items() if v > 0},
    # }


@router.post("/evaluate-bulk")
async def evaluate_bulk(resumes: List[UploadFile] = File(...)):
    jd = state.get_jd()
    if jd is None:
        raise HTTPException(
            status_code=400,
            detail="No JD loaded. Upload a job description first via POST /upload-jd.",
        )

    if len(resumes) < 2:
        raise HTTPException(
            status_code=400,
            detail="Send at least 2 resumes for bulk evaluation.",
        )

    if len(resumes) > 20:
        raise HTTPException(
            status_code=400,
            detail="Maximum 20 resumes per bulk request.",
        )

    os.makedirs("uploads", exist_ok=True)
    results = []

    for resume in resumes:
        file_path = f"uploads/{resume.filename}"
        with open(file_path, "wb") as f:
            f.write(await resume.read())

        resume_text = extract_text_from_pdf(file_path)

        # skip silently if file is invalid — don't crash the whole batch
        if len(resume_text.strip()) < 200 or not is_resume(resume_text):
            results.append({
                "filename"  : resume.filename,
                "status"    : "skipped",
                "reason"    : "File too short or does not appear to be a resume.",
            })
            continue

        score_result  = calculate_score(jd, resume_text)
        all_skills    = jd["required_skills"] + jd["preferred_skills"]
        skill_result  = calculate_skill_confidence(all_skills, resume_text)

        try:
            ai_result = generate_ai_analysis(jd, resume_text)
        except Exception:
            ai_result = {
                "summary"       : None,
                "strengths"     : [],
                "weaknesses"    : [],
                "recommendation": "N/A",
            }

        results.append({
            "filename"      : resume.filename,
            "status"        : "evaluated",
            "score"         : score_result["score"],
            "category"      : score_result["category"],
            "summary"       : ai_result.get("summary"),
            "strengths"     : ai_result.get("strengths"),
            "weaknesses"    : ai_result.get("weaknesses"),
            "recommendation": ai_result.get("recommendation"),
            "matched_skills": skill_result["matched"],
            "skill_gaps"    : skill_result["gaps"],
        })

    # separate evaluated from skipped, sort evaluated by score desc
    evaluated = [r for r in results if r["status"] == "evaluated"]
    skipped   = [r for r in results if r["status"] == "skipped"]

    evaluated.sort(key=lambda x: x["score"], reverse=True)

    # add rank after sorting
    for i, r in enumerate(evaluated, start=1):
        r["rank"] = i

    return {
        "job_title" : jd["job_title"],
        "total"     : len(resumes),
        "evaluated" : len(evaluated),
        "skipped"   : len(skipped),
        "rankings"  : evaluated,
        "skipped_files": skipped,
    }