import re
import json
from config import client, MODEL


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
        model=MODEL,
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


# def calculate_skill_confidence(skills: list, resume_text: str) -> dict:
#     resume_lower = resume_text.lower()
#     confidence = {}
    
#     # AI call: project usage + experience phrases (one shot for all skills)
#     ai_scores = _get_ai_skill_scores(skills, resume_text)

#     for skill in skills:
#         s = skill.lower()
#         score = 0

#         # bare mention
#         if s in resume_lower:
#             score += 2          

#         # repeat mentions
#         if resume_lower.count(s) >= 3:
#             score += 1          

#         # years of experience
#         if re.search(rf'\d+\+?\s+years?\s+of\s+(?:\w+\s+)?{re.escape(s)}', resume_lower):
#             score += 3         
#         # AI project usage
#         score += ai_scores.get(skill, {}).get("project", 0)   # keep 0 or 3

#         # AI experience depth
#         score += ai_scores.get(skill, {}).get("experience", 0) # keep 0 or 2

#         # 2 + 1 + 3 + 3 + 2 = 11... still off
#         confidence[skill] = min(score, 10)

#     return confidence

## if skills and gaps need to be separate, we can return them as two dicts:

def calculate_skill_confidence(skills: list, resume_text: str) -> dict:
    resume_lower = resume_text.lower()
    matched = {}
    gaps = []

    ai_scores = _get_ai_skill_scores(skills, resume_text)

    for skill in skills:
        s = skill.lower()
        score = 0

        if s in resume_lower:
            score += 2
        if resume_lower.count(s) >= 3:
            score += 1
        if re.search(rf'\d+\+?\s+years?\s+of\s+(?:\w+\s+)?{re.escape(s)}', resume_lower):
            score += 3

        score += ai_scores.get(skill, {}).get("project", 0)
        score += ai_scores.get(skill, {}).get("experience", 0)

        final = min(score, 10)

        if final > 0:
            matched[skill] = final
        else:
            gaps.append(skill)

    return {"matched": matched, "gaps": gaps}