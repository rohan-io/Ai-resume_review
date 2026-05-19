import json
from config import client, MODEL


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
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=1024,
    )

    content = response.choices[0].message.content.strip()
    content = content.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    return json.loads(content)