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