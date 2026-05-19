RESUME_KEYWORDS = [
    "education", "experience", "skills", "projects",
    "work experience", "certifications", "internship", "summary",
]


def is_resume(resume_text: str) -> bool:
    text_lower = resume_text.lower()
    matches = sum(1 for word in RESUME_KEYWORDS if word in text_lower)
    return matches >= 2