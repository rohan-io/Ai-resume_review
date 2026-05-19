# AI Resume Screener — Backend

A FastAPI backend that parses a Job Description (JD) and evaluates resumes against it using rule-based scoring, skill confidence analysis, and LLM-generated recruiter narrative.

---

## Tech Stack

- **FastAPI** — API framework
- **Groq (LLaMA 3.3 70B)** — LLM for JD parsing and AI analysis
- **PyMuPDF / pdfplumber** — PDF text extraction
- **Python 3.12**

---

## Project Structure

```
backend/
├── app/
│   ├── main.py                  # FastAPI app entry point
│   ├── config.py                # Groq client + model config
│   ├── state.py                 # In-memory JD state
│   ├── routers/
│   │   └── resume.py            # All API endpoints
│   ├── services/
│   │   ├── jd_parser.py         # LLM-based JD parsing
│   │   ├── scorer.py            # Rule-based resume scoring
│   │   ├── skill_confidence.py  # Skill confidence scoring (0–10)
│   │   └── ai_analysis.py       # LLM recruiter narrative
│   └── utils/
│       ├── pdf.py               # PDF text extraction
│       └── resume_validator.py  # Resume validity check
├── uploads/                     # Uploaded files (auto-created)
├── results/                     # Output results (auto-created)
├── requirements.txt
└── .env                         # API keys (not committed)
```

---

## Getting Started

### 1. Clone the repo

```bash
git clone https://github.com/rohan-io/Ai-resume_review.git
cd Ai-resume_review
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set up environment variables

Create a `.env` file in the `backend/` root:

```env
GROQ_API_KEY=your_groq_api_key_here
```

Get a free API key at [console.groq.com](https://console.groq.com).

### 5. Run the server

```bash
cd app
uvicorn main:app --reload
```

Server runs at `http://localhost:8000`.  
Interactive docs at `http://localhost:8000/docs`.

---

## API Endpoints

### `POST /upload-jd`
Upload a Job Description PDF. Parses it with LLM and stores it in memory.

**Request:** `multipart/form-data` — field `jd` (PDF file)

**Response:**
```json
{
  "message": "JD uploaded and parsed successfully.",
  "job_title": "Backend Engineer",
  "required_skills": ["Python", "PostgreSQL"],
  "preferred_skills": ["Redis", "Docker"],
  "experience": { "min": 2, "max": 5 },
  "education": ["B.Tech", "B.Sc Computer Science"]
}
```

---

### `GET /jd-status`
Check if a JD is currently loaded.

**Response:**
```json
{
  "loaded": true,
  "job_title": "Backend Engineer",
  ...
}
```

---

### `POST /evaluate`
Evaluate a single resume PDF against the loaded JD.

**Request:** `multipart/form-data` — field `resume` (PDF file)

**Response:**
```json
{
  "job_title": "Backend Engineer",
  "score": 78,
  "category": "Good Match",
  "summary": "...",
  "strengths": ["...", "..."],
  "weaknesses": ["...", "..."],
  "recommendation": "Good Match",
  "skill_confidence": {
    "matched": { "Python": 9, "PostgreSQL": 7 },
    "gaps": ["Kubernetes"]
  }
}
```

---

### `POST /evaluate-bulk`
Evaluate 2–20 resumes at once and get a ranked leaderboard.

**Request:** `multipart/form-data` — field `resumes` (multiple PDF files)

**Response:**
```json
{
  "job_title": "Backend Engineer",
  "total": 5,
  "evaluated": 4,
  "skipped": 1,
  "rankings": [
    { "rank": 1, "filename": "alice.pdf", "score": 88, ... },
    { "rank": 2, "filename": "bob.pdf", "score": 74, ... }
  ],
  "skipped_files": [
    { "filename": "random.pdf", "status": "skipped", "reason": "..." }
  ]
}
```

---

## Notes

- JD state is stored **in memory** — it resets on server restart. Upload the JD again after restarting.
- Only **PDF** files are accepted for both JD and resumes.
- Bulk evaluation skips invalid files silently rather than failing the whole batch.
- The `uploads/` folder is auto-created but not committed to git.
