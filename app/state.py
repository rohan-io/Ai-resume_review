# ─────────────────────────────────────────────
# IN-MEMORY JD STORE
# Replace this module with a Redis/DB layer later
# without touching any other file.
# ─────────────────────────────────────────────

current_jd: dict | None = None


def get_jd() -> dict | None:
    return current_jd


def set_jd(jd: dict) -> None:
    global current_jd
    current_jd = jd


def clear_jd() -> None:
    global current_jd
    current_jd = None