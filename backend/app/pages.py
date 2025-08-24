from flask import Blueprint, render_template


pages_bp = Blueprint("pages", __name__)


@pages_bp.get("/")
def home():
    return render_template("home.html")


@pages_bp.get("/login")
def login():
    return render_template("login.html")


@pages_bp.get("/record")
def record():
    return render_template("record.html")


@pages_bp.get("/dashboard")
def dashboard():
    return render_template("dashboard.html")


@pages_bp.get("/profile")
def profile():
    return render_template("profile.html")


@pages_bp.get("/spiral")
def spiral():
    return render_template("spiral.html")


@pages_bp.get("/demographics")
def demographics():
    return render_template("demographics.html")


@pages_bp.get("/treatments")
def treatments():
    return render_template("treatments.html")


@pages_bp.get("/assessment")
def complete_assessment():
    return render_template("assessment.html")


@pages_bp.get("/logs")
def logs():
    return render_template("logs.html")


## Removed timeline and insights routes per update


@pages_bp.get("/export")
def export():
    return render_template("export.html")


@pages_bp.get("/activity")
def activity():
    return render_template("activity.html")


@pages_bp.get("/ai-coach")
def ai_coach():
    return render_template("ai-coach.html")


