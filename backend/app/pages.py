from flask import Blueprint, render_template


pages_bp = Blueprint("pages", __name__)


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


