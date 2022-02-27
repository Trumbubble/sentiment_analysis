from flask import Flask, render_template, request, flash

app = Flask(__name__)

@app.route("/hello")
def index():
    flash("what's your name?")
    return render_template("index.html")

