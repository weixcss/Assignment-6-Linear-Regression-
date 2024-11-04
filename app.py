# app.py

from flask import Flask, render_template, request
import numpy as np
import matplotlib.pyplot as plt
import io
import base64
from sklearn.linear_model import LinearRegression

app = Flask(__name__)

import matplotlib
matplotlib.use('Agg')

def generate_random_data(N, mu, sigma2):
    X = np.random.rand(N, 1)
    Y = mu + np.sqrt(sigma2) * np.random.randn(N, 1)
    return X, Y

def plot_regression(X, Y):
    model = LinearRegression()
    model.fit(X, Y)
    slope = model.coef_[0][0]
    intercept = model.intercept_[0]
    
    plt.figure(figsize=(8, 6))
    plt.scatter(X, Y, color='blue', label="Data Points")
    plt.plot(X, model.predict(X), color='red', label=f"Linear Fit: y = {intercept:.2f} + {slope:.2f}x")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.title(f"Linear Fit: y = {intercept:.2f} + {slope:.2f}x")    
    
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode('utf8')
    plt.close()
    return f"data:image/png;base64,{plot_url}", slope, intercept

def plot_histograms(slopes, intercepts, init_slope, init_intercept):
    plt.figure(figsize=(8, 6))
    plt.hist(slopes, bins=30, alpha=0.5, label="Slopes", color="blue")
    plt.hist(intercepts, bins=30, alpha=0.5, label="Intercepts", color="brown")
    plt.axvline(init_slope, color="blue", linestyle="--", label=f"Slope: {init_slope:.2f}")
    plt.axvline(init_intercept, color="orange", linestyle="--", label=f"Intercept: {init_intercept:.2f}")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.legend()
    plt.title("Histograms of Slopes and Intercepts")    
    
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode('utf8')
    plt.close()
    return f"data:image/png;base64,{plot_url}"

@app.route("/", methods=["GET", "POST"])
def index():
    plot1 = None
    plot2 = None
    slope_extreme = None
    intercept_extreme = None

    if request.method == "POST":
        N = int(request.form["N"])
        mu = float(request.form["mu"])
        sigma2 = float(request.form["sigma2"])
        S = int(request.form["S"])

        X, Y = generate_random_data(N, mu, sigma2)
        plot1, init_slope, init_intercept = plot_regression(X, Y)

        slopes = []
        intercepts = []
        for _ in range(S):
            X_sim, Y_sim = generate_random_data(N, mu, sigma2)
            model = LinearRegression()
            model.fit(X_sim, Y_sim)
            slopes.append(model.coef_[0][0])
            intercepts.append(model.intercept_[0])

        plot2 = plot_histograms(slopes, intercepts, init_slope, init_intercept)

        slope_extreme = np.mean(np.abs(slopes) > abs(init_slope))
        intercept_extreme = np.mean(np.abs(intercepts) > abs(init_intercept))

    return render_template("index.html", plot1=plot1, plot2=plot2, slope_extreme=slope_extreme, intercept_extreme=intercept_extreme)

if __name__ == "__main__":
    app.run(debug=True)
