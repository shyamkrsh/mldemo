from flask import Flask, request, render_template
import joblib
import numpy as np

app = Flask(__name__)

model = joblib.load("house_price_model.pkl")

@app.route("/")
def home():
    # return "Hello World, from Flask!"  # delete after run successfully this line
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        area = float(request.form["area"])
        bedrooms = int(request.form["bedrooms"])
        bathrooms = int(request.form["bathrooms"])
        floors = int(request.form["floors"])
        condition = int(request.form["condition"])

        input_features = np.array([[area, bedrooms, bathrooms, floors, condition]])
        prediction = model.predict(input_features)[0]

        return render_template("index.html", prediction_text=f"Estimated House Price: Rs {prediction:,.2f}")

    except Exception as e:
        return render_template("index.html", prediction_text="Error: Invalid input!")

# if __name__ == "__main__":
#     app.run(debug=True)
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)