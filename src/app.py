import joblib
import pandas as pd
from flask import Flask, request, render_template_string

model = joblib.load(r"C:\Users\Amin Zaini\Downloads\junior-ds-takehome\DS-Assignment-Delivery-Time-Prediction\src\delivery_time_poly_model_with_pipeline.pkl")

app = Flask(__name__)

HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Delivery Time Predictor</title>
</head>
<body style="font-family: Arial; margin: 40px;">

<h2>Delivery Time Prediction</h2>

<form method="post">

    Delivery Date:<br>
    <input type="date" name="delivery_date" required><br><br>

    Distance (km):<br>
    <input type="number" step="0.1" name="distance_km" required><br><br>

    Traffic Level:<br>
    <select name="traffic_level">
        <option value="Low">Low</option>
        <option value="Medium">Medium</option>
        <option value="High">High</option>
    </select><br><br>

    Weather Condition:<br>
    <select name="weather_condition">
        <option value="Sunny">Sunny</option>
        <option value="Cloudy">Cloudy</option>
        <option value="Rainy">Rainy</option>
        <option value="Stormy">Stormy</option>
    </select><br><br>

    <button type="submit">Predict</button>
</form>

{% if prediction is not none %}
    <h3>Estimated Delivery Time: {{ prediction }} minutes</h3>
{% endif %}

{% if error %}
    <p style="color:red;">{{ error }}</p>
{% endif %}

</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def predict():

    prediction = None
    error = None

    if request.method == "POST":
        try:
            delivery_date = pd.to_datetime(request.form["delivery_date"])
            day_of_week = delivery_date.weekday()
            is_weekend = 1 if day_of_week in [5, 6] else 0

            input_data = {
                "distance_km": float(request.form["distance_km"]),
                "DayOfWeek": day_of_week,
                "IsWeekend": is_weekend,
                "traffic_level": request.form["traffic_level"],
                "weather_condition": request.form["weather_condition"]
            }

            df = pd.DataFrame([input_data])
            result = model.predict(df)

            if len(result) == 0:
                error = "Invalid input values."
            else:
                prediction = round(float(result[0]), 2)

        except Exception as e:
            error = str(e)

    return render_template_string(
        HTML,
        prediction=prediction,
        error=error
    )

if __name__ == "__main__":
    app.run(debug=True)

