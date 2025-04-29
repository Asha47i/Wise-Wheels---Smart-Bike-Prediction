import numpy as np
import pickle
import streamlit as st
from sklearn.preprocessing import LabelEncoder
from datetime import datetime

# Load the trained model
with open('bike_demand_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Example encoders for categorical variables
le_seasons = LabelEncoder()
le_seasons.fit(["Spring", "Summer", "Autumn", "Winter"])  # Seasons as names

le_holiday = LabelEncoder()
le_holiday.fit(["No", "Yes"])  # Holiday as No/Yes

le_functioning_day = LabelEncoder()
le_functioning_day.fit(["No", "Yes"])  # Functioning Day as No/Yes

le_weekday = LabelEncoder()
le_weekday.fit(["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])  # Weekday names

le_isweekend = LabelEncoder()
le_isweekend.fit(["No", "Yes"])  # Weekend as No/Yes

le_month = LabelEncoder()
le_month.fit(["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])  # Month abbreviations

# Function to determine the season based on the month
def get_season(month):
    if month in [3, 4, 5]:
        return "Spring"
    elif month in [6, 7, 8]:
        return "Summer"
    elif month in [9, 10, 11]:
        return "Autumn"
    else:
        return "Winter"

# Predict function
def predict_bike_availability(input_data):
    input_data['Seasons'] = le_seasons.transform([input_data['Seasons']])[0]
    input_data['Holiday'] = le_holiday.transform([input_data['Holiday']])[0]
    input_data['Functioning Day'] = le_functioning_day.transform([input_data['Functioning Day']])[0]
    input_data['Weekday'] = le_weekday.transform([input_data['Weekday']])[0]
    input_data['IsWeekend'] = le_isweekend.transform([input_data['IsWeekend']])[0]
    input_data['Month'] = le_month.transform([input_data['Month']])[0]

    features = [
        input_data['Hour'],
        input_data['Temperature(°C)'],
        input_data['Humidity(%)'],
        input_data['Wind speed (m/s)'],
        input_data['Visibility (10m)'],
        input_data['Dew point temperature(°C)'],
        input_data['Solar Radiation (MJ/m2)'],
        input_data['Rainfall(mm)'],
        input_data['Snowfall (cm)'],
        input_data['Seasons'],
        input_data['Holiday'],
        input_data['Functioning Day'],
        input_data['Day'],
        input_data['Month'],
        input_data['Year'],
        input_data['Weekday'],
        input_data['IsWeekend'],
    ]

    features = np.array(features).reshape(1, -1)
    prediction = model.predict(features)
    return prediction

# --- Streamlit App ---
st.title("🚲 Wise Wheels - Predict Your Bike Demand! 🔥")

st.markdown("""
Wise Wheels helps bike companies predict demand smartly!  
Knowing how many bikes will be needed helps **reduce costs**, **increase rentals**, and **keep customers happy** 🚴‍♂️📈✨  
""")

# --- Date and Time Selection (Choose Hour Only) ---
st.header("📅 Choose Date and Time")

date = st.date_input("Select the date (Year, Month, Day)")

# Extract parts
year = date.year
month_number = date.month
day = date.day

# Month Name
month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", 
               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
month = month_names[month_number - 1]

# Determine the season based on the month
season = get_season(month_number)

# Weekday Name
weekday_name = date.strftime('%A')

# Is it weekend?
isweekend = "Yes" if date.weekday() >= 5 else "No"

st.info(f"📆 You selected: {date} — {weekday_name} (Weekend: {isweekend}) - {season}")

# --- Select Hour Only ---
hour = st.slider("Select the hour of the day", min_value=0, max_value=23, step=1)

# --- Additional Inputs with Constraints Based on Temperature ---
st.header("🌡️ Weather Conditions")

temperature = st.slider("Temperature (°C)", min_value=-10, max_value=40, step=1)

# Constraints based on Temperature
dew_point_min = max(-10, temperature - 15)
dew_point_max = min(temperature, 40)

# Ensure that min_value < max_value
if dew_point_min == dew_point_max:
    dew_point_max += 1  # Adjust max_value to ensure it's greater than min_value

dew_point = st.slider(f"Dew Point (°C) (must be ≤ temperature)", min_value=dew_point_min, max_value=dew_point_max, value=dew_point_min)

humidity = st.slider("Humidity (%)", min_value=0, max_value=100, step=1)

solar_radiation = st.slider("Solar Radiation (MJ/m2)", min_value=0.0, max_value=1000.0, step=0.1)

visibility_min = max(0, int((temperature + 10) * 0.5))  # simple model
visibility = st.slider(f"Visibility (10m units) (higher if hot)", min_value=visibility_min, max_value=20, step=1)

wind_speed = st.slider("Wind speed (m/s)", min_value=0.0, max_value=10.0, step=0.1)
rainfall = st.slider("Rainfall (mm)", min_value=0.0, max_value=200.0, step=0.1)
snowfall = st.slider("Snowfall (cm)", min_value=0.0, max_value=200.0, step=0.1)

# --- Holiday and Functioning Day ---
st.header("🏖️ Special Days")

holiday = st.selectbox("Is it a holiday?", ["No", "Yes"])
functioning_day = st.selectbox("Is it a functioning day?", ["Yes", "No"])

# --- Pack input for prediction ---
input_data = {
    'Hour': hour,
    'Temperature(°C)': temperature,
    'Humidity(%)': humidity,
    'Wind speed (m/s)': wind_speed,
    'Visibility (10m)': visibility,
    'Dew point temperature(°C)': dew_point,
    'Solar Radiation (MJ/m2)': solar_radiation,
    'Rainfall(mm)': rainfall,
    'Snowfall (cm)': snowfall,
    'Seasons': season,
    'Holiday': holiday,
    'Functioning Day': functioning_day,
    'Day': day,
    'Month': month,
    'Year': year,
    'Weekday': weekday_name,
    'IsWeekend': isweekend,
}

# --- Predict button ---
if st.button("🚴‍♀️ Predict Bike Rentals"):
    prediction = predict_bike_availability(input_data)
    st.success(f"🔮 Predicted Bike Rentals: {prediction[0]:.0f}")
