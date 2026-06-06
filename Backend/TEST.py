# ================================
# 1. IMPORT LIBRARIES
# ================================
import numpy as np
import joblib

# ================================
# 2. LOAD SAVED MODEL & SCALER
# ================================
model = joblib.load("svm_model.pkl")   # or svm_model.pkl / logistic_regression.pkl
scaler = joblib.load("scaler.pkl")

# ================================
# 3. TAKE USER INPUT
# ================================
print("Enter the following details:\n")

temperature = float(input("Temperature: "))
rainfall = float(input("Rainfall: "))
ph = float(input("pH: "))
turbidity = float(input("Turbidity: "))
dissolved_oxygen = float(input("Dissolved Oxygen: "))
nitrate = float(input("Nitrate: "))
lead = float(input("Lead: "))
bacterial_count = float(input("Bacterial Count: "))
clean_water_percentage = float(input("Clean Water %: "))
sanitation_level = float(input("Sanitation Level: "))
healthcare_access = float(input("Healthcare Access: "))
symptom_diarrhea = int(input("Diarrhea (0/1): "))
symptom_fever = int(input("Fever (0/1): "))
water_dirty = int(input("Dirty Water (0/1): "))
water_scarcity = int(input("Water Scarcity (0/1): "))

# ================================
# 4. PREPARE INPUT ARRAY
# ================================
input_data = np.array([[ 
    temperature, rainfall, ph, turbidity, dissolved_oxygen,
    nitrate, lead, bacterial_count, clean_water_percentage,
    sanitation_level, healthcare_access,
    symptom_diarrhea, symptom_fever,
    water_dirty, water_scarcity
]])

# ================================
# 5. SCALE INPUT
# ================================
input_scaled = scaler.transform(input_data)

# ================================
# 6. PREDICT
# ================================
prediction = model.predict(input_scaled)[0]

# ================================
# 7. OUTPUT RESULT
# ================================
if prediction == 0:
    print("\nPredicted Risk: LOW")
elif prediction == 1:
    print("\nPredicted Risk: MEDIUM")
else:
    print("\nPredicted Risk: HIGH")