import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import tkinter as tk
from tkinter import messagebox

# --- Step 1: Load CSV and scale features ---
df = pd.read_csv("C:/Users/Logo/Desktop/Ai project/Retail-Buyer-Segmentation/Classification/retail_customers_with_2_clusters.csv")

numeric_features = [
    "annual_income",
    "spend_wine", "spend_fruits", "spend_meat", "spend_fish",
    "spend_sweets", "spend_gold",
    "num_web_purchases", "num_catalog_purchases", "num_store_purchases",
    "num_discount_purchases"
]

X = df[numeric_features]
y = df["cluster_kmeans"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_scaled, y)

def predict_cluster():
    try:
        # Read inputs from GUI
        input_values = [float(entries[feature].get()) for feature in numeric_features]
        input_scaled = scaler.transform([input_values])  # scale input like training data
        prediction = lr_model.predict(input_scaled)
        messagebox.showinfo("Prediction", f"Predicted Cluster: {prediction[0]}")
    except ValueError:
        messagebox.showerror("Error", "Please enter valid numeric values!")

root = tk.Tk()
root.title("Retail Customer Cluster Prediction")

entries = {}

for i, feature in enumerate(numeric_features):
    tk.Label(root, text=feature).grid(row=i, column=0, padx=5, pady=5, sticky="w")
    entry = tk.Entry(root)
    entry.grid(row=i, column=1, padx=5, pady=5)
    entries[feature] = entry

predict_button = tk.Button(root, text="Predict Cluster", command=predict_cluster)
predict_button.grid(row=len(numeric_features), column=0, columnspan=2, pady=10)

root.mainloop()
