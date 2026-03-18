import pandas as pd
from sklearn.cluster import KMeans
import joblib

# Load dataset
df = pd.read_csv("Mall_Customers.csv")

# Select important features
X = df[['Annual Income (k$)', 'Spending Score (1-100)']]

# Train KMeans model
model = KMeans(n_clusters=5, random_state=42)
model.fit(X)

# Save model
joblib.dump(model, "kmeans_model.pkl")

print("Model trained successfully!")