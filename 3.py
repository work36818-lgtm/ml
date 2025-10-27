import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# Step 1: Get input from the user
num_patients = int(input("Enter number of patients: "))

age = []
bp = []
chol = []

for i in range(num_patients):
    print(f"\nEnter details for patient {i+1}:")
    age.append(int(input("  Age: ")))
    bp.append(int(input("  Blood Pressure: ")))
    chol.append(int(input("  Cholesterol: ")))

data = {
    'Age': age,
    'BloodPressure': bp,
    'Cholesterol': chol
}

df = pd.DataFrame(data)
print("\nPatient Data:\n", df)

# Step 2: Normalize the Data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

# Step 3: Apply K-Means Clustering
kmeans = KMeans(n_clusters=2, random_state=42)
kmeans_labels = kmeans.fit_predict(X_scaled)

# Step 4: Apply EM (Gaussian Mixture Model)
gmm = GaussianMixture(n_components=2, random_state=42)
gmm_labels = gmm.fit_predict(X_scaled)

# Step 5: Evaluate with Silhouette Score (only if more than 1 cluster and >2 samples)
if len(set(kmeans_labels)) > 1 and len(df) > 2:
    kmeans_score = silhouette_score(X_scaled, kmeans_labels)
    gmm_score = silhouette_score(X_scaled, gmm_labels)
else:
    kmeans_score = gmm_score = None

# Step 6: Print Clustering Results
df['KMeans Cluster'] = kmeans_labels
df['GMM Cluster'] = gmm_labels

print("\nClustering Results:")
print(df)

if kmeans_score:
    print("\nSilhouette Score - KMeans:", round(kmeans_score, 2))
    print("Silhouette Score - GMM (EM):", round(gmm_score, 2))
else:
    print("\nSilhouette Scores not available (need at least 2 clusters and 3+ samples)")

# Step 7: Visualization
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=kmeans_labels, cmap='viridis')
plt.title('K-Means Clustering')
plt.xlabel('Age (scaled)')
plt.ylabel('Blood Pressure (scaled)')

plt.subplot(1, 2, 2)
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=gmm_labels, cmap='plasma')
plt.title('EM Clustering (GMM)')
plt.xlabel('Age (scaled)')
plt.ylabel('Blood Pressure (scaled)')

plt.tight_layout()
plt.show()
