import joblib
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from config import FEATURES, MODEL_DIR, OUTPUT_DIR, TRAIN_FILE

df = pd.read_csv(TRAIN_FILE)
X = df[FEATURES]
 
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_scaled)

pca = PCA(n_components=2, random_state=42)
components = pca.fit_transform(X_scaled)

result = pd.DataFrame({
    "cluster": clusters,
    "pca_1": components[:, 0],
    "pca_2": components[:, 1],
})
result.to_csv(OUTPUT_DIR / "behavior_clusters.csv", index=False)

joblib.dump(scaler, MODEL_DIR / "clustering_scaler.joblib")
joblib.dump(kmeans, MODEL_DIR / "kmeans.joblib")
joblib.dump(pca, MODEL_DIR / "pca.joblib")

print("Behavior clustering completed.")
