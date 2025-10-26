import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# --- Load and prepare housing data ---
def load_and_prepare_housing(filepath='housing.csv'):
    housing = pd.read_csv(filepath)
    housing_num = housing[['longitude', 'latitude', 'median_income']]

    imputer = SimpleImputer(strategy='median')
    scaler = StandardScaler()

    housing_imputed = imputer.fit_transform(housing_num)
    housing_scaled = scaler.fit_transform(housing_imputed)

    return housing, housing_scaled, scaler

# --- K-Means clustering analysis ---
def kmeans_best_cluster(data_scaled, k_range=range(2, 11)):
    silhouette_scores = []
    kmeans_models = {}

    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(data_scaled)
        score = silhouette_score(data_scaled, labels)
        silhouette_scores.append(score)
        kmeans_models[k] = kmeans
        print(f"K={k}, Silhouette Score={score:.4f}")

    plt.figure(figsize=(8,5))
    plt.plot(list(k_range), silhouette_scores, marker='o', linestyle='--', color='b')
    plt.xlabel("Number of Clusters K")
    plt.ylabel("Silhouette Score")
    plt.title("Silhouette Scores for Different K values")
    plt.grid(True)
    plt.show()

    best_k = k_range[np.argmax(silhouette_scores)]
    print(f"Optimal K based on Silhouette Score: {best_k}")

    return best_k, kmeans_models, silhouette_scores

# --- Plot best K-Means clusters ---
def plot_kmeans_clusters_best(data_scaled, kmeans_model, feature_names=['longitude','latitude']):
    labels = kmeans_model.labels_
    score = silhouette_score(data_scaled, labels)
    
    plt.figure(figsize=(8,6))
    plt.scatter(data_scaled[:,0], data_scaled[:,1], c=labels, cmap='viridis', s=30)
    plt.xlabel(feature_names[0])
    plt.ylabel(feature_names[1])
    plt.title(f"K-Means Clusters (K={kmeans_model.n_clusters}, Silhouette={score:.3f})")
    plt.colorbar(label='Cluster')
    plt.grid(True)
    plt.show()

# --- DBSCAN clustering ---
def dbscan_clustering(data_original, data_scaled, eps, min_samples):
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(data_original)

    unique_labels = set(labels)
    n_clusters = len(unique_labels - {-1})
    score = None
    if n_clusters > 1:
        score = silhouette_score(data_scaled[labels != -1], labels[labels != -1])
    num_noise = np.sum(labels == -1)
    print(f"DBSCAN run with eps={eps}, min_samples={min_samples}")
    print(f"Clusters: {n_clusters} | Noise points: {num_noise} | Silhouette Score: {score}")

    return labels, score


# --- Plot best DBSCAN clusters ---
def plot_dbscan_clusters_best(data_scaled, labels, score, eps, min_samples, feature_names=['longitude','latitude']):
    plt.figure(figsize=(8,6))
    plt.scatter(data_scaled[:,0], data_scaled[:,1], c=labels, cmap='viridis', s=30)
    plt.xlabel(feature_names[0])
    plt.ylabel(feature_names[1])
    plt.title(f"DBSCAN Clusters (eps={eps}, min_samples={min_samples}, Silhouette={score:.3f})")
    plt.colorbar(label='Cluster')
    plt.grid(True)
    plt.show()

# --- Sweep DBSCAN parameters ---
def dbscan_parameter_sweep(data_original, data_scaled, eps_values, min_samples_values):
    results = []
    for eps in eps_values:
        for min_samples in min_samples_values:
            labels, score = dbscan_clustering(data_original, data_scaled, eps, min_samples)
            n_clusters = len(set(labels) - {-1})
            n_noise = np.sum(labels == -1)
            results.append({
                'eps': eps,
                'min_samples': min_samples,
                'n_clusters': n_clusters,
                'n_noise': n_noise,
                'silhouette': score,
                'labels': labels
            })
            print(f"eps={eps}, min_samples={min_samples} | Clusters={n_clusters}, Noise={n_noise}, Silhouette={score}")
    return results

# --- DBSCAN silhouette heatmap ---
def plot_dbscan_silhouette_heatmap(dbscan_results):
    df_results = pd.DataFrame(dbscan_results)
    df_results = df_results[df_results['silhouette'].notnull()]
    heatmap_data = df_results.pivot(index='min_samples', columns='eps', values='silhouette')

    plt.figure(figsize=(8,6))
    sns.heatmap(heatmap_data, annot=True, fmt=".4f", cmap='viridis')
    plt.title("DBSCAN Silhouette Scores (min_samples vs eps)")
    plt.xlabel("eps")
    plt.ylabel("min_samples")
    plt.show()

# --- Analyze K-Means clusters for all K ---
def analyze_clusters_for_all_k(data, kmeans_models, k_range=range(2, 11),
                               feature_names=['longitude', 'latitude', 'median_income']):
    df_clusters = data[feature_names].copy()
    for k in k_range:
        model = kmeans_models[k]
        labels = model.labels_
        df_clusters[f'cluster_{k}'] = labels
        print(f"\n=== Cluster Summary for K = {k} ===")
        print("Cluster Sizes:")
        print(df_clusters[f'cluster_{k}'].value_counts())
        print(f"\n=== Cluster-wise Median Income for K = {k} ===")
        cluster_stats = df_clusters.groupby(f'cluster_{k}')['median_income'].describe().round(3)
        print(cluster_stats)
    return df_clusters

def plot_clusters_for_all_k_combined(data, kmeans_models, k_range=range(2, 11), feature_names=['longitude', 'latitude']):
    n_k = len(k_range)
    n_cols = min(3, n_k)
    n_rows = (n_k + n_cols - 1) // n_cols
    plt.figure(figsize=(5 * n_cols, 5 * n_rows))
    for i, k in enumerate(k_range):
        model = kmeans_models[k]
        labels = model.labels_
        plt.subplot(n_rows, n_cols, i + 1)
        plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis', s=30)
        plt.xlabel(feature_names[0])
        plt.ylabel(feature_names[1])
        plt.title(f'K = {k}')
    plt.tight_layout()
    plt.show()

from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import pandas as pd

def run_dbscan(housing, X_scaled, eps=0.3, min_samples=10):

    # Use only longitude and latitude for DBSCAN
    X_geo_scaled = X_scaled[:, :2]

    # Initialize and fit DBSCAN
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    db_labels = dbscan.fit_predict(X_geo_scaled)

    # Add cluster labels to DataFrame
    housing['dbscan_cluster'] = db_labels

    # Print cluster counts
    print("DBSCAN Cluster Counts:")
    print(pd.Series(db_labels).value_counts())

    # Scatter plot of DBSCAN clusters
    plt.figure(figsize=(8,6))
    plt.scatter(X_geo_scaled[:,1], X_geo_scaled[:,0], c=db_labels, cmap='tab20', s=10)
    plt.xlabel("Latitude (scaled)")
    plt.ylabel("Longitude (scaled)")
    plt.title("DBSCAN Clusters")
    plt.show()

    return housing, db_labels


# --- Main function ---
def main():
    housing, housing_scaled, scaler = load_and_prepare_housing(
        r"C:\Users\Poornima Sarwadi\Documents\Assignment\Assignment-6\housing.csv"
    )

    K_RANGE = range(2, 11)

    # --- K-Means ---
    best_k, kmeans_models, sil_scores = kmeans_best_cluster(housing_scaled, K_RANGE)
    best_k_model = kmeans_models[best_k]
    print(f"\nBest K-Means: K={best_k}, Silhouette={max(sil_scores):.3f}")
    plot_kmeans_clusters_best(housing_scaled, best_k_model)

    # Analyze and visualize clusters
    housing_clusters_df = analyze_clusters_for_all_k(housing, kmeans_models, k_range=K_RANGE)
    plot_clusters_for_all_k_combined(housing_scaled, kmeans_models, k_range=K_RANGE)

    # --- DBSCAN Sweep ---
    EPS_VALUES = [0.3, 0.4, 0.5, 0.6]
    MIN_SAMPLES_VALUES = [10, 12, 14]

    dbscan_results = dbscan_parameter_sweep(
        housing[['longitude','latitude','median_income']],
        housing_scaled,
        EPS_VALUES,
        MIN_SAMPLES_VALUES
    )

    # Plot DBSCAN silhouette heatmap
    plot_dbscan_silhouette_heatmap(dbscan_results)

    # Find best DBSCAN
    best_dbscan = max([r for r in dbscan_results if r['silhouette'] is not None], key=lambda x: x['silhouette'])
    print(f"\nBest DBSCAN → eps={best_dbscan['eps']}, min_samples={best_dbscan['min_samples']}, Silhouette={best_dbscan['silhouette']:.3f}")

    # DBSCAN cluster-wise summary for median_income
    housing_dbscan = housing.copy()
    housing_dbscan['cluster'] = best_dbscan['labels']
    print("\n=== DBSCAN Cluster Sizes ===")
    print(housing_dbscan['cluster'].value_counts())
    print("\n=== DBSCAN Cluster-wise Median Income ===")
    cluster_income_stats = housing_dbscan.groupby('cluster')['median_income'].describe().round(3)
    print(cluster_income_stats)


    housing, db_labels = run_dbscan(housing, housing_scaled, eps=0.6, min_samples=14)

    # Plot best DBSCAN
    plot_dbscan_clusters_best(
        housing_scaled,
        best_dbscan['labels'],
        best_dbscan['silhouette'],
        best_dbscan['eps'],
        best_dbscan['min_samples']
    )

    # Side-by-side comparison
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.scatter(housing_scaled[:,0], housing_scaled[:,1], c=best_k_model.labels_, cmap='viridis', s=30)
    plt.title(f"Best K-Means K={best_k}")
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')

    plt.subplot(1,2,2)
    plt.scatter(housing_scaled[:,0], housing_scaled[:,1], c=best_dbscan['labels'], cmap='viridis', s=30)
    plt.title(f"Best DBSCAN eps={best_dbscan['eps']}, min_samples={best_dbscan['min_samples']}")
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.tight_layout()
    plt.show()




if __name__ == "__main__":
    main()
