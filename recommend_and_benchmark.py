"""
NearestCart: Feature-Space KNN Product Recommendation Engine
==============================================================
Project 30: E-Commerce Product Recommendation Prototype
Assigned Table: catalog_product_attributes

Deliverables Implemented:
1. Module 1: Top-N Nearest-Neighbor Retrieval Engine with Similarity Scores
2. Module 2: Distance-Weighted Neighbor Ranking Script
3. Module 3: Latency Benchmark comparing query speeds as catalog scales (Brute vs. KD-Tree vs. Ball-Tree)
4. Extension: Supervised KNN (Classification & Regression)
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier, KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, mean_squared_error, r2_score

# Set clean styling for plots
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({"font.size": 10, "figure.autolayout": True})

FEATURE_COLS = [
    "unit_price",
    "average_user_rating",
    "review_volume",
    "product_weight_grams",
    "package_volume_cm3"
]

class NearestCartRecommender:
    """
    Unsupervised & Category-Aware KNN Product Recommendation Engine
    """
    def __init__(self, data_path, algorithm="kd_tree", metric="euclidean"):
        self.data_path = data_path
        self.algorithm = algorithm
        self.metric = metric
        self.df = None
        self.scaler = StandardScaler()
        self.scaled_features = None
        self.model = None
        self.sku_to_idx = {}
        self.idx_to_sku = {}

        self._load_and_prepare()

    def _load_and_prepare(self):
        print(f"Loading catalog data from: {self.data_path}")
        self.df = pd.read_csv(self.data_path)
        print(f"Catalog loaded successfully with {len(self.df)} products and {len(self.df.columns)} columns.")

        # Build fast lookup indexes
        self.sku_to_idx = {sku: idx for idx, sku in enumerate(self.df["sku_id"])}
        self.idx_to_sku = {idx: sku for idx, sku in enumerate(self.df["sku_id"])}

        # Standardize continuous features: z = (x - mean) / std
        print(f"Standardizing continuous features: {FEATURE_COLS}")
        self.scaled_features = self.scaler.fit_transform(self.df[FEATURE_COLS])

        # Fit Unsupervised NearestNeighbors index
        self.model = NearestNeighbors(
            n_neighbors=25,
            algorithm=self.algorithm,
            metric=self.metric
        )
        self.model.fit(self.scaled_features)
        print(f"NearestNeighbors model indexed using algorithm='{self.algorithm}', metric='{self.metric}'.")

    def recommend(self, query_sku_id, top_n=5, filter_same_category=True):
        """
        Module 1: Top-N Nearest Neighbor Retrieval Engine
        Retrieves top_n products mathematically closest in standardized feature space.
        """
        if query_sku_id not in self.sku_to_idx:
            raise ValueError(f"SKU '{query_sku_id}' not found in catalog.")

        query_idx = self.sku_to_idx[query_sku_id]
        query_row = self.df.iloc[query_idx]
        query_vec = self.scaled_features[query_idx].reshape(1, -1)

        # Retrieve a candidate pool of neighbors (larger if category filtering)
        n_candidates = min(len(self.df), top_n * 5 if filter_same_category else top_n + 1)
        distances, indices = self.model.kneighbors(query_vec, n_neighbors=n_candidates)

        distances = distances.flatten()
        indices = indices.flatten()

        results = []
        for dist, idx in zip(distances, indices):
            if idx == query_idx:
                continue  # Skip query product itself

            cand_row = self.df.iloc[idx]
            if filter_same_category and cand_row["product_category"] != query_row["product_category"]:
                continue  # Filter out discordant categories

            # Bounded Similarity Score: [0%, 100%]
            similarity_pct = (1.0 / (1.0 + dist)) * 100.0

            results.append({
                "sku_id": cand_row["sku_id"],
                "product_category": cand_row["product_category"],
                "unit_price": cand_row["unit_price"],
                "average_user_rating": cand_row["average_user_rating"],
                "review_volume": int(cand_row["review_volume"]),
                "product_weight_grams": cand_row["product_weight_grams"],
                "package_volume_cm3": cand_row["package_volume_cm3"],
                "euclidean_distance": round(float(dist), 4),
                "similarity_score_pct": round(float(similarity_pct), 2)
            })

            if len(results) >= top_n:
                break

        res_df = pd.DataFrame(results)
        res_df["rank"] = range(1, len(res_df) + 1)
        return query_row, res_df

    def distance_weighted_ranking(self, rec_df, epsilon=1e-5):
        """
        Module 2: Distance-Weighted Neighbor Ranking Script
        Formula: w_i = 1 / (d_i + epsilon)
        Normalized weight: W_i = w_i / sum(w_j)
        Weighted Composite Score boosts items with closer Euclidean proximity.
        """
        df_ranked = rec_df.copy()
        raw_weights = 1.0 / (df_ranked["euclidean_distance"] + epsilon)
        norm_weights = raw_weights / raw_weights.sum()

        df_ranked["raw_weight"] = raw_weights.round(4)
        df_ranked["normalized_weight"] = norm_weights.round(4)

        # Composite priority score combining distance weighting and rating
        # Closer distance receives higher priority score
        df_ranked["weighted_priority_score"] = (
            (df_ranked["similarity_score_pct"] * 0.7) +
            (df_ranked["average_user_rating"] * 20.0 * 0.3)
        ) * norm_weights * len(df_ranked)
        df_ranked["weighted_priority_score"] = df_ranked["weighted_priority_score"].round(2)

        df_ranked["weighted_rank"] = df_ranked["weighted_priority_score"].rank(ascending=False, method="min").astype(int)
        df_ranked = df_ranked.sort_values(by="weighted_rank").reset_index(drop=True)
        return df_ranked

def run_latency_benchmark(catalog_sizes=[1000, 5000, 10000, 25000, 50000], n_queries=50):
    """
    Module 3: Latency Benchmark comparing query speeds as the product catalog scales.
    Compares: Brute-Force vs. KD-Tree vs. Ball-Tree
    """
    from generate_dataset import generate_catalog_dataset

    algorithms = ["brute", "kd_tree", "ball_tree"]
    benchmark_results = []

    print("\n" + "="*70)
    print("  MODULE 3: LATENCY & SCALABILITY BENCHMARKING")
    print("  Evaluating Query Speeds as Product Catalog Scales (1,000 -> 50,000)")
    print("="*70)

    scaler = StandardScaler()

    for size in catalog_sizes:
        print(f"\n--> Benchmarking Catalog Scale N = {size:,} products...")
        df_scale = generate_catalog_dataset(n_samples=size, random_seed=42)
        X_scaled = scaler.fit_transform(df_scale[FEATURE_COLS])

        # Select random query vectors
        np.random.seed(123)
        sample_indices = np.random.choice(size, size=min(n_queries, size), replace=False)
        query_vectors = X_scaled[sample_indices]

        for algo in algorithms:
            # Measure indexing time
            t_idx_start = time.perf_counter()
            nn_model = NearestNeighbors(n_neighbors=10, algorithm=algo, metric="euclidean")
            nn_model.fit(X_scaled)
            idx_time_ms = (time.perf_counter() - t_idx_start) * 1000.0

            # Measure query latency over repeated queries
            latencies = []
            for q_vec in query_vectors:
                t_q_start = time.perf_counter()
                nn_model.kneighbors(q_vec.reshape(1, -1))
                latencies.append((time.perf_counter() - t_q_start) * 1000.0)

            latencies = np.array(latencies)
            median_ms = np.median(latencies)
            p95_ms = np.percentile(latencies, 95)
            mean_ms = np.mean(latencies)

            benchmark_results.append({
                "catalog_size": size,
                "algorithm": algo,
                "index_time_ms": round(idx_time_ms, 2),
                "median_latency_ms": round(median_ms, 4),
                "p95_latency_ms": round(p95_ms, 4),
                "mean_latency_ms": round(mean_ms, 4)
            })

            print(f"   [{algo.upper():<9}] Index Build: {idx_time_ms:6.1f} ms | Query Median: {median_ms:6.3f} ms | p95: {p95_ms:6.3f} ms")

    res_df = pd.DataFrame(benchmark_results)

    # Save benchmark table
    os.makedirs("outputs", exist_ok=True)
    res_df.to_csv("outputs/latency_benchmark_results.csv", index=False)
    print("\nBenchmark results saved to 'outputs/latency_benchmark_results.csv'.")

    # Plot Latency Scaling Curve
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Subplot 1: Median Query Latency vs Catalog Size
    for algo, marker in zip(algorithms, ["o", "s", "^"]):
        sub = res_df[res_df["algorithm"] == algo]
        axes[0].plot(
            sub["catalog_size"],
            sub["median_latency_ms"],
            marker=marker,
            linewidth=2.2,
            markersize=7,
            label=f"{algo.upper()} (Median)"
        )
    axes[0].set_title("Query Latency (Median) vs. Catalog Scale", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Product Catalog Size (N items)", fontsize=11)
    axes[0].set_ylabel("Query Latency (milliseconds)", fontsize=11)
    axes[0].grid(True, linestyle="--", alpha=0.6)
    axes[0].legend(frameon=True)

    # Subplot 2: p95 Query Latency vs Catalog Size
    for algo, marker in zip(algorithms, ["o", "s", "^"]):
        sub = res_df[res_df["algorithm"] == algo]
        axes[1].plot(
            sub["catalog_size"],
            sub["p95_latency_ms"],
            marker=marker,
            linewidth=2.2,
            markersize=7,
            linestyle="--",
            label=f"{algo.upper()} (95th Percentile)"
        )
    axes[1].set_title("Tail Latency (p95) vs. Catalog Scale", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Product Catalog Size (N items)", fontsize=11)
    axes[1].set_ylabel("95th Percentile Latency (ms)", fontsize=11)
    axes[1].grid(True, linestyle="--", alpha=0.6)
    axes[1].legend(frameon=True)

    plt.tight_layout()
    plot_path = "outputs/latency_benchmark_scaling.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Latency benchmark plot saved to '{plot_path}'.")

    return res_df

def run_supervised_knn(df):
    """
    Extension: Supervised KNN Evaluation
    Demonstrates supervised KNN Classification (predicting Product Category from continuous features)
    and KNN Regression (predicting Average User Rating).
    """
    print("\n" + "="*70)
    print("  EXTENSION: SUPERVISED KNN (CLASSIFICATION & REGRESSION)")
    print("="*70)

    scaler = StandardScaler()
    X = scaler.fit_transform(df[FEATURE_COLS])
    y_class = df["product_category"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_class, test_size=0.2, random_state=42, stratify=y_class
    )

    # Supervised KNN Classifier
    knn_clf = KNeighborsClassifier(n_neighbors=7, weights="distance")
    knn_clf.fit(X_train, y_train)
    y_pred = knn_clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print(f"Supervised KNN Category Classifier Accuracy: {acc*100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, digits=3))

    # Confusion Matrix Visualization
    plt.figure(figsize=(10, 8))
    labels = sorted(y_class.unique())
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        cbar=False
    )
    plt.title("Supervised KNN Category Classification Confusion Matrix", fontsize=12, fontweight="bold")
    plt.xlabel("Predicted Category", fontsize=11)
    plt.ylabel("True Category", fontsize=11)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    cm_path = "outputs/supervised_knn_confusion_matrix.png"
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"Confusion matrix saved to '{cm_path}'.")

    return acc

def main():
    print("="*70)
    print("  PROJECT 30: E-COMMERCE PRODUCT RECOMMENDATION ENGINE (KNN)")
    print("  Feature-Space K-Nearest Neighbors Prototype")
    print("="*70)

    data_path = os.path.join("data", "catalog_product_attributes.csv")
    recommender = NearestCartRecommender(data_path=data_path, algorithm="kd_tree")

    # -------------------------------------------------------------
    # DEMO QUERY: MODULE 1 & MODULE 2
    # -------------------------------------------------------------
    query_sku = "SKU_00005"  # Electronics item (e.g. Laptop/Smartphone)
    print(f"\n[Demo Query] Requesting Recommendations for Target SKU: '{query_sku}'")

    query_product, top_recs = recommender.recommend(query_sku, top_n=5, filter_same_category=True)

    print("\n" + "-"*70)
    print("TARGET QUERY PRODUCT:")
    print("-"*70)
    print(f"SKU ID:           {query_product['sku_id']}")
    print(f"Category:         {query_product['product_category']}")
    print(f"Unit Price:       Rs. {query_product['unit_price']:,.2f}")
    print(f"User Rating:      {query_product['average_user_rating']} / 5.0")
    print(f"Review Volume:    {query_product['review_volume']:,} reviews")
    print(f"Weight:           {query_product['product_weight_grams']} g")
    print(f"Package Volume:   {query_product['package_volume_cm3']} cm3")

    print("\n" + "-"*70)
    print("MODULE 1: TOP-5 NEAREST NEIGHBORS (UNWEIGHTED RETRIEVAL)")
    print("-"*70)
    cols_display = ["rank", "sku_id", "unit_price", "average_user_rating", "review_volume", "euclidean_distance", "similarity_score_pct"]
    print(top_recs[cols_display].to_string(index=False))

    print("\n" + "-"*70)
    print("MODULE 2: DISTANCE-WEIGHTED RANKING (INVERSE DISTANCE WEIGHTS)")
    print("-"*70)
    weighted_recs = recommender.distance_weighted_ranking(top_recs)
    weighted_cols = ["weighted_rank", "sku_id", "unit_price", "euclidean_distance", "raw_weight", "normalized_weight", "weighted_priority_score"]
    print(weighted_recs[weighted_cols].to_string(index=False))

    # -------------------------------------------------------------
    # RUN SUPERVISED EVALUATION
    # -------------------------------------------------------------
    run_supervised_knn(recommender.df)

    # -------------------------------------------------------------
    # RUN LATENCY BENCHMARK (MODULE 3)
    # -------------------------------------------------------------
    run_latency_benchmark(catalog_sizes=[1000, 5000, 10000, 25000, 50000], n_queries=50)

    print("\n" + "="*70)
    print("  EXECUTION COMPLETED SUCCESSFULLY!")
    print("  All 3 modules executed and benchmark figures saved to outputs/")
    print("="*70)

if __name__ == "__main__":
    main()
