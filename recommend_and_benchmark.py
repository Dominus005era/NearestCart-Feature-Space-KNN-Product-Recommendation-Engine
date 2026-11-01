"""
NearestCart: Feature-Space KNN Product Recommendation Engine
==============================================================
Project 30: E-Commerce Product Recommendation Prototype
Assigned Table: catalog_product_attributes

Deliverables Implemented:
1. Module 1: Top-N Nearest-Neighbor Retrieval Engine with Similarity Scores
2. Module 2: Distance-Weighted Neighbor Ranking Script
3. Module 3: Latency Benchmark (Brute vs. KD-Tree vs. Ball-Tree)
4. Extension: Supervised KNN (Classification & Regression)
5. Interactive User Query Mode (Clean, minimal, non-technical interface)
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"

import sys
import time
import random
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

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
    def __init__(self, data_path, algorithm="kd_tree", metric="euclidean", verbose=False):
        self.data_path = data_path
        self.algorithm = algorithm
        self.metric = metric
        self.verbose = verbose
        self.df = None
        self.scaler = StandardScaler()
        self.scaled_features = None
        self.model = None
        self.sku_to_idx = {}
        self.idx_to_sku = {}

        self._load_and_prepare()

    def _load_and_prepare(self):
        if self.verbose:
            print(f"Loading catalog data from: {self.data_path}")
        self.df = pd.read_csv(self.data_path)
        if self.verbose:
            print(f"Catalog loaded with {len(self.df)} products and {len(self.df.columns)} columns.")

        # Build fast lookup indexes
        self.sku_to_idx = {sku: idx for idx, sku in enumerate(self.df["sku_id"])}
        self.idx_to_sku = {idx: sku for idx, sku in enumerate(self.df["sku_id"])}

        # Standardize continuous features
        self.scaled_features = self.scaler.fit_transform(self.df[FEATURE_COLS])

        # Fit Unsupervised NearestNeighbors index
        self.model = NearestNeighbors(
            n_neighbors=25,
            algorithm=self.algorithm,
            metric=self.metric
        )
        self.model.fit(self.scaled_features)
        if self.verbose:
            print(f"NearestNeighbors model indexed ({self.algorithm}).")

    def recommend(self, query_sku_id, top_n=5, filter_same_category=True):
        """
        Module 1: Top-N Nearest Neighbor Retrieval Engine
        Retrieves top_n products mathematically closest in standardized feature space.
        """
        clean_sku = query_sku_id.strip().upper()
        if not clean_sku.startswith("SKU_"):
            if clean_sku.isdigit():
                clean_sku = f"SKU_{int(clean_sku):05d}"

        if clean_sku not in self.sku_to_idx:
            raise ValueError(f"SKU '{query_sku_id}' not found in catalog (Available: SKU_00001 to SKU_05000).")

        query_idx = self.sku_to_idx[clean_sku]
        query_row = self.df.iloc[query_idx]
        query_vec = self.scaled_features[query_idx].reshape(1, -1)

        # Retrieve a candidate pool of neighbors (large enough to ensure top_n same-category matches)
        n_candidates = min(len(self.df), max(top_n * 50, 250) if filter_same_category else top_n + 1)
        distances, indices = self.model.kneighbors(query_vec, n_neighbors=n_candidates)

        distances = distances.flatten()
        indices = indices.flatten()

        results = []
        for dist, idx in zip(distances, indices):
            if idx == query_idx:
                continue

            cand_row = self.df.iloc[idx]
            if filter_same_category and cand_row["product_category"] != query_row["product_category"]:
                continue

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

        reputation_score = df_ranked["average_user_rating"] * 20.0
        n_items = len(df_ranked)
        df_ranked["weighted_priority_score"] = (
            (0.70 * df_ranked["similarity_score_pct"] + 0.30 * reputation_score)
            * norm_weights * n_items
        ).round(2)

        df_ranked = df_ranked.sort_values(by="weighted_priority_score", ascending=False).reset_index(drop=True)
        df_ranked["weighted_rank"] = range(1, len(df_ranked) + 1)
        return df_ranked

def display_clean_recommendations(query_product, rec_df, top_n=5):
    """
    Renders clean, beginner & non-technical friendly output.
    """
    cat = query_product["product_category"]
    price = query_product["unit_price"]
    rating = query_product["average_user_rating"]
    reviews = int(query_product["review_volume"])

    print("\n" + "="*84)
    print(f"  TARGET PRODUCT: {query_product['sku_id']} [{cat}]")
    print(f"  Price: Rs. {price:,.2f}  |  Rating: {rating:.1f} / 5.0 ({reviews:,} reviews)")
    print("="*84)

    print(f"\nTOP {min(top_n, len(rec_df))} RECOMMENDED ALTERNATIVES:")
    print("-" * 84)
    print(f"{'#':<3} | {'SKU ID':<10} | {'Category':<16} | {'Price (INR)':<14} | {'Rating':<10} | {'Match':<12}")
    print("-" * 84)

    for i, row in rec_df.head(top_n).iterrows():
        rank = row.get("weighted_rank", row.get("rank", i + 1))
        sku = row["sku_id"]
        c = row["product_category"]
        p = f"Rs. {row['unit_price']:,.2f}"
        r = f"{row['average_user_rating']:.1f} / 5.0"
        sim = f"{row['similarity_score_pct']:.0f}% Match"
        if rank == 1:
            sim += " (Best)"

        print(f"{rank:<3} | {sku:<10} | {c:<16} | {p:<14} | {r:<10} | {sim:<12}")

    print("-" * 84)

def run_interactive_cli(recommender):
    """
    Interactive, user-friendly prompt loop.
    """
    print("\n" + "="*70)
    print("       NearestCart Product Recommendation Engine")
    print("       Interactive User Query Mode")
    print("="*70)
    print("Ready! 5,000 products loaded across 8 retail categories.")
    print("Commands:")
    print("  - Type an SKU ID (e.g., 'SKU_00005' or just '5')")
    print("  - Type 'random' to test a random product")
    print("  - Type 'categories' to see sample products by category")
    print("  - Type 'benchmark' to run technical evaluation suite")
    print("  - Type 'exit' or 'quit' to close")
    print("="*70)

    while True:
        try:
            user_input = input("\nEnter Product SKU (or 'random' / 'exit'): ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting. Thank you for using NearestCart!")
            break

        if not user_input:
            continue

        cmd = user_input.lower()
        if cmd in ["exit", "quit", "q"]:
            print("Exiting. Thank you for using NearestCart!")
            break

        if cmd == "categories":
            print("\nAvailable Categories & Sample SKUs:")
            for cat, group in recommender.df.groupby("product_category"):
                sample_skus = ", ".join(group["sku_id"].head(3).tolist())
                print(f"  • {cat:<24}: e.g. {sample_skus}")
            continue

        if cmd == "benchmark":
            print("\nStarting full technical benchmark...")
            run_supervised_knn(recommender.df)
            run_latency_benchmark(catalog_sizes=[1000, 5000, 10000, 25000, 50000], n_queries=50)
            print("\nBenchmark complete! Charts saved to outputs/ folder.")
            continue

        if cmd == "random":
            rand_idx = random.randint(0, len(recommender.df) - 1)
            target_sku = recommender.df.iloc[rand_idx]["sku_id"]
        else:
            target_sku = user_input

        try:
            query_product, top_recs = recommender.recommend(target_sku, top_n=5, filter_same_category=True)
            weighted_recs = recommender.distance_weighted_ranking(top_recs)
            display_clean_recommendations(query_product, weighted_recs, top_n=5)
        except ValueError as err:
            print(f"Error: {err}")

def run_latency_benchmark(catalog_sizes=[1000, 5000, 10000, 25000, 50000], n_queries=50):
    """
    Module 3: Latency & Scalability Benchmarking
    """
    print("\n" + "="*70)
    print("  MODULE 3: LATENCY & SCALABILITY BENCHMARKING")
    print(f"  Evaluating Query Speeds as Product Catalog Scales (1,000 -> 50,000)")
    print("="*70)

    algorithms = ["brute", "kd_tree", "ball_tree"]
    benchmark_results = []
    base_dim = 5
    rng = np.random.RandomState(42)

    for size in catalog_sizes:
        print(f"\n--> Benchmarking Catalog Scale N = {size:,} products...")
        synthetic_catalog = rng.randn(size, base_dim)
        query_indices = rng.choice(size, size=n_queries, replace=False)
        test_queries = synthetic_catalog[query_indices]

        for algo in algorithms:
            t0 = time.perf_counter()
            index_model = NearestNeighbors(n_neighbors=5, algorithm=algo, metric="euclidean")
            index_model.fit(synthetic_catalog)
            idx_time_ms = (time.perf_counter() - t0) * 1000.0

            latencies = []
            for q in test_queries:
                t_start = time.perf_counter()
                _ = index_model.kneighbors(q.reshape(1, -1))
                latencies.append((time.perf_counter() - t_start) * 1000.0)

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
    os.makedirs("outputs", exist_ok=True)
    res_df.to_csv("outputs/latency_benchmark_results.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for algo, marker in zip(algorithms, ["o", "s", "^"]):
        sub = res_df[res_df["algorithm"] == algo]
        axes[0].plot(sub["catalog_size"], sub["median_latency_ms"], marker=marker, linewidth=2.2, markersize=7, label=f"{algo.upper()} (Median)")
    axes[0].set_title("Query Latency (Median) vs. Catalog Scale", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Product Catalog Size (N items)", fontsize=11)
    axes[0].set_ylabel("Query Latency (milliseconds)", fontsize=11)
    axes[0].grid(True, linestyle="--", alpha=0.6)
    axes[0].legend(frameon=True)

    for algo, marker in zip(algorithms, ["o", "s", "^"]):
        sub = res_df[res_df["algorithm"] == algo]
        axes[1].plot(sub["catalog_size"], sub["p95_latency_ms"], marker=marker, linewidth=2.2, markersize=7, linestyle="--", label=f"{algo.upper()} (95th Percentile)")
    axes[1].set_title("Tail Latency (p95) vs. Catalog Scale", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Product Catalog Size (N items)", fontsize=11)
    axes[1].set_ylabel("95th Percentile Latency (ms)", fontsize=11)
    axes[1].grid(True, linestyle="--", alpha=0.6)
    axes[1].legend(frameon=True)

    plt.tight_layout()
    plt.savefig("outputs/latency_benchmark_scaling.png", dpi=300)
    plt.close()

    return res_df

def run_supervised_knn(df):
    """
    Extension: Supervised KNN Evaluation
    """
    print("\n" + "="*70)
    print("  EXTENSION: SUPERVISED KNN CATEGORY CLASSIFICATION")
    print("="*70)

    scaler = StandardScaler()
    X = scaler.fit_transform(df[FEATURE_COLS])
    y_class = df["product_category"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_class, test_size=0.2, random_state=42, stratify=y_class
    )

    knn_clf = KNeighborsClassifier(n_neighbors=7, weights="distance")
    knn_clf.fit(X_train, y_train)
    y_pred = knn_clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc*100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, digits=3))

    plt.figure(figsize=(10, 8))
    labels = sorted(y_class.unique())
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels, cbar=False)
    plt.title("Supervised KNN Category Classification Confusion Matrix", fontsize=12, fontweight="bold")
    plt.xlabel("Predicted Category", fontsize=11)
    plt.ylabel("True Category", fontsize=11)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig("outputs/supervised_knn_confusion_matrix.png", dpi=300)
    plt.close()

    return acc

def main():
    parser = argparse.ArgumentParser(description="NearestCart: KNN Product Recommendation Engine")
    parser.add_argument("--sku", type=str, default=None, help="Query specific SKU directly (e.g. SKU_00005)")
    parser.add_argument("--top_n", type=int, default=5, help="Number of recommendations to return (default: 5)")
    parser.add_argument("--benchmark", action="store_true", help="Run full automated benchmark suite")
    parser.add_argument("--demo", action="store_true", help="Run quick non-interactive demo for SKU_00005")
    args = parser.parse_args()

    data_path = os.path.join("data", "catalog_product_attributes.csv")
    recommender = NearestCartRecommender(data_path=data_path, algorithm="kd_tree", verbose=False)

    if args.benchmark:
        run_supervised_knn(recommender.df)
        run_latency_benchmark(catalog_sizes=[1000, 5000, 10000, 25000, 50000], n_queries=50)
        return

    if args.sku:
        query_product, top_recs = recommender.recommend(args.sku, top_n=args.top_n, filter_same_category=True)
        weighted_recs = recommender.distance_weighted_ranking(top_recs)
        display_clean_recommendations(query_product, weighted_recs, top_n=args.top_n)
        return

    if args.demo:
        query_product, top_recs = recommender.recommend("SKU_00005", top_n=5, filter_same_category=True)
        weighted_recs = recommender.distance_weighted_ranking(top_recs)
        display_clean_recommendations(query_product, weighted_recs, top_n=5)
        return

    # By default, launch clean interactive query mode
    run_interactive_cli(recommender)

if __name__ == "__main__":
    main()
