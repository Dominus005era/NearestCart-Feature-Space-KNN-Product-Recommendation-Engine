# NearestCart: Complete Code Explanation Guide
> **File Walkthrough for `recommend_and_benchmark.py`**  
> *Written in simple, clear, and easy-to-understand English with practical explanations of what each block does, why it is needed, and how it works.*

---

## 1. What Is the Goal of This Program?

When an online shopper looks at an item on an e-commerce website (like a smartphone, a dress, or a laptop), the website usually suggests similar items below it. 

Most traditional systems use **Collaborative Filtering**, which looks at past customer purchases. But if a product is brand new, it has zero clicks and zero purchases—this is called the **Cold-Start Problem**.

**NearestCart solves this problem using K-Nearest Neighbors (KNN):**
1. It looks at the product's actual physical and commercial details (price, customer rating, review volume, product weight, and package volume).
2. It balances those numbers using $z$-score feature scaling so that expensive prices do not drown out customer ratings.
3. It finds the mathematically closest alternatives in sub-millisecond speeds using a spatial search tree (KD-Tree).
4. It presents the top recommendations in a clean, user-friendly table showing prices, ratings, and percentage match scores.

---

## 2. Block-by-Block Code Walkthrough

---

### Block 1: Imports and System Environment Settings (Lines 15–30)

```python
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
```

#### What this block does:
* **`os.environ["OMP_NUM_THREADS"] = "1"`:** Sets the number of CPU computation threads to 1. In spatial neighbor searches on single machines, multi-threading adds CPU thread-switching overhead. Locking it to 1 thread makes our speed tests accurate and consistent.
* **`sys`:** Lets Python interact with standard system inputs and exits.
* **`time`:** Measures high-precision query speeds in milliseconds using `time.perf_counter()`.
* **`random`:** Used when testing random catalog products.
* **`argparse`:** Reads command-line options so users can pass flags like `--sku SKU_00005` or `--benchmark`.
* **`numpy` (as `np`):** Fast array math library used for calculating distance percentiles, medians, and matrix operations.
* **`pandas` (as `pd`):** The data-table tool. It loads our CSV file into an in-memory table (called a DataFrame) so we can filter and sort rows.
* **`matplotlib.pyplot` and `seaborn`:** Visualization packages that draw our benchmark curves and confusion matrix heatmap.
* **Scikit-Learn tools:**
  - `StandardScaler`: Normalizes numeric columns so they share the same scale.
  - `NearestNeighbors`: The spatial indexing algorithm that finds the $k$ closest neighbors.
  - `KNeighborsClassifier`: The supervised version used to test how accurately features predict categories.
  - `train_test_split`: Divides data into 80% training and 20% testing sets.
  - `classification_report`, `accuracy_score`, `confusion_matrix`: Calculate precision, recall, and accuracy numbers.

---

### Block 2: Selecting the Feature Columns (Lines 32–38)

```python
FEATURE_COLS = [
    "unit_price",
    "average_user_rating",
    "review_volume",
    "product_weight_grams",
    "package_volume_cm3"
]
```

#### What this block does:
* Defines the 5 continuous numbers that describe each product:
  1. `unit_price`: The retail price in Indian Rupees (INR).
  2. `average_user_rating`: The customer satisfaction score (1.0 to 5.0 stars).
  3. `review_volume`: How many people reviewed the item (shows popularity).
  4. `product_weight_grams`: How heavy the product is.
  5. `package_volume_cm3`: The size of the shipping box (Length $\times$ Width $\times$ Height).
* **Why these 5?** Columns like `sku_id` (a text ID like `SKU_00005`) and `product_category` (text) are labels, not physical measurements. We only do distance math on continuous numerical measurements.

---

### Block 3: The Recommendation Class & Data Setup (Lines 40–80)

```python
class NearestCartRecommender:
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
```

#### What this block does:
* **`class NearestCartRecommender`:** Packages all recommendation logic into a reusable class.
* **`__init__`:** Runs automatically when the class is initialized. It configures the algorithm (`kd_tree` for logarithmic searches) and the distance formula (`euclidean` for straight-line geometric distance).
* **`self.sku_to_idx` and `self.idx_to_sku`:** Two lookup dictionaries. They allow instant lookup of which row number belongs to an SKU ID like `SKU_00100` without scanning the whole dataset.

#### Inside `_load_and_prepare(self)` (Lines 58–80):
```python
    def _load_and_prepare(self):
        self.df = pd.read_csv(self.data_path)
        self.sku_to_idx = {sku: idx for idx, sku in enumerate(self.df["sku_id"])}
        self.idx_to_sku = {idx: sku for idx, sku in enumerate(self.df["sku_id"])}

        self.scaled_features = self.scaler.fit_transform(self.df[FEATURE_COLS])

        self.model = NearestNeighbors(
            n_neighbors=25,
            algorithm=self.algorithm,
            metric=self.metric
        )
        self.model.fit(self.scaled_features)
```
* **Step 1 (Load CSV):** Loads `catalog_product_attributes.csv` into `self.df`.
* **Step 2 (Feature Scaling):** Calls `self.scaler.fit_transform(self.df[FEATURE_COLS])`.  
  *Why is this necessary?* Price can be Rs. 50,000, while ratings are only between 1 and 5. Without scaling, a difference of Rs. 100 would completely overwhelm a 2-star rating difference. `StandardScaler` shifts each column to have a mean of 0 and a standard deviation of 1.
* **Step 3 (Build Spatial Index):** Creates and fits the Scikit-Learn `NearestNeighbors` model on the scaled data. It builds a KD-Tree in memory, organizing data into binary partitions so searches run in fractions of a millisecond.

---

### Block 4: Module 1 — Retrieving Top-N Nearest Neighbors (Lines 82–134)

```python
    def recommend(self, query_sku_id, top_n=5, filter_same_category=True):
```

#### What this block does:
This is the core retrieval function that finds the most similar products.

1. **Input Cleanup (Lines 87–93):**
   ```python
   clean_sku = query_sku_id.strip().upper()
   if not clean_sku.startswith("SKU_"):
       if clean_sku.isdigit():
           clean_sku = f"SKU_{int(clean_sku):05d}"
   ```
   * Makes the input flexible. If a user enters `5`, it automatically expands it to `SKU_00005`. If the SKU does not exist, it stops gracefully and prints an error message.

2. **Fetching the Target Product Vector (Lines 95–97):**
   ```python
   query_idx = self.sku_to_idx[clean_sku]
   query_row = self.df.iloc[query_idx]
   query_vec = self.scaled_features[query_idx].reshape(1, -1)
   ```
   * Finds the exact row index of the product and extracts its 5 standardized values as a 2D vector for the algorithm.

3. **Querying the KD-Tree (Lines 100–105):**
   ```python
   n_candidates = min(len(self.df), max(top_n * 50, 250) if filter_same_category else top_n + 1)
   distances, indices = self.model.kneighbors(query_vec, n_neighbors=n_candidates)
   ```
   * Asks the KD-Tree to retrieve the closest neighboring points. We ask for a candidate pool of 250 neighbors so that even after filtering for the same category, we easily find 5 matching items.

4. **Category Filtering and Similarity Calculation (Lines 107–130):**
   ```python
   for dist, idx in zip(distances, indices):
       if idx == query_idx:
           continue  # Don't recommend the item to itself
       cand_row = self.df.iloc[idx]
       if filter_same_category and cand_row["product_category"] != query_row["product_category"]:
           continue  # Keep recommendations relevant to the department

       similarity_pct = (1.0 / (1.0 + dist)) * 100.0
   ```
   * **The Similarity Formula in Plain English:**  
     Raw Euclidean distance starts at 0 and can go up to infinity. Normal shoppers find raw distance numbers confusing.  
     We convert distance to a clean percentage:
     $$\text{Similarity \%} = \frac{1}{1 + \text{distance}} \times 100$$
     - If distance is 0 (identical item), similarity is 100%.
     - As distance grows, similarity drops smoothly toward 0%.

5. **Packaging Output:** Returns the target product details and a DataFrame of the top 5 closest items.

---

### Block 5: Module 2 — Reciprocal Distance-Weighted Ranking (Lines 136–160)

```python
    def distance_weighted_ranking(self, rec_df, epsilon=1e-5):
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
```

#### What this block does and why it is needed:
* **The Problem:** Standard KNN treats all top-5 neighbors as equally important. But an item with a tiny distance of 0.1 is much closer to what the customer wants than an item with distance 0.3.
* **Step 1 (Inverse Weighting):**  
  `raw_weights = 1.0 / (distance + epsilon)`.  
  Weight is simply 1 divided by distance ($\epsilon = 0.00001$ ensures we never divide by zero). Closer items get huge weights; farther items get tiny weights.
* **Step 2 (Normalize Weights):**  
  `norm_weights = raw_weights / sum(raw_weights)`.  
  Makes all candidate weights add up to 1.0.
* **Step 3 (Composite Priority Score):**  
  We blend 70% of feature similarity with 30% customer reputation (rating $\times 20$, turning a 5-star rating into a 100-point scale), multiplied by the closeness weight:
  $$\text{Priority Score} = (0.70 \times \text{Similarity} + 0.30 \times \text{Rating Score}) \times \text{Weight} \times K$$
* **Result:** The closest, highest-rated item gets pushed to the very top. In our demo, Candidate 1 earns a priority score of 135.31, more than double Candidate 5 (60.63).

---

### Block 6: Displaying Minimal Recommendations (Lines 161–192)

```python
def display_clean_recommendations(query_product, rec_df, top_n=5):
```

#### What this block does:
* Formats recommendation results into a clean, human-friendly presentation table.
* **Target Summary:** Prints the target product SKU, category, price formatted with currency commas (`Rs. 13,728.60`), star rating (`4.3 / 5.0`), and review volume.
* **Recommendations Table:** Displays 6 clean columns: Rank (`#`), `SKU ID`, `Category`, `Price (INR)`, `Rating`, and `Match %` (with a `(Best)` tag next to rank 1).
* **Why it matters:** It strips away all technical noise (raw coordinates, inverse weight fractions, package volume in cm³) so a store manager, buyer, or examiner sees only clear, actionable information.

---

### Block 7: The Minimalist Interactive CLI Loop (Lines 194–226)

```python
def run_interactive_cli(recommender):
    print("\n" + "="*50)
    print("    NearestCart Product Recommendation System")
    print("="*50)
    print("Enter a Product Number or SKU to get recommendations.")
    print("Type 'exit' to quit.")
    print("="*50)

    while True:
        try:
            user_input = input("\nEnter Product (e.g. 5 or SKU_00005): ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting. Thank you!")
            break
        ...
```

#### What this block does:
* Provides an ongoing interactive terminal prompt.
* **How it works:**
  - Prints a simple header.
  - Waits for user input via `input()`.
  - If the user types `exit` or `q`, the loop breaks cleanly.
  - When the user enters a number or SKU ID, it calls `recommend()` followed by `distance_weighted_ranking()` and prints the clean table.
  - If an invalid ID is typed, it catches the error and informs the user without crashing.

---

### Block 8: Module 3 — Latency & Scalability Benchmarking (Lines 228–301)

```python
def run_latency_benchmark(catalog_sizes=[1000, 5000, 10000, 25000, 50000], n_queries=50):
```

#### What this block does:
* **The Question:** Will this search engine slow down when our store adds thousands of new products?
* **How it tests:**
  - Evaluates catalogs across 5 scales: 1,000, 5,000, 10,000, 25,000, and 50,000 products.
  - Runs 50 random test queries at each catalog size.
  - Compares 3 algorithms:
    1. **Brute Force (`brute`):** Checks every single catalog item sequentially ($\mathcal{O}(N \cdot D)$).
    2. **KD-Tree (`kd_tree`):** Organizes points into a binary space-partitioning tree ($\mathcal{O}(D \log N)$).
    3. **Ball-Tree (`ball_tree`):** Organizes points into nested multidimensional spheres.
* **Results & Visuals:**
  - Measures index build time, median query latency, and 95th-percentile tail latency.
  - Saves the results table to `outputs/latency_benchmark_results.csv`.
  - Uses Matplotlib to plot latency scaling curves and saves them to `outputs/latency_benchmark_scaling.png`.
* **Key Finding:** As the catalog expanded 50 times (from 1k to 50k), brute-force latency increased by over 460%, while KD-Tree latency stayed practically flat at **~0.24 milliseconds**, proving it is ready for production.

---

### Block 9: Extension — Supervised KNN Category Classification (Lines 303–340)

```python
def run_supervised_knn(df):
```

#### What this block does:
* **The Question:** Do these 5 physical features actually separate products into meaningful retail departments?
* **How it tests:**
  - Uses `StandardScaler` on all 5 features.
  - Splits the data into 80% training and 20% holdout test sets using stratified sampling (`train_test_split`).
  - Trains a `KNeighborsClassifier(n_neighbors=7, weights="distance")`.
  - Predicts the category of the 1,000 unseen test products.
* **Diagnostic Confusion Matrix:**
  - Computes `confusion_matrix(y_test, y_pred)`.
  - Draws an 8-class heatmap with Seaborn and saves it to `outputs/supervised_knn_confusion_matrix.png`.
* **Key Finding:** Achieves **61.7% accuracy** (compared to 18.6% for random guessing). Furniture reaches an F1-score of 0.97 because heavy weight and large volume form a clean, isolated cluster.

---

### Block 10: Program Entry Point & CLI Arguments (Lines 342–374)

```python
def main():
    parser = argparse.ArgumentParser(description="NearestCart: KNN Product Recommendation Engine")
    parser.add_argument("--sku", type=str, default=None, help="Query specific SKU directly")
    parser.add_argument("--top_n", type=int, default=5, help="Number of recommendations")
    parser.add_argument("--benchmark", action="store_true", help="Run full automated benchmark suite")
    parser.add_argument("--demo", action="store_true", help="Run quick non-interactive demo")
    args = parser.parse_args()

    data_path = os.path.join("data", "catalog_product_attributes.csv")
    recommender = NearestCartRecommender(data_path=data_path, algorithm="kd_tree", verbose=False)

    if args.benchmark:
        run_supervised_knn(recommender.df)
        run_latency_benchmark(...)
        return
    ...
    run_interactive_cli(recommender)

if __name__ == "__main__":
    main()
```

#### What this block does:
* **Command-line flexibility:**
  - If executed with `--benchmark`, it runs the speed tests and saves the charts.
  - If executed with `--sku SKU_00005`, it directly outputs recommendations for that item.
  - **Default:** If executed with no flags (`python recommend_and_benchmark.py`), it starts the clean, interactive prompt for the user.
* **`if __name__ == "__main__":`** Standard Python practice ensuring `main()` runs only when the file is executed directly, not when imported by another script.

---

## 3. Quick Summary Table

| Code Component | What It Does | Why It Is Important |
| :--- | :--- | :--- |
| **`StandardScaler`** | Normalizes all 5 continuous features | Stops high rupee prices from drowning out 5-star ratings. |
| **`NearestNeighbors (kd_tree)`** | Organizes data into spatial trees | Enables sub-millisecond neighbor lookups ($\mathcal{O}(D \log N)$). |
| **`recommend()`** | Finds the 5 closest same-category items | Core recommendation retrieval step (Module 1). |
| **`1.0 / (1.0 + distance) * 100`** | Converts distance to percentage | Gives normal shoppers an intuitive 0–100% match score. |
| **`distance_weighted_ranking()`** | Applies inverse distance weighting | Pushes the closest and highest-rated products to rank #1. |
| **`run_interactive_cli()`** | Simple terminal input loop | Lets anyone type a simple number (like `5`) and get instant recommendations. |
| **`run_latency_benchmark()`** | Compares Brute Force vs. KD-Tree | Scientifically proves that search stays fast up to 50,000 items. |
| **`run_supervised_knn()`** | 8-class category classification | Proves physical attributes carry genuine retail category signals. |
