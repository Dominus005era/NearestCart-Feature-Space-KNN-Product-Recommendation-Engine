# NearestCart: Feature-Space KNN Product Recommendation Engine
> **An Unsupervised & Distance-Weighted Nearest-Neighbor Prototype for Cold-Start Retail Discovery**

[![Python Version](https://img.shields.io/badge/Python-3.9%20%7C%203.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.0%2B-orange.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status: Complete](https://img.shields.io/badge/Status-Complete-brightgreen.svg)]()

---

## 1. Project Overview & Business Motivation

In modern e-commerce ecosystems (e.g., Amazon, Flipkart, Walmart), product discovery on Product Detail Pages (PDP) directly dictates cart addition rates, average order value (AOV), and overall conversion.

While traditional **Collaborative Filtering (CF)** algorithms rely heavily on past user interaction matrices (clickstream, purchase histories, ratings), they suffer from the critical **cold-start problem**: newly launched items or niche inventory lack interaction history and remain invisible to recommendation algorithms.

**NearestCart** provides a content-based, **Feature-Space K-Nearest Neighbors (KNN)** recommendation engine that computes similarity strictly from intrinsic physical, commercial, and customer-reputation attributes:
- **No User History Required:** Solves the cold-start challenge for newly listed SKUs instantly.
- **Continuous Multi-Attribute Geometry:** Operates over a 5-dimensional feature space spanning pricing, customer rating, review volume, item mass, and volumetric package size.
- **Dual-Phase Ranking Engine:** Combines unweighted spatial candidate retrieval ($k$-NN) with reciprocal distance weighting ($w_i = \frac{1}{d_i + \epsilon}$) to penalize distant items and prioritize high-affinity recommendations.
- **Production Scalability:** Benchmarks tree-based spatial partitioning structures (**KD-Tree** and **Ball-Tree**) against flat brute-force search across catalogs ranging from 1,000 to 50,000 products.

---

## 2. Dataset Architecture & Schema

The engine operates on `catalog_product_attributes.csv`, consisting of **5,000 retail catalog records** distributed across 8 product categories:

| Attribute Name | Data Type | Role in Pipeline | Domain Range | Description |
| :--- | :--- | :--- | :--- | :--- |
| `sku_id` | String | Unique Identifier | `SKU_00001` to `SKU_05000` | Primary key; excluded from distance metrics. |
| `unit_price` | Float | Continuous Feature | Rs. 199.00 – Rs. 93,658.81 | Retail price in Indian Rupees (INR). |
| `average_user_rating` | Float | Continuous Feature | 1.84 – 4.99 (Mean: 4.21) | 5-star customer feedback rating. |
| `review_volume` | Integer | Continuous Feature | 5 – 25,558 reviews | Log-normally distributed customer review count. |
| `product_weight_grams`| Float | Continuous Feature | 35.0g – 31,778.3g | Net physical product weight in grams. |
| `package_volume_cm3` | Float | Continuous Feature | 250.0 – 189,798.3 cm³ | 3D packaged shipping volume ($L \times W \times H$). |
| `product_category` | String | Filter / Target | 8 Classes | Department category (Electronics, Clothing, Furniture, etc.). |

### Data Quality & Integrity Audit
- **Null / Missing Values:** 0 across all 7 fields.
- **Duplicate SKUs:** 0 duplicate identifiers.
- **Category Representation:** Electronics (933), Clothing (895), Home & Kitchen (724), Footwear (709), Sports & Fitness (502), Beauty & Personal Care (466), Furniture (417), Books & Stationery (354).

---

## 3. End-to-End System Architecture

```text
                     +---------------------------------------+
                     |    catalog_product_attributes.csv     |
                     |         (5,000 Product SKUs)          |
                     +-------------------+-------------------+
                                         |
                                         v
                     +---------------------------------------+
                     |         Data Integrity Audit          |
                     |  - Zero Missing Values Validation     |
                     |  - Log & Scaler Distribution Checks   |
                     +-------------------+-------------------+
                                         |
                                         v
                     +---------------------------------------+
                     |        StandardScaler Pipeline        |
                     |          z = (x - mu) / sigma         |
                     |  (Equalized 5D Continuous Features)   |
                     +-------------------+-------------------+
                                         |
                                         v
                     +---------------------------------------+
                     |        Spatial Tree Indexing          |
                     |   KD-Tree / Ball-Tree Architecture    |
                     +---------+-------------------+---------+
                               |                   |
            +------------------+                   +------------------+
            v                                                         v
+-------------------------------+                         +-------------------------------+
|  MODULE 1: Top-N Retrieval    |                         |     SUPERVISED EXTENSION      |
|  - Euclidean Distance Matrix  |                         |  - KNeighborsClassifier       |
|  - Category-Aware Filtering   |                         |  - Stratified 80:20 Split     |
|  - Bounded Similarity (%):    |                         |  - 8-Class Category Accuracy  |
|    S = 1 / (1 + d) * 100%     |                         |  - Confusion Matrix Diagnostic|
+---------------+---------------+                         +-------------------------------+
                |
                v
+-------------------------------+                         +-------------------------------+
|  MODULE 2: Distance Weighting |                         |      MODULE 3: BENCHMARK      |
|  - Reciprocal Weight:         |                         |  - Scales: 1k, 5k, 10k, 25k,  |
|    w_i = 1 / (d_i + epsilon)  |                         |    50k products               |
|  - Normalized Weight W_i      |                         |  - Algorithms: Brute Force vs.|
|  - Priority Score:            |                         |    KD-Tree vs. Ball-Tree      |
|    (0.70*S + 0.30*R*20)*W_i*N |                         |  - Latency: Median & p95 tail |
+-------------------------------+                         +-------------------------------+
```

---

## 4. Mathematical Methodology & Formulation

### 4.1 Feature Space Standardization
Because `package_volume_cm3` spans up to $189,798\text{ cm}^3$ and `unit_price` reaches Rs. $93,658$, whereas `average_user_rating` varies only between $1.0$ and $5.0$, direct Euclidean calculation would be over $99.8\%$ dominated by scale variance. 

We apply z-score transformation via `StandardScaler`:
$$z = \frac{x - \mu}{\sigma}$$
guaranteeing each attribute possesses zero mean ($\mu = 0$) and unit variance ($\sigma = 1$).

### 4.2 Module 1: Unsupervised Top-$K$ Retrieval & Bounded Similarity
The pairwise distance between query vector $\mathbf{x}$ and candidate vector $\mathbf{y}$ in standardized 5-dimensional feature space is calculated as:
$$d(\mathbf{x}, \mathbf{y}) = \sqrt{\sum_{j=1}^{5} (x_j - y_j)^2}$$

Raw Euclidean distance is unbounded ($d \in [0, \infty)$). For front-end user interfaces, distance is mapped to an intuitive percentage score:
$$\text{Similarity Score (\%)} = \frac{1}{1 + d(\mathbf{x}, \mathbf{y})} \times 100$$
- When $d = 0$ (identical vector), $\text{Similarity} = 100\%$.
- As $d \to \infty$, $\text{Similarity} \to 0\%$.

### 4.3 Module 2: Reciprocal Distance-Weighted Ranking
To prevent distant items from receiving equal weight in recommendation lists, candidate neighbors are dynamically weighted using an inverse distance formulation:
$$w_i = \frac{1}{d_i + \epsilon}$$
where $\epsilon = 10^{-5}$ guarantees numerical stability. Weights are normalized across the top-$K$ candidate set:
$$W_i = \frac{w_i}{\sum_{j=1}^{K} w_j}$$

The final composite **Weighted Priority Score** combines geometric proximity with customer rating reputation:
$$\text{Weighted Priority Score} = \left( 0.70 \times \text{Similarity Score} + 0.30 \times (\text{Rating} \times 20) \right) \times W_i \times K$$

---

## 5. Empirical Results & Performance Diagnostics

### 5.1 Sample Query Demonstration (`SKU_00005` - Electronics Item)
- **Target Attributes:** Unit Price: Rs. 13,728.60 | User Rating: 4.29 / 5.0 | Review Count: 379 | Weight: 293.4g | Volume: 2,245.9 cm³

#### Module 1: Top-5 Unweighted Retrieval
| Rank | Candidate SKU | Unit Price (Rs.) | Rating | Reviews | Euclidean Distance | Similarity Score (%) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | `SKU_03413` | Rs. 13,467.30 | 4.25 | 384 | 0.1062 | **90.40%** |
| **2** | `SKU_03296` | Rs. 13,539.85 | 4.24 | 302 | 0.1586 | **86.31%** |
| **3** | `SKU_00540` | Rs. 13,243.14 | 4.29 | 253 | 0.1648 | **85.85%** |
| **4** | `SKU_00629` | Rs. 14,019.57 | 4.20 | 297 | 0.2182 | **82.09%** |
| **5** | `SKU_02808` | Rs. 16,323.89 | 4.27 | 394 | 0.2214 | **81.87%** |

#### Module 2: Distance-Weighted Ranking Breakdown
| Weighted Rank | Candidate SKU | Unit Price (Rs.) | Euclidean Distance | Normalized Weight ($W_i$) | Weighted Priority Score |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | `SKU_03413` | Rs. 13,467.30 | 0.1062 | **0.3048** | **135.31** |
| **2** | `SKU_03296` | Rs. 13,539.85 | 0.1586 | 0.2041 | **87.63** |
| **3** | `SKU_00540` | Rs. 13,243.14 | 0.1648 | 0.1964 | **84.31** |
| **4** | `SKU_00629` | Rs. 14,019.57 | 0.2182 | 0.1484 | **61.32** |
| **5** | `SKU_02808` | Rs. 16,323.89 | 0.2214 | 0.1462 | **60.63** |

*Key Insight:* Notice that `SKU_03413` ($d = 0.1062$) receives a priority score of **135.31**, more than double that of `SKU_02808` (**60.63**), mathematically amplifying the most relevant recommendation.

---

### 5.2 Module 3: Scalability & Latency Benchmark

Query latency was evaluated across 5 catalog scales ($N \in \{1\text{k}, 5\text{k}, 10\text{k}, 25\text{k}, 50\text{k}\}$) over 50 repeated query iterations:

| Catalog Size ($N$) | Algorithm | Index Build Time | Median Latency (p50) | 95th Percentile Latency (p95) |
| :---: | :---: | :---: | :---: | :---: |
| **1,000 items** | Brute Force | 0.2 ms | 0.130 ms | 0.317 ms |
| 1,000 items | KD-Tree | 0.6 ms | 0.224 ms | 0.379 ms |
| 1,000 items | Ball-Tree | 0.9 ms | 0.351 ms | 0.495 ms |
| **10,000 items** | Brute Force | 0.2 ms | 0.197 ms | 0.360 ms |
| 10,000 items | KD-Tree | 5.1 ms | **0.225 ms** | **0.272 ms** |
| 10,000 items | Ball-Tree | 3.6 ms | 0.246 ms | 0.429 ms |
| **50,000 items** | Brute Force | 1.1 ms | 0.608 ms | 0.891 ms |
| 50,000 items | **KD-Tree** | **30.9 ms** | **0.237 ms (Sub-ms)** | **0.523 ms (Best)** |
| 50,000 items | Ball-Tree | 21.6 ms | 0.341 ms | 0.540 ms |

#### Scalability Analysis:
1. **Brute Force:** Exhibits linear search complexity $\mathcal{O}(N \cdot D)$. Latency nearly quadruples as catalog size reaches 50,000 items.
2. **KD-Tree:** Demonstrates logarithmic complexity $\mathcal{O}(D \log N)$. Median latency stays remarkably flat at **~0.23 ms** across a **50x increase in catalog scale**, proving production readiness for high-throughput retail platforms.

---

## 6. Repository Structure

```text
NearestCart-Feature-Space-KNN-Product-Recommendation-Engine/
│
├── data/
│   └── catalog_product_attributes.csv           <- 5,000-record retail product dataset
│
├── outputs/
│   ├── latency_benchmark_results.csv            <- Tabulated latency scaling results
│   ├── latency_benchmark_scaling.png            <- Latency scaling curves (Brute vs KD vs Ball)
│   └── supervised_knn_confusion_matrix.png      <- 8-class confusion matrix diagnostic
│
├── report/                                      <- Project reports, presentation deck & visual assets
│
├── .gitignore                                   <- Environment and bytecode ignore rules
├── CODE_EXPLANATION.md                          <- Detailed line-by-line beginner code explanation
├── generate_dataset.py                          <- Reproducible dataset generation engine
├── project_walkthrough.ipynb                    <- Interactive Jupyter Notebook walkthrough
├── recommend_and_benchmark.py                   <- Production Python execution script
├── requirements.txt                             <- Python package dependencies
└── README.md                                    <- Master project documentation
```

---

## 7. Installation & Quickstart Guide

### Step 1: Clone the Repository
```bash
git clone https://github.com/Dominus005era/NearestCart-Feature-Space-KNN-Product-Recommendation-Engine.git
cd NearestCart-Feature-Space-KNN-Product-Recommendation-Engine
```

### Step 2: Set Up Python Virtual Environment
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux / macOS:
source venv/bin/activate
```

### Step 3: Install Required Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Run the Recommendation CLI
```bash
python recommend_and_benchmark.py
```
*How it works:*
- **Minimalist Prompt:** Simply enter a product number (e.g. `5` or `SKU_00005`) to instantly see the top 5 recommended alternatives with prices, ratings, and match percentages.
- **Exit:** Type `exit` to quit.

**Direct Command-Line Usage:**
```bash
# Query a specific product directly:
python recommend_and_benchmark.py --sku SKU_00005 --top_n 5

# Run automated academic benchmark suite:
python recommend_and_benchmark.py --benchmark
```

### Step 5: Launch the Interactive Jupyter Walkthrough
```bash
jupyter notebook project_walkthrough.ipynb
```

---

## 8. Author & Academic Context

- **Author:** Rahul
- **Institution:** United Institute of Technology, Prayagraj
- **Affiliation:** Dr. A.P.J. Abdul Kalam Technical University (AKTU), Lucknow
- **Course:** Machine Learning Mini Project (Project 30)
- **Domain:** Retail & E-Commerce Machine Learning Systems

---

## 9. License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for full details.
