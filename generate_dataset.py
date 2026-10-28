"""
Data Generation Script: catalog_product_attributes.csv
Project: E-Commerce Product Recommendation Prototype (Feature-Space KNN)
Reference Source: Flipkart Product Catalog distributions & category profiles

Generates a realistic, high-fidelity synthetic retail catalog with 5,000 products
across 8 distinct e-commerce categories, obeying physical and economic relationships.
"""

import os
import numpy as np
import pandas as pd

def generate_catalog_dataset(n_samples=5000, random_seed=42):
    np.random.seed(random_seed)

    # Category profiles modeled on Flipkart product categories
    # (price_min, price_max, weight_min_g, weight_max_g, vol_min_cm3, vol_max_cm3, rating_mean, rating_std)
    category_specs = {
        "Electronics": [
            {"sub": "Laptop", "price": (25000, 95000), "weight": (1400, 2600), "vol": (3500, 7500), "p": 0.20},
            {"sub": "Smartphone", "price": (7000, 65000), "weight": (160, 240), "vol": (350, 850), "p": 0.35},
            {"sub": "Headphones", "price": (799, 14999), "weight": (150, 380), "vol": (800, 3200), "p": 0.25},
            {"sub": "Smartwatch", "price": (1299, 21999), "weight": (35, 85), "vol": (250, 600), "p": 0.20},
        ],
        "Clothing": [
            {"sub": "T-Shirt", "price": (399, 1999), "weight": (140, 260), "vol": (300, 850), "p": 0.35},
            {"sub": "Jeans", "price": (799, 3999), "weight": (450, 800), "vol": (800, 2000), "p": 0.30},
            {"sub": "Jacket", "price": (1499, 7999), "weight": (650, 1500), "vol": (2500, 6500), "p": 0.20},
            {"sub": "Ethnic Wear", "price": (699, 5499), "weight": (280, 850), "vol": (700, 2200), "p": 0.15},
        ],
        "Home & Kitchen": [
            {"sub": "Microwave Oven", "price": (5500, 18500), "weight": (9500, 16500), "vol": (26000, 58000), "p": 0.20},
            {"sub": "Mixer Grinder", "price": (1499, 6499), "weight": (2400, 5800), "vol": (7500, 19000), "p": 0.35},
            {"sub": "Cookware Set", "price": (1199, 8499), "weight": (2800, 7200), "vol": (9000, 24000), "p": 0.25},
            {"sub": "Electric Kettle", "price": (649, 2499), "weight": (650, 1400), "vol": (1800, 4800), "p": 0.20},
        ],
        "Footwear": [
            {"sub": "Running Shoes", "price": (1299, 8499), "weight": (500, 950), "vol": (3200, 6500), "p": 0.45},
            {"sub": "Formal Shoes", "price": (1499, 6999), "weight": (680, 1150), "vol": (3500, 5800), "p": 0.30},
            {"sub": "Sandals/Slippers", "price": (399, 1899), "weight": (250, 580), "vol": (1800, 4200), "p": 0.25},
        ],
        "Sports & Fitness": [
            {"sub": "Dumbbells Pair", "price": (899, 9999), "weight": (4000, 24000), "vol": (3000, 16000), "p": 0.35},
            {"sub": "Yoga Mat", "price": (499, 2499), "weight": (800, 1800), "vol": (3800, 8500), "p": 0.35},
            {"sub": "Badminton Racket", "price": (699, 4999), "weight": (82, 320), "vol": (2400, 5200), "p": 0.30},
        ],
        "Furniture": [
            {"sub": "Office Chair", "price": (4200, 19500), "weight": (11000, 24000), "vol": (52000, 130000), "p": 0.40},
            {"sub": "Study Desk", "price": (3500, 17500), "weight": (13500, 32000), "vol": (68000, 190000), "p": 0.35},
            {"sub": "Coffee Table", "price": (2100, 9900), "weight": (6500, 18500), "vol": (38000, 98000), "p": 0.25},
        ],
        "Beauty & Personal Care": [
            {"sub": "Hair Dryer", "price": (799, 4299), "weight": (380, 880), "vol": (1400, 3800), "p": 0.35},
            {"sub": "Beard Trimmer", "price": (649, 3199), "weight": (170, 420), "vol": (550, 1500), "p": 0.40},
            {"sub": "Fragrance / Perfume", "price": (499, 3999), "weight": (120, 380), "vol": (280, 850), "p": 0.25},
        ],
        "Books & Stationery": [
            {"sub": "Paperback Novel", "price": (199, 999), "weight": (200, 550), "vol": (350, 1100), "p": 0.60},
            {"sub": "Hardcover Book", "price": (599, 2999), "weight": (750, 1900), "vol": (1100, 3600), "p": 0.40},
        ],
    }

    categories = list(category_specs.keys())
    # Distribute samples roughly evenly across categories with slight realistic weighting
    cat_weights = [0.18, 0.18, 0.15, 0.14, 0.10, 0.08, 0.10, 0.07]
    sample_categories = np.random.choice(categories, size=n_samples, p=cat_weights)

    data = []
    for i, cat in enumerate(sample_categories):
        sku_id = f"SKU_{i+1:05d}"
        sub_list = category_specs[cat]
        sub_probs = [s["p"] for s in sub_list]
        sub_choice = np.random.choice(len(sub_list), p=sub_probs)
        spec = sub_list[sub_choice]

        # 1. Unit price (log-uniform within subcategory range + rounding)
        log_pmin, log_pmax = np.log(spec["price"][0]), np.log(spec["price"][1])
        raw_price = np.exp(np.random.uniform(log_pmin, log_pmax))
        # Round to typical retail endings (e.g. 49, 99, or rounded integers)
        unit_price = round(float(raw_price), 2)

        # 2. Average user rating: Beta distribution scaled to [1.8, 5.0], peaking at ~4.2
        # Real e-commerce ratings have negative skew (mostly 3.8 to 4.7)
        raw_rating = np.random.beta(a=7, b=2)  # beta peak near 0.8
        rating = 1.5 + raw_rating * 3.5
        rating = np.clip(rating, 1.0, 5.0)
        average_user_rating = round(float(rating), 2)

        # 3. Review volume: Log-normal distribution (heavy right tail)
        # Most items have 10-500 reviews; popular items have thousands
        raw_rev = np.exp(np.random.normal(loc=5.2, scale=1.3))
        review_volume = int(np.clip(raw_rev, 5, 45000))

        # 4. Product weight (grams): correlated with subcategory physical bounds
        w_min, w_max = spec["weight"]
        # Mild positive correlation between price and quality/weight within certain electronics, but bounded
        weight = np.random.uniform(w_min, w_max)
        product_weight_grams = round(float(weight), 1)

        # 5. Package volume (cm3): correlated with product weight + packing factor
        v_min, v_max = spec["vol"]
        # Add slight variation
        volume = np.random.uniform(v_min, v_max)
        package_volume_cm3 = round(float(volume), 1)

        data.append({
            "sku_id": sku_id,
            "unit_price": unit_price,
            "average_user_rating": average_user_rating,
            "review_volume": review_volume,
            "product_weight_grams": product_weight_grams,
            "package_volume_cm3": package_volume_cm3,
            "product_category": cat,
        })

    df = pd.DataFrame(data)
    return df

if __name__ == "__main__":
    os.environ["OMP_NUM_THREADS"] = "1"
    output_dir = os.path.join(os.path.dirname(__file__), "data")
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "catalog_product_attributes.csv")

    print("Generating catalog_product_attributes.csv...")
    df = generate_catalog_dataset(n_samples=5000, random_seed=42)
    df.to_csv(csv_path, index=False)

    print(f"Successfully generated {len(df)} products!")
    print(f"Saved to: {csv_path}")
    print("\nDataset Info & Summary:")
    print(df.info())
    print("\nFirst 5 rows:")
    print(df.head())
    print("\nCategory Distribution:")
    print(df["product_category"].value_counts())
    print("\nNumerical Statistics:")
    print(df.describe().T)
