import pickle, os, sys
import pandas as pd
import numpy as np

BASE       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE, "models", "best_model.pkl")

VALID_AREAS = [
    "DHA Phase 1-3", "DHA Phase 4-6", "DHA Phase 7-9",
    "Gulberg I-II",  "Gulberg III-V", "Model Town",
    "Johar Town",    "Bahria Town",   "Lake City",
    "Valencia",      "Garden Town",   "Iqbal Town",
    "Wapda Town",    "Allama Iqbal Town", "Faisal Town",
    "Samanabad",     "Township",      "Shahdara",
]

VALID_TYPES = ["House", "Upper Portion", "Lower Portion", "Flat"]

GATED = {"DHA Phase 1-3","DHA Phase 4-6","DHA Phase 7-9","Bahria Town","Lake City","Valencia"}


def load_model():
    if not os.path.exists(MODEL_PATH):
        print("❌ Model not found. Run `python src/train_model.py` first.")
        sys.exit(1)
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


def predict(area, size_marla, bedrooms, bathrooms, property_age,
            property_type="House", has_garage=0, has_servant_quarter=0,
            has_basement=0, corner_plot=0, near_park=0):
    """
    Predict house price in PKR Lakhs.

    Parameters:
        area            : str  — one of VALID_AREAS
        size_marla      : float — plot size in Marlas (e.g. 10 = 10 Marla)
        bedrooms        : int
        bathrooms       : int
        property_age    : int  — years old
        property_type   : str  — one of VALID_TYPES
        has_garage      : 0/1
        has_servant_quarter : 0/1
        has_basement    : 0/1
        corner_plot     : 0/1
        near_park       : 0/1

    Returns:
        predicted price in Lakhs (float)
    """
    model = load_model()
    gated = 1 if area in GATED else 0

    X = pd.DataFrame([{
        "area":                area,
        "size_marla":          size_marla,
        "bedrooms":            bedrooms,
        "bathrooms":           bathrooms,
        "property_age":        property_age,
        "has_garage":          has_garage,
        "has_servant_quarter": has_servant_quarter,
        "has_basement":        has_basement,
        "corner_plot":         corner_plot,
        "near_park":           near_park,
        "gated_society":       gated,
        "property_type":       property_type,
    }])

    price = model.predict(X)[0]
    return round(price, 2)


# ── Interactive demo ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "=" * 55)
    print("  Lahore Housing Price Predictor")
    print("=" * 55)

    examples = [
        dict(area="DHA Phase 4-6",   size_marla=10, bedrooms=4, bathrooms=4, property_age=5,
             property_type="House",  has_garage=1,  has_servant_quarter=1),
        dict(area="Johar Town",       size_marla=7,  bedrooms=3, bathrooms=3, property_age=10,
             property_type="House",  has_garage=1),
        dict(area="Bahria Town",      size_marla=5,  bedrooms=3, bathrooms=2, property_age=2,
             property_type="House"),
        dict(area="Allama Iqbal Town",size_marla=5,  bedrooms=3, bathrooms=2, property_age=15,
             property_type="Upper Portion"),
        dict(area="Model Town",       size_marla=20, bedrooms=5, bathrooms=5, property_age=8,
             property_type="House",  has_garage=1, has_servant_quarter=1, corner_plot=1),
    ]

    for i, props in enumerate(examples, 1):
        price = predict(**props)
        size  = props["size_marla"]
        area  = props["area"]
        beds  = props["bedrooms"]
        ptype = props.get("property_type","House")
        age   = props["property_age"]
        print(f"\n  Example {i}: {size} Marla {ptype} in {area}")
        print(f"    Bedrooms: {beds} | Age: {age} yrs")
        print(f"  ➜ Predicted Price: PKR {price:,.2f} Lakhs  (≈ PKR {price/100:.2f} Crore)")

    print("\n" + "=" * 55)
    print("  Import predict() in your own script to use!")
    print("=" * 55 + "\n")
