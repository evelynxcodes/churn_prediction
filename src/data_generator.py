"""
Synthetic Telco-style customer churn dataset generator.

Mirrors the IBM Telco Customer Churn dataset structure and distributions
so the project runs without any external data download.
"""

import numpy as np
import pandas as pd


def generate_churn_data(n_samples: int = 7043, random_state: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    n = n_samples

    # ── Demographics ────────────────────────────────────────────────────────
    gender = rng.choice(["Male", "Female"], n)
    senior_citizen = rng.choice([0, 1], n, p=[0.84, 0.16])
    partner = rng.choice(["Yes", "No"], n)
    dependents = rng.choice(["Yes", "No"], n, p=[0.30, 0.70])

    # ── Account ─────────────────────────────────────────────────────────────
    tenure = rng.gamma(shape=2, scale=18, size=n).clip(1, 72).astype(int)
    contract = rng.choice(
        ["Month-to-month", "One year", "Two year"], n, p=[0.55, 0.21, 0.24]
    )
    paperless_billing = rng.choice(["Yes", "No"], n, p=[0.59, 0.41])
    payment_method = rng.choice(
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)",
        ],
        n,
        p=[0.34, 0.23, 0.22, 0.21],
    )

    # ── Services ─────────────────────────────────────────────────────────────
    phone_service = rng.choice(["Yes", "No"], n, p=[0.90, 0.10])
    multiple_lines = np.where(
        phone_service == "No",
        "No phone service",
        rng.choice(["Yes", "No"], n),
    )
    internet_service = rng.choice(
        ["DSL", "Fiber optic", "No"], n, p=[0.34, 0.44, 0.22]
    )

    def _internet_addon(base, yes_prob: float) -> np.ndarray:
        chosen = rng.choice(["Yes", "No"], n, p=[yes_prob, 1 - yes_prob])
        return np.where(base == "No", "No internet service", chosen)

    online_security = _internet_addon(internet_service, 0.28)
    online_backup = _internet_addon(internet_service, 0.34)
    device_protection = _internet_addon(internet_service, 0.34)
    tech_support = _internet_addon(internet_service, 0.29)
    streaming_tv = _internet_addon(internet_service, 0.38)
    streaming_movies = _internet_addon(internet_service, 0.39)

    # ── Charges ──────────────────────────────────────────────────────────────
    base_charge = np.where(
        internet_service == "Fiber optic", 70,
        np.where(internet_service == "DSL", 45, 20)
    )
    phone_charge = np.where(phone_service == "Yes", rng.uniform(15, 30, n), 0)
    addon_count = (
        (online_security == "Yes").astype(int)
        + (online_backup == "Yes").astype(int)
        + (device_protection == "Yes").astype(int)
        + (tech_support == "Yes").astype(int)
        + (streaming_tv == "Yes").astype(int)
        + (streaming_movies == "Yes").astype(int)
    )
    monthly_charges = (
        base_charge + phone_charge + addon_count * 5 + rng.normal(0, 3, n)
    ).clip(18, 120).round(2)
    total_charges = (monthly_charges * tenure + rng.normal(0, 10, n)).clip(18).round(2)

    # ── Churn label (logistic model of known risk factors) ────────────────────
    score = (
        -0.05 * tenure
        + 0.80 * (contract == "Month-to-month").astype(float)
        - 0.40 * (contract == "Two year").astype(float)
        + 0.50 * (internet_service == "Fiber optic").astype(float)
        + 0.30 * (payment_method == "Electronic check").astype(float)
        + 0.20 * senior_citizen
        - 0.30 * (online_security == "Yes").astype(float)
        - 0.20 * (tech_support == "Yes").astype(float)
        + 0.01 * monthly_charges
        + rng.normal(0, 0.3, n)
        - 1.0
    )
    churn_prob = 1 / (1 + np.exp(-score))
    churn = rng.binomial(1, churn_prob).astype(bool)

    df = pd.DataFrame(
        {
            "customerID": [f"C{i:05d}" for i in range(1, n + 1)],
            "gender": gender,
            "SeniorCitizen": senior_citizen,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "PhoneService": phone_service,
            "MultipleLines": multiple_lines,
            "InternetService": internet_service,
            "OnlineSecurity": online_security,
            "OnlineBackup": online_backup,
            "DeviceProtection": device_protection,
            "TechSupport": tech_support,
            "StreamingTV": streaming_tv,
            "StreamingMovies": streaming_movies,
            "Contract": contract,
            "PaperlessBilling": paperless_billing,
            "PaymentMethod": payment_method,
            "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges,
            "Churn": np.where(churn, "Yes", "No"),
        }
    )
    return df
