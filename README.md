# Airline Customer Segmentation & Churn Prediction

## Project Overview

**Objective:** Segment 62K+ airline loyalty customers using the LRFMC framework and predict churn risk among high-value customers, enabling data-driven marketing and retention strategies.

**Business Value:** Targeted marketing based on distinct customer value profiles and proactive retention outreach for at-risk valuable customers.

---

## Data

- **Data Source:** Chinese airline frequent flyer program dataset
- **Records:** 62,988 customers (62,044 after cleaning)
- **Raw Features:** 44 columns including flight frequency, mileage, membership duration, points activity, discount behavior, and demographics

---

## Methodology

### 1. Data Cleaning

- Dropped rows with missing first-year or second-year ticket prices
- Removed records with zero ticket price, zero average discount, or zero total kilometers
- Final dataset: 62,044 × 44

### 2. LRFMC Feature Engineering

Condensed 44 raw columns into 5 behavioral features using the industry-standard LRFMC framework:

| Code  | Feature          | Definition                                      | Source               |
| ----- | ---------------- | ----------------------------------------------- | -------------------- |
| **L** | Loyalty Duration | Months since joining the frequent flyer program | LOAD_TIME − FFP_DATE |
| **R** | Recency          | Days since last flight                          | LAST_TO_END          |
| **F** | Frequency        | Total flights during observation window         | FLIGHT_COUNT         |
| **M** | Miles Flown      | Total kilometers during observation window      | SEG_KM_SUM           |
| **C** | Discount Rate    | Average discount coefficient across bookings    | avg_discount         |

**How to interpret LRFMC values:**

- L higher is better — longer tenure means more loyal
- R lower is better — smaller gap since last flight means more active
- F higher is better — more flights mean higher engagement
- M higher is better — more kilometers mean higher revenue contribution
- C higher signals price sensitivity and reliance on discounts

### 3. Standardization

All five features scaled to zero mean and unit variance using StandardScaler, so distance-based clustering isn't dominated by features with larger magnitudes (miles in tens of thousands vs. discount rate near 1).

### 4. Optimal k Selection

Evaluated two cluster-quality metrics across k = 3 to 9:

| Metric                  | Best k | Score    |
| ----------------------- | ------ | -------- |
| Silhouette Score        | 6      | ≈ 0.28   |
| Calinski-Harabasz Index | 4      | ≈ 21,800 |

**Final choice: k = 5** — a business-driven compromise between the two metrics. k=4 would merge small but strategically important segments (VIP, Important Retention); k=6 would fragment segments beyond actionable size.

### 5. K-Means Clustering

Trained with `n_clusters=5, random_state=123, n_init=10` on standardized LRFMC features.

### 6. Churn Prediction with LSTM

**Target variable:** For High-Value Customers (n=15,728), churn is defined as `LAST_TO_END > 120 days`. This threshold reflects 2-3x the typical high-value flyer activity cycle, and falls between the median (76 days) and 75th percentile (156 days) of the observed distribution.

**Input features:** 10 behavioral indicators including recent-vs-historical activity ratios (`Ration_L1Y_Flight_Count`, `Ration_L1Y_BPS`), point accumulation (`L1Y_BP_SUM`, `AVG_BP_SUM`), flight frequency (`FLIGHT_COUNT`, `AVG_FLIGHT_COUNT`), total miles (`SEG_KM_SUM`), annual spend (`SUM_YR_1`, `SUM_YR_2`), and discount rate (`avg_discount`).

**Model:** 2-layer LSTM, hidden size 64, BCE loss, Adam optimizer (lr=0.001), Early Stopping with patience=20.

---

## Key Results

### Customer Segmentation — 5 Clusters

Cluster centers reported as z-scores (positive = above average; negative = below average).

| Cluster               | Size   | %     | L     | R     | F     | M     | C     | Standout |
| --------------------- | ------ | ----- | ----- | ----- | ----- | ----- | ----- | -------- |
| General Developmental | 24,611 | 39.7% | −0.70 | −0.41 | −0.16 | −0.16 | −0.26 | R        |
| High-Value            | 15,728 | 25.3% | +1.16 | −0.38 | −0.09 | −0.09 | −0.16 | L, R     |
| VIP                   | 5,337  | 8.6%  | +0.48 | −0.80 | +2.48 | +2.42 | +0.31 | F, M, R  |
| Potential Churn       | 12,111 | 19.5% | −0.31 | +1.69 | −0.57 | −0.54 | −0.18 | None     |
| Important Retention   | 4,257  | 6.9%  | +0.04 | −0.00 | −0.23 | −0.24 | +2.17 | C        |

### Cluster Profiles and Targeted Strategy

| Cluster                   | Behavioral Profile                                          | Marketing / Retention Strategy                                     |
| ------------------------- | ----------------------------------------------------------- | ------------------------------------------------------------------ |
| **General Developmental** | Short tenure, recently flew but low frequency and miles     | Brand promotion and onboarding campaigns to grow engagement        |
| **High-Value**            | Long tenure and still actively flying — the core loyal base | Rewards programs and tenure-based perks to maintain loyalty        |
| **VIP**                   | Longest tenure, highest flight count, highest miles         | Premium services, dedicated support, ambassador programs           |
| **Potential Churn**       | No standout strength; long gap since last flight            | Targeted coupons and re-engagement campaigns, churn-cause analysis |
| **Important Retention**   | Heavy discount reliance, average on other dimensions        | Tiered discount retention before eligibility lapses                |

### Churn Prediction Performance

On High-Value Customers (n=15,728):

- **Accuracy:** 0.71
- **AUC:** 0.77
- **Log Loss:** 0.54

The model identifies high-value customers at risk of churning, so the airline can step in with targeted retention campaigns.

---

## Strategic Recommendations

| Decision                    | Recommendation                                                                                                   |
| --------------------------- | ---------------------------------------------------------------------------------------------------------------- |
| Marketing budget allocation | Differentiate spend by segment: VIP gets service investment, Potential Churn gets defensive discounts            |
| Retention prioritization    | Combine LSTM churn score with segment tier — High-Value members with high churn score get first outreach         |
| Product positioning         | Build two loyalty tracks: service-led for VIP / High-Value, price-led for Potential Churn / Important Retention  |
| New member onboarding       | General Developmental (40% of base) is the growth engine — invest in onboarding to graduate them into High-Value |

---

## Tech Stack

- **Python 3** — Pandas, NumPy
- **Scikit-learn** — StandardScaler, K-Means, Silhouette Score, Calinski-Harabasz Index
- **PyTorch** — LSTM model for churn prediction
- **Matplotlib, Seaborn** — visualization

---

## Repository Structure

```
airline-customer-segmentation/
├── airline_customer_segmentation.ipynb   # Main notebook
├── README.md
├── requirements.txt
└── .gitignore
```

Note: the raw data file `air_data.csv` is not tracked in Git due to size and source constraints.

---

## How to Run

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/airline-customer-segmentation.git
cd airline-customer-segmentation

# 2. Install dependencies
pip install -r requirements.txt

# 3. Place air_data.csv in the project root directory

# 4. Launch Jupyter
jupyter notebook airline_customer_segmentation.ipynb
```

---

## Author

Zihan Luo
