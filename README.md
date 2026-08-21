# Credit Risk Early Warning System

**A production-style credit default prediction system built on Home Credit Group's applicant data — with explainable predictions, cost-sensitive decisioning, and a fairness audit, deployed end-to-end as a live API and dashboard.**

🔗 **Live dashboard:** [credit-risk-loan-default-early-warn.vercel.app](https://credit-risk-loan-default-early-warn.vercel.app)
🔗 **API:** [credit-risk-loan-default-early-warning.onrender.com](https://credit-risk-loan-default-early-warning.onrender.com)
📊 **Dataset:** [Home Credit Default Risk (Kaggle)](https://www.kaggle.com/c/home-credit-default-risk)

> Note: the backend runs on Render's free tier, which spins down after ~15 minutes of inactivity. The first request after idle time may take 30–60 seconds to respond while the server wakes up.

---

## The problem

Lenders approving credit applications face two costs that are rarely equal: approving an applicant who defaults costs far more than rejecting an applicant who would have repaid. Most introductory credit-scoring projects optimize for accuracy and stop there — this one instead asks three questions a real underwriting desk would ask:

1. **How well can we predict default risk** from an applicant's financial profile and credit history?
2. **Why does the model think what it thinks** — can a loan officer explain a rejection to an applicant, or to a regulator?
3. **Where should the decision line actually sit**, given that a missed default is more expensive than a wrongly rejected good applicant — and does the model treat applicants fairly across protected attributes?

## Results at a glance

| Metric | Value |
|---|---|
| Out-of-fold AUC (5-fold CV) | **0.777** |
| Training applicants | 307,511 |
| Features engineered | 156, across 6 linked tables |
| Optimal decision threshold (cost-sensitive) | **0.15** (vs. naive 0.50) |
| `CODE_GENDER` AUC contribution | +0.001 — excluded from final model |

---

## Methodology

### 1. Data engineering

The Home Credit dataset spans six relational tables beyond the core application form — prior credit bureau history, previous loan applications, installment payment records, POS/cash balances, and credit card balances. Each was aggregated to one row per applicant and merged into a single feature matrix:

- **`bureau` + `bureau_balance`** (1.7M + 27M rows) → credit history from other institutions: active credit count, total debt exposure, worst delinquency status
- **`previous_application`** (1.7M rows) → prior Home Credit applications: approval/refusal ratio, average requested vs. granted amount
- **`installments_payments`** (13.6M rows) → payment behavior: late-payment ratio, payment shortfall
- **`POS_CASH_balance`** (10M rows) and **`credit_card_balance`** (3.8M rows) → days-past-due history, credit utilization

Structural missingness was treated as signal, not noise — for example, a missing `OWN_CAR_AGE` reliably means the applicant doesn't own a car, so it's encoded as an explicit flag rather than imputed away. Known dataset anomalies (a `DAYS_EMPLOYED` placeholder value affecting ~18% of applicants) were identified and corrected before feature engineering.

### 2. Modeling

A LightGBM classifier was trained with 5-fold stratified cross-validation to get an honest, out-of-fold performance estimate before training a final model on the full dataset. Categorical features were passed as native LightGBM categoricals rather than one-hot encoded, keeping the feature space compact.

**Result: 0.777 OOF AUC**, consistent across folds (0.773–0.784) — competitive with public single-model baselines on this dataset.

### 3. Explainability (SHAP)

A model that flags an applicant as high-risk without saying why is not deployable in a regulated lending context. SHAP (TreeExplainer) was used to generate both global and per-applicant explanations.

**Figure 1 — Global feature importance.** External credit bureau scores (`EXT_SOURCE_1/2/3`) dominate, followed by the engineered `INST_LATE_RATIO` (payment delinquency) and `PREV_REFUSED_RATIO` (prior refusal history) — confirming that the aggregated features carry real signal, not just noise.

![SHAP Summary](images/Figure%201.png)

**Figure 2 — Per-applicant explanation.** A single applicant's risk score decomposed into the individual factors that pushed it up or down, in the exact units the model saw. This is the artifact a loan officer would actually use to explain a decision.

![SHAP Waterfall](images/Figure%202.png)

### 4. Cost-sensitive threshold optimization

A missed default and a wrongly rejected good applicant are not equally costly. Using a 5:1 cost ratio (false negative : false positive) as a working assumption, the decision threshold was optimized against out-of-fold predictions rather than left at the default 0.5.

**Figure 3 — Threshold optimization.** The cost-minimizing threshold is **0.15** — far below the naive default, reflecting that the model should flag risk more aggressively once the asymmetric cost of a missed default is accounted for.

![Threshold Optimization](images/Figure%203.png)

### 5. Fairness audit

`CODE_GENDER` ranked among the top 5 features by SHAP importance, and the model's flagged-as-risky rate diverged more sharply by gender (M: 19.8%, F: 11.0%) than the underlying actual default rate justified (M: 10.1%, F: 7.0%) — the model was amplifying, not just reflecting, an existing disparity.

Retraining without `CODE_GENDER` cost only **0.001 AUC** (0.777 → 0.776) — the feature's predictive signal was almost entirely redundant with other applicant attributes. Given gender's status as a protected attribute under EU/GDPR-adjacent anti-discrimination frameworks, **the final deployed model excludes `CODE_GENDER`**, trading negligible performance for materially reduced fairness risk.

---

## System architecture

```
┌─────────────────┐        ┌──────────────────┐        ┌────────────────┐
│   Dashboard      │  HTTP  │   FastAPI         │        │  LightGBM Model │
│   (Vercel)       │ ─────► │   (Render)        │ ─────► │  + SHAP         │
│   Static HTML/JS │  POST  │   /predict         │        │  Explainer      │
└─────────────────┘        └──────────────────┘        └────────────────┘
```

- **Backend:** FastAPI serving a single `/predict` endpoint. Loads the trained LightGBM model and a SHAP `TreeExplainer` at startup; returns a risk probability, approve/reject decision (against the tuned threshold), and the top contributing factors per request.
- **Frontend:** Static HTML/CSS/JS dashboard — no build step. Calls the API directly from the browser.
- **Deployment:** Backend on Render (free tier), frontend on Vercel (free tier), both connected to this repository for continuous deployment on push.

---

## Tech stack

**Modeling:** Python, pandas, LightGBM, scikit-learn, SHAP
**Backend:** FastAPI, uvicorn
**Frontend:** HTML, CSS, vanilla JavaScript
**Deployment:** Render (API), Vercel (dashboard)

---

## Running locally

```bash
git clone https://github.com/aaryakhaire/credit-risk-loan-default-early-warning-system.git
cd credit-risk-loan-default-early-warning-system

python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

pip install -r requirements.txt
uvicorn app.main:app --reload
```

The API will be live at `http://127.0.0.1:8000`. Open `frontend/index.html` directly in a browser — update the `fetch()` URL inside it to `http://127.0.0.1:8000/predict` if testing against the local server instead of the deployed one.

---

## Project structure

```
├── app/
│   ├── main.py          # FastAPI app + routes
│   ├── model.py         # Model loading, prediction, SHAP explanation logic
│   └── schemas.py       # Request schema
├── frontend/
│   └── index.html       # Dashboard (static, no build step)
├── models/
│   ├── credit_risk_model.txt   # Trained LightGBM model
│   └── model_metadata.pkl      # Feature list, categorical features, decision threshold
├── images/               # Figures referenced above
└── requirements.txt
```

---

## Limitations and future work

- The public dashboard collects a reduced set of inputs (9 fields) for usability; the full model uses 156 features. Predictions from the live demo should be read as illustrative rather than production-grade, since most features arrive as missing values in that context.
- The cost ratio used for threshold optimization (5:1) is a working assumption, not derived from real loss data — a production deployment would calibrate this against actual historical loss amounts.
- Fairness analysis covered `CODE_GENDER` only; a fuller audit would extend to other potentially correlated attributes (e.g. `NAME_FAMILY_STATUS`, `OCCUPATION_TYPE`) and intersectional effects.
- A natural extension is **Project 2** in this portfolio: a portfolio optimization and market-regime detection system, applying similar rigor (real finance metrics, not just ML accuracy) to a distinct problem in quantitative finance.

---

## Author

Aarya Khaire — B.E. Information Technology, Vidyalankar Institute of Technology
[GitHub](https://github.com/aaryakhaire) · [LinkedIn](www.linkedin.com/in/aarya-khaire-b4a53728a)
