# ML-CICD-Assignment

A CI/CD pipeline for a simple Machine Learning model, built with **scikit-learn**, **pytest** and **GitHub Actions**, and deployed to **GitHub Pages**.

**Live page:** https://abhinavboby.github.io/ML-CICD-Assignment/

## Overview

| Item | Details |
|------|---------|
| Model | Logistic Regression (scikit-learn) |
| Dataset | Iris (150 samples, 4 features, 3 classes) |
| Train/Test split | 80% / 20% (stratified, `random_state=42`) |
| Accuracy | ~96.67% |
| Testing | pytest |
| CI/CD | GitHub Actions |
| Deployment | GitHub Pages |

## Repository Structure

```
ML-CICD-Assignment/
├── model.py            # Loads data, trains model, predicts, prints accuracy
├── test_model.py       # pytest test cases
├── requirements.txt    # Python dependencies
├── index.html          # Results page template deployed to GitHub Pages
├── README.md
└── .github/
    └── workflows/
        └── cicd.yml    # CI/CD workflow
```

## How It Works

### CI stage (`ci` job)
Runs automatically on every push to `main`:
1. Checkout the source code
2. Set up Python 3.11
3. Install dependencies from `requirements.txt`
4. Train/run the ML program (`python model.py`), which saves the results to `metrics.json`
5. Run the test cases (`pytest -v`)

### CD stage (`cd` job)
Uses `needs: ci`, so it runs **only if CI passes**:

```
CI → PASS → CD → Deployment
```

1. Reads the model name, dataset and accuracy from `metrics.json`
2. Fills them into `index.html`
3. Deploys the page to GitHub Pages

The deployed page shows the model name, dataset used, model accuracy and the CI/CD deployment status.

## Tests

`test_model.py` checks that:
- the dataset loads with the correct shape,
- the model trains and makes predictions,
- the accuracy is at least 90%.

## Run Locally

```bash
pip install -r requirements.txt
python model.py
pytest -v
```

Expected output:

```
Model: Logistic Regression
Dataset: Iris
Accuracy: 96.67%
```

## Setup Note

In the repository, go to **Settings → Pages** and set **Source** to **GitHub Actions** so that the CD stage can deploy.
