# 🔬 Generalizing Food Labeling Effects: A TMLE-BART Causal Framework

[![Language](https://img.shields.io/badge/Language-R-276DC3.svg)](https://www.r-project.org/)
[![Dataset](https://img.shields.io/badge/Dataset-RCT_%2B_NDNS_Survey-blue.svg)](#)
[![Models](https://img.shields.io/badge/Models-TMLE--BART_%7C_XGBoost_SDG-lightgrey.svg)](#)
[![Status](https://img.shields.io/badge/Status-Under_Review-orange.svg)](#)

> **Executive Summary:** This repository provides the full reproducible R pipeline for estimating the **Population Average Treatment Effect (PATE)** of food labeling on consumer calorie choices, by integrating a Randomized Controlled Trial (RCT) with a nationally representative survey (NDNS) using Targeted Maximum Likelihood Estimation with Bayesian Additive Regression Trees (TMLE-BART).

## 🌍 Why This Matters

* **From Lab to Population:** RCTs offer strong internal validity but limited external generalizability. This research bridges that gap by transporting experimental treatment effects to a UK-representative population using rigorous causal inference.
* **Policy-Relevant Evidence:** The PATE estimates provide actionable, population-level evidence for policymakers designing front-of-pack nutritional labeling regulations.
* **Methodological Innovation:** Combines XGBoost-based synthetic data generation with ELSA-calibrated cognitive decline penalties and doubly-robust TMLE estimation under full bootstrap uncertainty propagation.

## 📊 Dataset

The analysis harmonizes two complementary data sources into a unified analytical framework.

| Data Source | Coverage | Sample | Role |
| :--- | :--- | :--- | :--- |
| **RCT** | 2024 | Online Experimental Survey | Treatment effect estimation (food label conditions: Absent, Coarse, Detailed) |
| **NDNS** | 2008–2023 (Waves 1–15) | UK National Diet and Nutrition Survey | Target population for PATE generalization |

- **Key Variables:** Demographics, BMI, cognitive capacity (d-prime working memory scores), calorie intake, survey weights, treatment group assignment.
- **Data File:** `iso_df_v2.csv`

## 🔬 Methodology Pipeline

`Bootstrap Resampling` ➔ `XGBoost Imputation (mixgb)` ➔ `Synthetic Cognition Generation (XGBoost + ELSA Penalties)` ➔ `Propensity Score Estimation (BART)` ➔ `Doubly-Robust TMLE Outcome Modeling` ➔ `PATE Estimation with 95% CIs`

## 🤖 Models & Analysis Benchmarked

| Analytical Method | Purpose in Study |
| :--- | :--- |
| **MICE-XGBoost (mixgb)** | Global imputation of missing covariates across pooled RCT and NDNS data using gradient-boosted trees. |
| **XGBoost Synthetic Data Generation** | Predicts cognitive scores (d-prime) for the NDNS population, calibrated with ELSA age-decline penalties. |
| **Piecewise Linear Model** | Appendix robustness check providing a parametric alternative to the XGBoost synthesis engine. |
| **BART (Propensity Score)** | Non-parametric estimation of trial membership probabilities for inverse-probability weighting. |
| **TMLE-BART (Doubly Robust)** | Core estimator combining outcome modeling with propensity score targeting to yield bias-corrected PATE estimates. |
| **Sensitivity Analysis (±10%)** | Systematic variation of synthetic cognitive scores to assess robustness of causal conclusions. |

## 🚀 How to Run Locally

### 1. Prerequisites

Ensure you have **R (≥ 4.2)** or **RStudio** installed with the following packages:

```R
install.packages(c("tidyverse", "dbarts", "xgboost", "mixgb", "magrittr", "mice", "dplyr"))
```

### 2. Execution

1. Clone the repository to your local machine.
2. Ensure `iso_df_v2.csv` is in the same directory as the R script.
3. Run from the command line (for batch execution on a cluster):

```bash
Rscript causal_inference_analysis.R 1
```

Or open in RStudio and execute interactively. The script produces:
- Diagnostic validation tables (cognitive scores by age bracket and survey year).
- PATE estimates with bootstrap 95% confidence intervals across 12 model specifications.
- Results saved as `.rds` files for downstream aggregation.

## 📁 Project Structure

The repository is kept intentionally flat for maximum reproducibility:

* `causal_inference_analysis.R` — The primary reproducible analysis pipeline implementing the full TMLE-BART framework with bootstrap uncertainty.
* `iso_df_v2.csv` — The harmonized dataset combining RCT experimental data and NDNS survey waves.

## 📄 Citation

This paper is currently **under review** at the *Journal of the Royal Statistical Society*. Citation details will be updated upon publication.

> Avalos-Valdebenito, C. (2026). *Generalizing Food Labelling Trials under Unmeasured Confounding: A Doubly Robust BART Approach with Synthetic Cognitive Proxies*. Manuscript submitted for publication in the Journal of the Royal Statistical Society.

---
👤 **Author**: Constanza Avalos-Valdebenito
