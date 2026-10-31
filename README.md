# A Deep Learning Framework for Predicting CO₂ Emissions from Gas Flaring in Nigeria's Niger Delta

##### Satellite-Based Spatio-Temporal Modelling for Emissions Monitoring and Policy Support



\[!\[Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)

\[!\[PyTorch](https://img.shields.io/badge/PyTorch-CUDA-EE4C2C.svg)](https://pytorch.org/)

\[!\[License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)



\---



###### \## Executive Summary \& Project Motivation



Nigeria consistently ranks among the top gas-flaring nations globally, with routine flaring across the Niger Delta representing a dual crisis: millions of tonnes of preventable $\\text{CO}\_2$ equivalent emissions entering the atmosphere annually, and billions of dollars in wasted natural gas that could otherwise power domestic energy infrastructure. 



While Nigeria has introduced ambitious regulatory frameworks—including the \*\*2018 Flare Gas (Prevention of Waste and Pollution) Regulations\*\*, the \*\*Petroleum Industry Act (PIA) 2021\*\*, and the \*\*Nigerian Gas Flare Commercialisation Programme (NGFCP)\*\*—effective enforcement and carbon accounting remain bottlenecked by a critical data gap:

1\. \*\*Reporting Lags \& Opacity:\*\* Self-reported operator logs and metered flare volumes are often delayed, aggregated monthly, or subject to reporting discrepancies.

2\. \*\*Monitoring Blind Spots:\*\* Public satellite trackers (such as the NOSDRA Gas Flare Tracker) provide historical visibility but lack predictive, forward-looking capabilities to warn regulators of imminent threshold exceedances or evaluate policy shocks in real time.



This project—developed for my M.Sc. in Data Science dissertation at Pan-Atlantic University—introduces an \*\*end-to-end spatio-temporal deep learning framework\*\* that transforms daily satellite thermal anomalies (VIIRS Fire Radiative Power) and macroeconomic indicators into calibrated, multi-horizon $\\text{CO}\_2$ emission forecasts and automated regulatory compliance alarms across \*\*124 active flare sites\*\* in the Niger Delta.



\---



###### \## Strategic Impact: Climate, Policy \& Energy Sector Applications



This framework is designed as a decision-support tool for three primary stakeholders:



\* \*\*Environmental Regulators (NOSDRA, NUPRC, Federal Ministry of Environment):\*\* Provides an independent, satellite-verified early-warning system. By utilizing calibrated 90th-percentile ($P\_{90}$) probabilistic prediction intervals, regulators can move from reactive auditing to proactive exceedance alerting (achieving \*\*85% precision and 78% recall\*\* on extreme flaring spikes).

\* \*\*Energy Transition \& Gas Commercialisation (NGFCP \& Midstream Investors):\*\* Identifies persistent "super-emitting" clusters (Gini coefficient = \*\*0.7542\*\*) and quantifies site-level supply stability across wet and dry seasons, helping midstream developers de-risk gas-capture infrastructure investments.

\* \*\*Climate Policy \& Carbon Markets:\*\* Evaluates how structural interventions—such as the 2018 penalty hike, COVID-19 production shocks, and the PIA 2021—actually impacted flaring trajectories on the ground, isolating policy-driven abatement from macroeconomic volatility (Brent crude price shifts and Naira exchange rate fluctuations).



\---



###### \## Methodological Pipeline



Rather than relying on static annual estimates, this pipeline reconstructs and models daily emissions across \*\*4,372 days (2012–2024)\*\* using a five-stage architecture:



1\. \*\*Physical Calibration \& Panel Reconstruction:\*\* Converts daily VIIRS/FIRMS Fire Radiative Power (FRP) detections into gas volume (BCM/MSCF) and $\\text{CO}\_2$ mass (Tonnes), calibrated against World Bank Global Gas Flaring Reduction (GGFR) and NNPC benchmarks (\*\*15.01% MAPE\*\* post-calibration across 74,666 site-day observations).

2\. \*\*Spatio-Temporal Feature Engineering:\*\* Integrates 24 domain-specific predictors, including autoregressive persistence lags (1 to 30 days), rolling volatility windows, harmonic seasonal encodings, atmospheric cloud cover flags (handling tropical wet-season occlusion), and macroeconomic drivers (`brent\_spot\_price`, `fx\_rate`).

3\. \*\*Econometric \& Spatial Diagnostics:\*\* Validates panel dynamics via Augmented Dickey-Fuller stationarity testing ($p < 0.0001$), Hurst exponent memory analysis ($H = 0.2093$, confirming mean-reverting operational spikes), and Global Moran's $I$ ($I = 0.0965, p = 0.0320$), confirming significant spatial clustering across neighboring flow stations.

4\. \*\*Multi-Horizon Deep Learning \& Graph Topologies:\*\* Benchmarks tree-based ensembles (\*\*XGBoost\*\*), recurrent networks (\*\*Stabilized LSTM/RNN\*\*), \*\*Temporal Fusion Transformers (TFT)\*\*, and \*\*Spatio-Temporal Graph Neural Networks (DCRNN, Graph WaveNet, ST-GCN)\*\* constructed over a $124 \\times 124$ geodesic adjacency matrix (mean inter-site distance: 131.3 km).

5\. \*\*Probabilistic Calibration \& Policy Attribution:\*\* Generates $P\_{10}, P\_{50},$ and $P\_{90}$ quantile forecasts evaluated via Pinball Loss and Prediction Interval Coverage Probability (PICP), alongside SHAP feature attribution and structural event-window error tracking.



\---



###### \## Key Empirical Findings



\### 1. Multi-Horizon Forecasting Accuracy (WAPE %)

Temporal Fusion Transformers (TFT) and XGBoost demonstrate strong predictive stability from short-term operational horizons (1-day and 7-day) up to quarterly strategic planning horizons (90-day), capturing both national trajectories and site-level intermittency:



| Model Architecture | Aggregation Tier | 1-Day Horizon | 7-Day Horizon | 30-Day Horizon | 90-Day Horizon |

| :--- | :---: | :---: | :---: | :---: | :---: |

| \*\*Temporal Fusion Transformer (TFT)\*\* | National-Level | \*\*7.60%\*\* | \*\*8.20%\*\* | \*\*10.50%\*\* | \*\*16.50%\*\* |

| \*\*XGBoost\*\* | National-Level | 8.66% | 9.62% | 13.30% | 22.90% |

| \*\*Recurrent Neural Network (RNN)\*\* | National-Level | 10.22% | 11.54% | 16.60% | 29.80% |

| \*\*Temporal Fusion Transformer (TFT)\*\* | Site-Level (124 Sites) | \*\*12.15%\*\* | \*\*13.05%\*\* | \*\*16.50%\*\* | \*\*25.50%\*\* |

| \*\*XGBoost\*\* | Site-Level (124 Sites) | 13.72% | 15.04% | 20.10% | 33.30% |

| \*\*Recurrent Neural Network (RNN)\*\* | Site-Level (124 Sites) | 15.30% | 17.10% | 24.00% | 42.00% |



\### 2. What Drives Flaring Emissions? (SHAP \& Spatial Findings)

\* \*\*Operational Inertia Dominates:\*\* SHAP attribution reveals that immediate operational persistence (`lag\_1`), radiative heat intensity, 7-day rolling averages, and gas volume dominate daily predictions, while exchange rate fluctuations (`fx\_rate`) and seasonal harmonics capture macro-level regime shifts.

\* \*\*Site-Independent Attention vs. Graph Diffusion:\*\* Two-sided Diebold-Mariano ($p < 0.001$) ablation tests across DCRNN and Graph WaveNet variants ($k \\in \\{5, 8, 10\\}$) show that while flare sites are geographically clustered, individual flow-station flaring is governed primarily by localized wellhead pressure and facility maintenance schedules rather than physical spatial diffusion across sites.

\* \*\*Zero-Shot Spatial Vulnerability:\*\* Leave-Region-Out robustness testing reveals that while models are invariant to emission-factor adjustments ($0.66\\%$ variation) and random seed initialization ($\\pm 0.15\\%$), forecasting on completely unseen geographic sub-regions increases WAPE from \*\*7.60% to 13.90%\*\*, underscoring the need for localized sensor calibration when onboarding new concessions.



\### 3. Regulatory Uncertainty Quantification

For policy enforcement, point forecasts are insufficient—regulators require confidence bounds to distinguish measurement noise from true regulatory violations. Our calibrated quantile models achieved an empirical coverage (PICP) of \*\*80.27%–80.37%\*\* against an 80% nominal target, with TFT providing the tightest, most actionable uncertainty intervals (`364,668.54 Tonnes` average width at 1-day horizon).



\---



###### \## Repository Structure 



```text

├── data/                  

├── notebooks/             

├── outputs/               

├── requirements.txt       

└── README.md              




\###### \*\*ENVIRONMENT SETUP\*\*

git clone \\\[https://github.com/your-username/niger-delta-co2-forecasting.git](https://github.com/Jesselyn-Ayanka/niger-delta-co2-forecasting.git)
cd niger-delta-co2-forecasting
pip install -r requirements.txt


###### \*\* Citation\*\*

If you reference this methodology, dataset reconstruction, or empirical findings, please cite:

@mastersthesis{ayanka2026co2,

\&#x20; author       = {Jesselyn Oritsemisan Ayanka},

\&#x20; title        = {A Deep Learning Framework for Predicting CO2 Emissions from Gas Flaring in Nigeria's Niger Delta: Satellite-Based Spatio-Temporal Modelling for Emissions Monitoring and Policy Support},

\&#x20; school       = {Pan-Atlantic University, Lagos, Nigeria},

\&#x20; year         = {2026}

}


