<h1>Spatio Temporal Deep Learning for Gas Flaring CO2 Emissions Monitoring in Nigeria's Niger Delta</h1>
<h3>Bridging Satellite Remote Sensing and Predictive Analytics for Climate Policy and Energy Regulation</h3>

<p>
  <img src="https://img.shields.io/badge/python-3.8%2B-blue.svg" alt="Python 3.8+">
  <img src="https://img.shields.io/badge/PyTorch-CUDA-EE4C2C.svg" alt="PyTorch">
  <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT">
</p>

<h2> Summary and Project Motivation</h2>

<p>Nigeria consistently ranks among the top gas flaring nations globally, with routine flaring across the Niger Delta representing a dual crisis: millions of tonnes of preventable CO2 equivalent emissions entering the atmosphere annually, and billions of dollars in wasted natural gas that could otherwise power domestic energy infrastructure.</p>

<p>While Nigeria has introduced ambitious regulatory frameworks, including the <b>2018 Flare Gas (Prevention of Waste and Pollution) Regulations</b>, the <b>Petroleum Industry Act (PIA) 2021</b>, and the <b>Nigerian Gas Flare Commercialisation Programme (NGFCP)</b>, effective enforcement and carbon accounting remain bottlenecked by a critical data gap:</p>

<ol>
  <li><b>Reporting Lags and Opacity:</b> Self reported operator logs and metered flare volumes are often delayed, aggregated monthly, or subject to reporting discrepancies.</li>
  <li><b>Monitoring Blind Spots:</b> Public satellite trackers (such as the NOSDRA Gas Flare Tracker) provide historical visibility but lack predictive, forward looking capabilities to warn regulators of imminent threshold exceedances or evaluate policy shocks in real time.</li>
</ol>

<p>This project, developed for my M.Sc. in Data Science dissertation at Pan Atlantic University, introduces an <b>end to end spatio temporal deep learning framework</b> that transforms daily satellite thermal anomalies (VIIRS Fire Radiative Power) and macroeconomic indicators into calibrated, multi horizon CO2 emission forecasts and automated regulatory compliance alarms across <b>124 active flare sites</b> in the Niger Delta.</p>

<h2>Strategic Impact: Climate, Policy and Energy Sector Applications</h2>

<p>This framework is designed as a decision support tool for three primary stakeholders:</p>

<ul>
  <li><b>Environmental Regulators (NOSDRA, NUPRC, Federal Ministry of Environment):</b> Provides an independent, satellite verified early warning system. By utilizing calibrated 90th percentile (P90) probabilistic prediction intervals, regulators can move from reactive auditing to proactive exceedance alerting (achieving <b>85% precision and 78% recall</b> on extreme flaring spikes).</li>
  <li><b>Energy Transition and Gas Commercialisation (NGFCP and Midstream Investors):</b> Identifies persistent super emitting clusters (Gini coefficient = <b>0.7542</b>) and quantifies site level supply stability across wet and dry seasons, helping midstream developers reduce risk in gas capture infrastructure investments.</li>
  <li><b>Climate Policy and Carbon Markets:</b> Evaluates how structural interventions, such as the 2018 penalty hike, COVID 19 production shocks, and the PIA 2021, actually impacted flaring trajectories on the ground, isolating policy driven abatement from macroeconomic volatility (Brent crude price shifts and Naira exchange rate fluctuations).</li>
</ul>

<h2>Methodological Pipeline</h2>

<p>Rather than relying on static annual estimates, this pipeline reconstructs and models daily emissions across <b>4,372 days (2012 to 2024)</b> using a five stage architecture:</p>

<ol>
  <li><b>Physical Calibration and Panel Reconstruction:</b> Converts daily VIIRS and FIRMS Fire Radiative Power (FRP) detections into gas volume (BCM and MSCF) and CO2 mass (Tonnes), calibrated against World Bank Global Gas Flaring Reduction (GGFR) and NNPC benchmarks (<b>15.01% MAPE</b> post calibration across 74,666 site day observations).</li>
  <li><b>Spatio Temporal Feature Engineering:</b> Integrates 24 domain specific predictors, including autoregressive persistence lags (1 to 30 days), rolling volatility windows, harmonic seasonal encodings, atmospheric cloud cover flags (handling tropical wet season occlusion), and macroeconomic drivers (Brent spot price, FX rate).</li>
  <li><b>Econometric and Spatial Diagnostics:</b> Validates panel dynamics via Augmented Dickey Fuller stationarity testing (p &lt; 0.0001), Hurst exponent memory analysis (H = 0.2093, confirming mean reverting operational spikes), and Global Moran's I (I = 0.0965, p = 0.0320), confirming significant spatial clustering across neighboring flow stations.</li>
  <li><b>Multi Horizon Deep Learning and Graph Topologies:</b> Benchmarks tree based ensembles (<b>XGBoost</b>), recurrent networks (<b>Stabilized LSTM and RNN</b>), <b>Temporal Fusion Transformers (TFT)</b>, and <b>Spatio Temporal Graph Neural Networks (DCRNN, Graph WaveNet, ST GCN)</b> constructed over a 124x124 geodesic adjacency matrix (mean inter site distance: 131.3 km).</li>
  <li><b>Probabilistic Calibration and Policy Attribution:</b> Generates P10, P50, and P90 quantile forecasts evaluated via Pinball Loss and Prediction Interval Coverage Probability (PICP), alongside SHAP feature attribution and structural event window error tracking.</li>
</ol>

<h2>Key Empirical Findings</h2>

<h3>1. Multi Horizon Forecasting Accuracy (WAPE %)</h3>
<p>Temporal Fusion Transformers (TFT) and XGBoost demonstrate strong predictive stability from short term operational horizons (1 day and 7 day) up to quarterly strategic planning horizons (90 day), capturing both national trajectories and site level intermittency:</p>

<table>
  <thead>
    <tr>
      <th align="left">Model Architecture</th>
      <th align="center">Aggregation Tier</th>
      <th align="center">1 Day Horizon</th>
      <th align="center">7 Day Horizon</th>
      <th align="center">30 Day Horizon</th>
      <th align="center">90 Day Horizon</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><b>Temporal Fusion Transformer (TFT)</b></td>
      <td align="center">National Level</td>
      <td align="center"><b>7.60%</b></td>
      <td align="center"><b>8.20%</b></td>
      <td align="center"><b>10.50%</b></td>
      <td align="center"><b>16.50%</b></td>
    </tr>
    <tr>
      <td><b>XGBoost</b></td>
      <td align="center">National Level</td>
      <td align="center">8.66%</td>
      <td align="center">9.62%</td>
      <td align="center">13.30%</td>
      <td align="center">22.90%</td>
    </tr>
    <tr>
      <td><b>Recurrent Neural Network (RNN)</b></td>
      <td align="center">National Level</td>
      <td align="center">10.22%</td>
      <td align="center">11.54%</td>
      <td align="center">16.60%</td>
      <td align="center">29.80%</td>
    </tr>
    <tr>
      <td><b>Temporal Fusion Transformer (TFT)</b></td>
      <td align="center">Site Level (124 Sites)</td>
      <td align="center"><b>12.15%</b></td>
      <td align="center"><b>13.05%</b></td>
      <td align="center"><b>16.50%</b></td>
      <td align="center"><b>25.50%</b></td>
    </tr>
    <tr>
      <td><b>XGBoost</b></td>
      <td align="center">Site Level (124 Sites)</td>
      <td align="center">13.72%</td>
      <td align="center">15.04%</td>
      <td align="center">20.10%</td>
      <td align="center">33.30%</td>
    </tr>
    <tr>
      <td><b>Recurrent Neural Network (RNN)</b></td>
      <td align="center">Site Level (124 Sites)</td>
      <td align="center">15.30%</td>
      <td align="center">17.10%</td>
      <td align="center">24.00%</td>
      <td align="center">42.00%</td>
    </tr>
  </tbody>
</table>

<h3>2. What Drives Flaring Emissions? (SHAP and Spatial Findings)</h3>
<ul>
  <li><b>Operational Inertia Dominates:</b> SHAP attribution reveals that immediate operational persistence (lag_1), radiative heat intensity, 7 day rolling averages, and gas volume dominate daily predictions, while exchange rate fluctuations (fx_rate) and seasonal harmonics capture macro level regime shifts.</li>
  <li><b>Site Independent Attention vs. Graph Diffusion:</b> Two sided Diebold Mariano (p &lt; 0.001) ablation tests across DCRNN and Graph WaveNet variants (k in {5, 8, 10}) show that while flare sites are geographically clustered, individual flow station flaring is governed primarily by localized wellhead pressure and facility maintenance schedules rather than physical spatial diffusion across sites.</li>
  <li><b>Zero Shot Spatial Vulnerability:</b> Leave Region Out robustness testing reveals that while models are invariant to emission factor adjustments (0.66% variation) and random seed initialization (±0.15%), forecasting on completely unseen geographic sub regions increases WAPE from <b>7.60% to 13.90%</b>, underscoring the need for localized sensor calibration when onboarding new concessions.</li>
</ul>

<h3>3. Regulatory Uncertainty Quantification</h3>
<p>For policy enforcement, point forecasts are insufficient because regulators require confidence bounds to distinguish measurement noise from true regulatory violations. Our calibrated quantile models achieved an empirical coverage (PICP) of <b>80.27% to 80.37%</b> against an 80% nominal target, with TFT providing the tightest, most actionable uncertainty intervals (364,668.54 Tonnes average width at a 1 day horizon).</p>

<h2>Repository Structure</h2>

<pre>
data/                  World Bank GGFR benchmarks, macroeconomic series, and reconstructed panels
notebooks/             Master GPU execution notebook (EDA, Training, Evaluation, Figures)
outputs/               High resolution spatial maps, SHAP plots, and evaluation matrices
requirements.txt       Python environment dependencies
README.md              Project overview and policy brief
</pre>

<h2>Reproducibility and Environment Setup</h2>

<p>1. Clone the repository and install the required Python packages:</p>

<pre><code>git clone https://github.com/Jesselyn-Ayanka/Niger-Delta-CO2_forecasting.git
cd Niger-Delta-CO2_forecasting
pip install -r requirements.txt</code></pre>

<p>2. Launch the execution notebook in Jupyter Lab or a CUDA enabled environment (such as a Kaggle P100 or T4 GPU instance) and run all cells sequentially:</p>

<pre><code>jupyter notebook notebooks/niger_delta_co2_forecasting.ipynb</code></pre>

<h2>Citation</h2>

<pre>
@mastersthesis{ayanka2026co2,
  author       = {Jesselyn Oritsemisan Ayanka},
  title        = {A Deep Learning Framework for Predicting CO2 Emissions from Gas Flaring in Nigeria's Niger Delta: Satellite Based Spatio Temporal Modelling for Emissions Monitoring and Policy Support},
  school       = {Pan Atlantic University, Lagos, Nigeria},
  year         = {2026}
}
</pre>
