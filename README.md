# AEI Corn & Soybean Demand Monitor

A subscriber-facing price forecasting tool built on a constant elasticity of demand framework with Shapley decomposition.

## Running locally

```bash
pip install -r requirements.txt
streamlit run demand_monitor/app.py
```

## Model

Price prediction formula: `G = (S/U)^(1/ε)` where S = total supply, U = total use, ε = own-price elasticity of demand.

Predicted price: `y = b₀ + b₁ × IV` where `IV = K × G × P_real` (the S/D Index, normalized to 100 in base year 2009).

The Shapley decomposition attributes each year's price change to supply-side and demand-side factors.

| Crop      | Elasticity (ε)   |
|-----------|-----------------|
| Corn      | −0.1651652067   |
| Soybeans  | −0.1656585432   |

Base year: 2009. Prices in real 2025 dollars.
