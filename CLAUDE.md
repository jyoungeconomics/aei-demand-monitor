# Corn & Soybean Demand Monitor

## Project Overview

This is a commodity price forecasting tool built on a **constant elasticity of demand** framework with **Shapley decomposition**. It currently lives in Excel and needs to be converted into a standalone, user-facing web application and pushed to GitHub.

The monitor tracks corn and soybean markets and produces model-implied price predictions based on USDA supply and usage data. The target audience is agricultural market subscribers (producers, traders, consultants).

## Core Model

### Price Prediction Formula

```
G = (S / U)^(1 / ε)
```

Where:
- **G** = predicted price ratio (relative to base year)
- **S** = supply (production + beginning stocks + imports)
- **U** = usage (total disappearance: feed, food/seed/industrial, ethanol, exports)
- **ε** = demand elasticity

### Key Parameters

| Commodity | Elasticity (ε) | Base Year |
|-----------|----------------|-----------|
| Corn      | −0.1656585432  | 2009      |
| Soybeans  | TBD — same framework, replicated from corn | 2009 |

### How It Works

1. Pull USDA WASDE supply/usage balance sheet data
2. Compute the S/U ratio for each marketing year
3. Raise to the power (1/ε) to get the predicted price ratio G
4. Multiply by the base-year price to get the predicted price level
5. Decompose changes in G across components using Shapley values

### Shapley Decomposition

The Shapley decomposition attributes the change in predicted price to individual supply and usage components (e.g., "how much of the price move came from exports vs. feed usage vs. production?"). Each component's marginal contribution is calculated by averaging its effect across all possible orderings of the other components.

### Soybean Replication

The soybean model uses the same structure as corn. **Known issue from prior debugging:** the soybean sheet originally had S0/U0 reference columns still pointing to corn template values. This has been corrected in the Excel version. Verify that base-year references are correct when porting to code.

**Open question:** A near-zero 2025 soybean predicted change may reflect either a genuine model result or a placeholder price artifact. Investigate when porting.

## Subscriber-Facing Features

### Guardrails for User Inputs

Subscribers can adjust assumptions (e.g., override a USDA export forecast). The guardrail system uses **output-based thresholds anchored to predicted price**:

- Rather than capping individual inputs, the system flags when a subscriber's adjusted scenario produces a predicted price outside a plausible range
- The plausible range is defined relative to the model's baseline prediction (e.g., ±X% or ±Y standard deviations of historical forecast error)
- This lets subscribers explore scenarios freely while flagging implausible combinations

### Visualization

The Excel version includes charts. The app version should include:

- Time series of predicted vs. actual price
- S/U ratio history
- Shapley decomposition waterfall chart (which components drove the price change)
- An interactive widget where subscribers can adjust individual S/U components and see the predicted price update in real time
- A confidence ellipse visualization (eigenvector-based angle calculation — this was corrected in a prior canvas-based prototype)

## Technical Direction

### Current State
- All logic lives in Excel workbooks (corn and soybean sheets)
- Charts and a canvas-based interactive widget have been prototyped in conversations with Claude on claude.ai

### Target Architecture
- **Language:** Python (for data processing and model logic)
- **Web framework:** TBD — keep it simple; this is a dashboard, not a complex app. Streamlit, Dash, or a lightweight Flask/FastAPI + HTML frontend are all reasonable
- **Data source:** USDA WASDE reports (consider automating the pull via USDA API or NASS QuickStats)
- **Deployment:** GitHub repo → hosted somewhere subscribers can access (Streamlit Cloud, Railway, Render, etc.)
- **Users:** Agricultural market participants; the interface should be clean, professional, and not require technical knowledge

### Conversion Priorities (in order)

1. **Extract the model logic from Excel into Python** — get the core G = (S/U)^(1/ε) calculation working with hardcoded data first
2. **Reproduce the Shapley decomposition in Python** — verify it matches Excel output
3. **Build a basic dashboard** — predicted price, S/U chart, Shapley waterfall
4. **Add the interactive scenario tool** — let users adjust components
5. **Automate data ingestion** — pull from USDA programmatically
6. **Add guardrails** — output-based threshold system
7. **Deploy and set up GitHub** — push to repo, deploy to web

## Conventions & Preferences

- Write clean, well-commented code — the project owner is an agricultural economist, not a software engineer
- Explain technical decisions in plain language when proposing changes
- Use descriptive variable names that match the economics (e.g., `supply`, `usage`, `elasticity`, not `s`, `u`, `e`)
- When in doubt, ask before making architectural choices
- Test against the Excel workbook outputs to verify correctness at each step
- Keep dependencies minimal — don't add libraries unless there's a clear reason

## Files in This Project

- Excel workbooks containing the corn and soybean models (to be added)
- This CLAUDE.md file

## Related Work (Context Only)

The project owner is also working on an academic paper (AJAE submission) involving a Cournot duopoly model for soybean seed with dicamba herbicide drift as a network externality. That is a separate project — do not conflate it with this demand monitor. However, the soybean demand elasticity work may inform both projects.
