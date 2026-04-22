# Demand Monitor Handoff — 2026-04-22

**Status:** Production. Multipage refactor complete, WASDE integration live, IV barometer deployed.

**Latest commit:** `cba4431` — Refactor demand monitor into multipage app with IV barometer & WASDE integration

---

## What's Live

### App Architecture
- **Main entry:** `demand_monitor/app.py` → Home page + shared sidebar + data loading
- **4 user pages:** `demand_monitor/pages/01-04_*.py`
  - `01_Price_Prediction.py` — Historical prices + Shapley decomposition
  - `02_Safe_Zones.py` — S/U scatter chart with guardrails & confidence ellipses
  - `03_Implied_Balance.py` — Scenario-based supply/usage implications
  - `04_Demand_Barometer.py` — IV tracker (speedometer + historical chart + zones)

### Key Features
1. **Sidebar (shared across all pages)**
   - Crop selector (Corn / Soybeans radio button)
   - Supply, Usage, Price inputs (synced sliders + number inputs)
   - WASDE MYA price button (resets to forecast)
   - Guardrail status display
   - Market's implied balance scenarios

2. **Demand Barometer (NEW)**
   - IV metric: current index value vs. 2009 baseline (=100)
   - Speedometer gauge: visual demand strength (red/yellow/green zones)
   - Historical line chart: IV time series 2000–2025
   - Key insights table: S/U, 5-year avg IV, real price
   - Full data table: sortable historical breakdown

3. **Price Source**
   - Old: BarChart.com futures (ZCZ26, ZSX26) — web scraping, unreliable
   - New: WASDE monthly MYA forecast — config dict, reliable, manual monthly update

### Data Loading
- **Corn:** `CornSBDemand.MVPData.xlsx` → "Corn - bu" sheet (million bushels, pre-converted)
- **Soybeans:** Same workbook → "Soybeans - bu" sheet (million bushels, pre-converted)
- **Real prices:** `source of change_corn.xlsx`, `source of change_sb.xlsx` (2025 dollars)
- **Caching:** `@st.cache_data` loads data once at startup, reused across all pages

---

## Bug Fixes Applied

### Corn Units Bug (FIXED)
- **Problem:** Corn sidebar was loading from "Corn" sheet (raw thousand bushels), not "Corn - bu" (converted)
- **Symptom:** Yields were impossible (~3-4 bu/acre for corn)
- **Fix:** Changed `_SHEET["corn"]` from `"Corn"` → `"Corn - bu"` in `data.py:47`
- **Verification:** Corn 2024 = 16,677 Mbu ÷ 90M acres = **185 bu/acre** ✓

### WASDE Price Integration (NEW)
- **Old:** `_get_futures_price_from_barchart()` → BarChart web scraping with retry logic
- **New:** `_get_wasde_mya_price()` → Simple dict lookup from `WASDE_MYA_PRICE` config
- **Update:** Edit `data.py:56-59` monthly with latest WASDE MYA forecast
  ```python
  WASDE_MYA_PRICE = {
      "corn":     3.85,      # ← Update with latest WASDE monthly
      "soybeans": 10.25,
  }
  ```

---

## Multipage Structure

### Session State Persistence
All pages access user inputs via `st.session_state` (automatically persists across navigation):
- `st.session_state.scen_crop` → "Corn" or "Soybeans" (set by sidebar radio)
- `st.session_state.supply_{crop}` → User's supply input
- `st.session_state.usage_{crop}` → User's usage input
- `st.session_state.spot_price_{crop}` → User's price input
- `st.session_state._corn_scen_row`, `_soy_scen_row` → Pre-computed scenario results
- `st.session_state._cached_ols_pred` → Cached OLS prediction

### Sidebar Logic
Remains in `app.py` (main entry point). All pages use the same sidebar—crop selection and scenario inputs carry across page navigation. No need to duplicate sidebar code.

### Pages Design
Each page:
1. Imports `AEI` palette and results dataframes from `app.py`
2. Gets crop choice from `st.session_state.scen_crop`
3. Selects appropriate dataframe (`corn_res`/`soy_res`) and dataset (`corn_df`/`soy_df`)
4. Renders content using shared functions (`render_tab()`, chart functions, etc.)

---

## How to Update Monthly

### WASDE MYA Price (Monthly)
1. Get latest WASDE monthly publication (USDA releases monthly)
2. Find MYA (marketing year average) price forecast for corn & soybeans
3. Edit `demand_monitor/data.py` lines 56–59:
   ```python
   WASDE_MYA_PRICE = {
       "corn":     <NEW_CORN_PRICE>,
       "soybeans": <NEW_SOY_PRICE>,
   }
   ```
4. Commit & push to master
5. Streamlit Cloud auto-deploys within minutes

### Supply/Usage Data (Annually)
- WASDE publishes updated S/U balance sheets annually
- When available, update the Excel workbook: `SOYBEANS/CornSBDemand.MVPData.xlsx`
- Python code will auto-load the new year on next restart

---

## Key Files Reference

| File | Purpose | Lines | Notes |
|------|---------|-------|-------|
| `app.py` | Main entry + sidebar + data loading | ~1,700 | Contains all shared functions & AEI palette |
| `data.py` | Excel loading + WASDE config | ~160 | Update `WASDE_MYA_PRICE` dict monthly |
| `model.py` | CED model formulas | ~220 | No changes needed |
| `shapley.py` | Shapley decomposition + IV calc | ~300 | No changes needed |
| `pages/01_Price_Prediction.py` | Price + decomposition chart | 73 | Uses `render_tab()` from app.py |
| `pages/02_Safe_Zones.py` | S/U scatter + guardrails | 93 | Uses `scenario_ellipse_chart()` |
| `pages/03_Implied_Balance.py` | Scenario implications table | 149 | Uses `_safe_compute_g()` |
| `pages/04_Demand_Barometer.py` | IV tracker (NEW) | 400+ | Standalone page, uses `corn_res`/`soy_res` |

---

## Deployment Status

✅ **GitHub:** Pushed to `jyoungeconomics/aei-demand-monitor` master branch  
✅ **Streamlit Cloud:** Auto-deploying (check your dashboard)  
✅ **Production URL:** Same as before (no URL change)  

---

## Testing Checklist (for next session's edits)

- [ ] Load app locally: `streamlit run demand_monitor/app.py`
- [ ] Navigate to all 4 pages; crop choice persists
- [ ] Sidebar: adjust Supply/Usage/Price on each page; changes persist across navigation
- [ ] Price button: resets to WASDE forecast
- [ ] Barometer page: IV metric + gauge + chart render correctly
- [ ] Deploy to Streamlit Cloud: push to master, verify live within 5 min

---

## Next Steps / Known Items

- Sidebar warnings about empty labels (non-critical; Streamlit deprecation)
- `use_container_width` deprecation warnings (non-critical; will fix in future)
- IV gauge color parsing: currently hardcoding RGBA from hex (works but could be cleaner)

---

## Questions for Next Session

1. Any edits to the barometer page layout or content?
2. Guardrail thresholds or zone colors need adjustment?
3. Chart styling or hover tooltips need tweaks?
4. Additional pages or features planned?

---

**Handoff complete.** App is clean, modular, and ready for edits. WASDE pricing is production-ready (update monthly).
