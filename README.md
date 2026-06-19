# Flipkart Grid Hackathon — Traffic Intelligence

AI-driven systems for traffic management across multiple hackathon themes.

## Themes

### Theme 1: Illegal Parking Intelligence
**[Theme 1/](Theme%201/)** — AI system that identifies illegal parking hotspots and quantifies their traffic impact.

- HDBSCAN spatial-temporal clustering (1,086 hotspots from 115K violations)
- Greenshields + HCM traffic impact estimation
- LightGBM predictive model (AUC 0.638)
- Streamlit dashboard with interactive heatmaps
- Daily enforcement dispatch recommendations

**Quick start:**
```bash
cd "Theme 1"
pip install -r requirements.txt
python main.py
streamlit run dashboard/app.py
```

### Traffic Gridlock (Trixie)
**[Trafic _Gridlock/](Trafic%20_Gridlock/)** — Travel demand prediction using LightGBM/XGBoost/CatBoost ensemble.

### Demo Theme 3
**[Demo theme 3/](Demo%20theme%203/)** — Camera-based license plate detection.

## Dataset
- `jan to may police violation_anonymized791b166.csv` — 298K illegal parking violations from Bengaluru (Nov 2023 – Mar 2024)

## Research
- `Research Papers/flip1.pdf`, `flip3.pdf` — Reference papers
