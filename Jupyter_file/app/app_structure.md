pneumonia_app/
│
├── app.py                  # Main Streamlit runner
├── menu.py                 # Sidebar/option menu with grouped navigation
│
├── overview/
│   ├── dashboard.py        # Home/dashboard view
│   └── data_overview.py    # EDA and data summary
│
├── analysis/
│   ├── insights.py         # Visual analytics, correlations, distributions
│   └── prediction_model.py # Model results, risk scoring
│
├── business/
│   ├── kpi_tracker.py      # Key metrics (accuracy, diagnosis time)
│   ├── business_impact.py  # Efficiency, cost savings, patient outcomes
│   └── recommendations.py  # Deployment strategy, scaling, next steps
│
├── doctor/
│   ├── patient_info.py     # Demographic, clinical metadata
│   ├── patient_image.py    # Upload & display scan/X-ray
│   ├── patient_notes.py    # Notes from patient visits
│   └── doctor_notes.py     # Clinical assessments and recommendations
│
└── utils.py                # Shared functions/utilities
