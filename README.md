🏦 Loan Rejection Prediction & Risk Analytics System

A Streamlit web application that predicts loan rejection risk using machine learning, built on the Survey of Consumer Finances (SCF) 2019 dataset.

🆕 Enhanced Edition — What's Different
This version was rebuilt for reliability and usability on top of the original:

Robust file loading — checks multiple likely paths for the dataset and model (SCFP2019.csv, data/SCFP2019.csv, etc.). If a file isn't found, the app shows an upload box instead of a blank page or crash.
One-click sidebar navigation — real buttons instead of radio dots, with a clearly highlighted active page.
Currency in ₹ (INR) — all financial figures across the app use rupees.
Batch Prediction — upload a CSV of many applicants and get predictions for all of them at once, with a downloadable results file.
Prediction History — every single-applicant prediction made in your session is logged and can be exported to CSV.
Live-computed fallbacks — if pre-rendered confusion matrix / ROC curve images aren't found, the app computes and plots them on the fly.
Defensive column handling — pages skip a chart gracefully (with a note) if an expected column is missing from the dataset, instead of crashing.
Downloadable reports — summary stats, dataset export, single prediction reports, and batch results can all be downloaded directly from the app.
📁 Project Structure
loan-rejection-prediction/
├── app.py                          # Main Streamlit application
├── requirements.txt                # Python dependencies
├── README.md                       # This file
├── SCFP2019.csv                    # Dataset (or place in data/SCFP2019.csv)
├── model/
│   ├── loan_rejection_model.pkl    # Trained model (pickle: model, scaler, feature_names, performance)
│   ├── feature_importance.csv      # Optional — feature importance table
│   └── visualizations/
│       ├── confusion_matrix.png    # Optional — pre-rendered confusion matrix
│       ├── model_comparison.png    # Optional — pre-rendered model comparison
│       └── roc_curves.png          # Optional — pre-rendered ROC curves
└── notebooks/                      # Training / EDA notebooks (optional)
Note: SCFP2019.csv and the model/ files are technically optional — the app won't crash without them. You'll just be prompted to upload them in-browser, and some pages (Model Performance, Predict, Batch Prediction) will show a message until a model is available.

⚙️ Setup & Installation
1. Clone the repository
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
2. (Recommended) Create a virtual environment
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
3. Install dependencies
pip install -r requirements.txt
4. Add your data and model files
Place SCFP2019.csv in the project root (or in a data/ subfolder), and your trained model at model/loan_rejection_model.pkl.

5. Run the app
streamlit run app.py
The app will open at http://localhost:8501.

📊 Dataset
Source: Survey of Consumer Finances (SCF) 2019
Records: 23,000+ households
Features: 200+ financial and demographic variables
Target Variable — LOAN_REJECTED
Derived from multiple indicators:

Source column	Meaning
TURNDOWN	Credit application turned down
FEARDENIAL	Fear of denial (didn't apply due to expected rejection)
BNKRUPLAST5	Bankruptcy in the last 5 years
FORECLLAST5	Foreclosure in the last 5 years
LATE60	60+ days late on a payment
HPAYDAY	Payday loan usage
Risk Indicators
Indicator	Rule
High DTI	Debt-to-Income ratio > 0.4
High Leverage	Leverage ratio > 0.5
Low Savings	No savings account
No Emergency Savings	No emergency fund
🤖 Models
The app expects a pickled dictionary at model/loan_rejection_model.pkl with:

{
    "model": <trained sklearn-compatible classifier>,
    "scaler": <fitted StandardScaler (or similar)>,
    "feature_names": [<list of feature columns in training order>],
    "model_name": "Random Forest",           # optional, for display
    "performance": {                          # optional, for display
        "accuracy": 0.78,
        "precision": 0.71,
        "recall": 0.66,
        "f1": 0.68,
        "roc_auc": 0.81
    }
}
Models experimented with during development: Logistic Regression (baseline), Random Forest, and Gradient Boosting.

🖥️ App Pages
Page	Purpose
🏠 Dashboard	Key metrics, rejection-rate breakdowns by risk factor, age, education
📊 Dataset Overview	Data preview, column info, missing-value analysis
📈 Risk Analytics	Correlation matrix, distributions, composite risk score breakdown
🤖 Model Performance	Accuracy/ROC-AUC, feature importance, confusion matrix, ROC curve
🎯 Predict Loan Status	Single-applicant prediction form with a downloadable report
📥 Batch Prediction	Upload a CSV of many applicants, download predictions
🕑 Prediction History	Session log of all single predictions made, exportable to CSV
📚 Documentation	In-app version of this README
🚀 Deploying to Streamlit Community Cloud
Push this repository to GitHub (make sure SCFP2019.csv and the model/ folder are committed — GitHub's web upload supports files up to 25 MB; larger files need Git LFS).
Go to share.streamlit.io, connect your GitHub account, and select this repository.
Set the main file path to app.py.
Deploy — Streamlit Cloud installs everything in requirements.txt automatically.
🛠️ Technical Stack
Frontend: Streamlit
Backend / ML: Python, scikit-learn
Visualization: Plotly
Data Processing: Pandas, NumPy
📄 License
Specify your project's license here (e.g., MIT, Apache 2.0).

📧 Support
For issues or questions, please open an issue in this repository.
