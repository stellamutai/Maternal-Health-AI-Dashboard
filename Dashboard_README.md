# Maternal Health Dashboard (Streamlit)

This is the interactive dashboard component of the Maternal Mortality Prediction system. It allows users to visualize trends, manipulate health indicators via sliders, and receive real-time risk assessments.

## Features
- **Interactive Sidebar**: Adjust `Year`, `Skilled Birth Attendance`, and `Antenatal Care` using intuitive sliders.
- **Real-time Forecasting**: Uses the trained XGBoost model to predict MMR instantaneously.
- **Dynamic Visualization**: Combines historical WHO data (2000-2023) with your future prediction on a single line chart.
- **Risk Alerts**: Automatic color-coded alerts (Green/Orange/Red) based on the predicted severity.

## Prerequisites
Ensure you have the project dependencies installed:
```bash
pip install -r requirements.txt
```
*Note: This includes `streamlit`, `pandas`, `xgboost`, and `scikit-learn`.*

## Running Locally
To launch the dashboard on your machine:

1. Open your terminal in the project directory.
2. Run the Streamlit command:
   ```bash
   streamlit run dashboard.py
   ```
3. The dashboard will automatically open in your default browser at `http://localhost:8501`.

## Deployment (Streamlit Cloud)
This dashboard is ready for **Streamlit Cloud** (the easiest way to share it).

1. Push your code to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io/).
3. Connect your GitHub account and select this repository.
4. Set the "Main file path" to `dashboard.py`.
5. Click **Deploy**.

## File Structure
- `dashboard.py`: The main application code.
- `engineered_maternal_mortality.csv`: Historical data source.
- `best_model.pkl`: The trained machine learning model.
