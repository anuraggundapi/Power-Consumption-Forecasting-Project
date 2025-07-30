# ⚡ Power Consumption Forecasting Project ⚡

This repository contains a comprehensive Data Science project focused on **forecasting power consumption** using machine learning and deep learning models, with deployment of the best-performing model as an interactive **Streamlit web application**.

## 📊 Project Overview

The primary goal is to **accurately predict future power consumption** based on historical data. Accurate forecasts are essential for **energy planning, resource allocation, and grid management**. This project explores multiple **time series forecasting techniques**, from classical statistical models to advanced **deep learning architectures** and **ensemble regressors**.

## 🚀 Features

* **Data Preprocessing**: Handling missing values, outliers, and time series transformations.
* **Model Building**: Implementation of various models:

  * **Prophet Model**: Captures trends, seasonality, and holiday effects.
  * **Deep Neural Network (DNN)**: Combines Conv1D and Bidirectional LSTM layers.
  * **Regression Models**: Linear Regression, Random Forest, XGBoost with lag features.
* **Model Evaluation**: Metrics like **MAE**, **RMSE**, and **MAPE** for performance benchmarking.
* **Interactive Streamlit App**: Allows real-time forecasting and visualization.
* **Model Persistence**: Trained model saved using `pickle` for deployment.

## 🗂️ Project Structure

```
.
├── P562_Forecasting_Group_3.ipynb          # Jupyter Notebook with analysis, model training, and evaluation
├── P562_Forecasting_Group_3.ipynb - Colab.pdf # PDF export of the notebook
├── app.py.py                               # Streamlit web application code
├── daily_data_last_7.csv                   # Last 7 days of daily power consumption (for app input)
├── dataset.csv                             # Original raw dataset
├── dataset_daily.csv                       # Aggregated daily dataset
└── final_rf_model.pkl                      # Pre-trained Random Forest model used in Streamlit app
```

## 🧠 Models Used

### 1. Prophet Model

* **Type**: Univariate Time Series Forecasting.
* **Strengths**: Captures trend, multiple seasonalities (daily/weekly/yearly), robust to missing data and outliers.
* **Application**: Hourly power consumption forecasting.

### 2. Deep Neural Network (DNN)

* **Type**: Sequence-to-Sequence Time Series Forecasting.
* **Architecture**: `Conv1D` → `Bidirectional LSTM` → `Dense` layers.
* **Application**: Hourly consumption forecasting using sliding windows.

### 3. Regression Models (Linear Regression, Random Forest, XGBoost)

* **Type**: Supervised Regression using Lag Features.
* **Features**: Past 7 days’ power consumption as inputs.
* **Best Performer**: **XGBoost** achieved the highest accuracy. However, **Random Forest** was chosen for the final Streamlit deployment due to its balance of simplicity and performance.

## 📈 Performance Highlights

| Model               | RMSE     | MAE      | MAPE    |
| ------------------- | -------- | -------- | ------- |
| **XGBoost (Daily)** | \~267.43 | \~191.01 | \~3.57% |

* **Prophet Model** captured hourly patterns well.
* **DNN performance** on hourly forecasting was **suboptimal** compared to Prophet.
* *(Hourly models and daily models operate on different granularities; hence direct metric comparison isn't appropriate.)*

## 🖥️ Run the Streamlit App Locally

### 1. Clone the Repository:

```bash
git clone https://github.com/your-username/power-consumption-forecast.git
cd power-consumption-forecast
```

### 2. (Optional but Recommended) Create a Virtual Environment:

```bash
python -m venv venv
# Windows:
.\venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate
```

### 3. Install Required Libraries:

Create a `requirements.txt` with:

```
streamlit
pandas
numpy
matplotlib
scikit-learn
# Include these if used:
# tensorflow
# xgboost
# prophet
```

Then install:

```bash
pip install -r requirements.txt
```

### 4. Ensure Required Files are Present:

Make sure these files are in the project directory:

* `final_rf_model.pkl`
* `daily_data_last_7.csv`
* `dataset_daily.csv`

### 5. Run the Streamlit App:

```bash
streamlit run app.py.py
```

The app will open in your browser, typically at `http://localhost:8501`.
