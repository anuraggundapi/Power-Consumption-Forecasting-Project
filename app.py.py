import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

# ----- CSS Styling -----
st.markdown("""
    <style>
    .stApp {
        background-color: #121212;
        color: #E0E0E0;
        font-family: 'Segoe UI', sans-serif;
    }
    .main-title {
        color: #FFA500;
        text-align: center;
        font-size: 36px;
        font-weight: bold;
        margin-bottom: 10px;
    }
    .block-container {
        padding: 2rem 2rem 2rem 2rem;
    }
    .stDataFrame {
        background-color: #1E1E1E;
        color: #FFFFFF;
    }
    .st-bd {
        background-color: #1E1E1E !important;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #1E1E1E;
        color: #FFFFFF;
    }
    .stTabs [data-baseweb="tab"]:hover {
        background-color: #333333;
        color: #FFA500;
    }
    .stTabs [aria-selected="true"] {
        background-color: #333333;
        color: #FFA500;
    }
    .css-1aumxhk {
        background-color: #333333;
        color: #FFFFFF;
    }
    .css-1d391kg {
        background-color: #1E1E1E;
    }
    .css-qrbaxs {
        color: #E0E0E0;
    }
    .css-ffhzg2 {
        background-color: #1E1E1E;
        color: #E0E0E0;
    }
    .css-1x8cf1d {
        color: #FFA500;
    }
    </style>
""", unsafe_allow_html=True)

# ----- Load Model and Data -----
@st.cache_resource
def load_model():
    return pickle.load(open("final_rf_model.pkl", "rb"))

@st.cache_data
def load_data():
    last_7 = pd.read_csv("daily_data_last_7.csv", header=None)
    data = pd.read_csv("dataset_daily.csv", header=0, index_col=0, parse_dates=True)
    return last_7, data

model1 = load_model()
daily_data_last_7, actual_data = load_data()

# ----- Main Title -----
st.markdown('<div class="main-title">⚡ Power Consumption Forecast App ⚡</div>', unsafe_allow_html=True)

# ----- Sidebar Inputs -----
st.sidebar.header("Forecast Settings")
days = st.sidebar.slider('Select number of days to forecast', min_value=1, max_value=60, value=7)

# ----- Forecast Calculation -----
with st.spinner('Forecasting in progress...'):
    z = daily_data_last_7[0].tail(7).to_numpy()
    for _ in range(days):
        r = z[-7:]
        r = np.array([r])
        pred = model1.predict(r)
        z = np.append(z, pred)

    future_pred = z[-days:]
    future_dates = pd.date_range(start='2018-08-04', periods=days, freq='D')
    future_df = pd.DataFrame({'Date': future_dates, 'Forecasted Power Consumption (MW)': future_pred})
    future_df.set_index('Date', inplace=True)

# ----- Main Layout -----
tab1, tab2, tab3, tab4 = st.tabs(["📊 Forecast Table", "📈 Forecast Line Chart", "🖼️ Matplotlib Graph", "📉 Actual vs Forecast"])

# ----- Forecast Table -----
with tab1:
    st.subheader(f"🔍 Forecasted Power Consumption for next {days} Days")
    st.dataframe(future_df.style.format({'Forecasted Power Consumption (MW)': '{:.2f}'}))

    # Download Button
    csv = future_df.to_csv().encode('utf-8')
    st.download_button("Download Forecast Data as CSV", data=csv, file_name='forecasted_power.csv', mime='text/csv')

# ----- Forecast Line Chart (Streamlit Native) -----
with tab2:
    st.subheader("📈 Interactive Forecast Line Chart (Streamlit)")
    st.line_chart(future_df)

# ----- Matplotlib Forecast Graph -----
with tab3:
    st.subheader("🖼️ Forecast Trend (Matplotlib Visual)")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(future_df.index, future_df['Forecasted Power Consumption (MW)'], color="#FFA500", marker='o', label='Forecast')
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Power Consumption (MW)', fontsize=12)
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    st.pyplot(fig)

# ----- Actual vs Forecast -----
with tab4:
    st.subheader("📉 Actual vs Forecast Comparison")
    fig2, ax2 = plt.subplots(figsize=(12, 6))
    ax2.plot(actual_data['Energy'][-365:].index, actual_data['Energy'][-365:].values, label='Actual (Last Year)', color='#1f77b4')
    ax2.plot(future_df.index, future_df['Forecasted Power Consumption (MW)'], label='Forecast', color='#FFA500', marker='o')
    ax2.set_xlabel('Date', fontsize=12)
    ax2.set_ylabel('Power Consumption (MW)', fontsize=12)
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    st.pyplot(fig2)
