import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import base64

# Set page config
st.set_page_config(page_title="Air Quality Forecasting 🌿", page_icon="🌍", layout="wide")

# Load data and models
df = pd.read_csv('merged_data.csv')
model_knn = joblib.load('model_knn.pkl')
model_lr = joblib.load('model_lr.pkl')
scaler = joblib.load('scaler.pkl')

# UI styling
st.markdown("""
<style>
body {
    background-color: #f4fcfb;
}
[data-testid="stSidebar"] {
    background-color: #e0f7fa;
    padding: 1rem;
}
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2 {
    color: #00796b;
}
.stMetric {
    border: 2px solid #2e8b57;
    background-color: #e0f2e9;
    border-radius: 8px;
    padding: 10px;
}
.stButton>button {
    background-color: #2e8b57;
    color: white;
    font-weight: 600;
    border-radius: 6px;
}
.non-home {
    background-color: #f4fcfb;
    padding: 2rem;
    border-radius: 15px;
}
</style>
""", unsafe_allow_html=True)

# Sidebar
stations = ['Dongsi', 'Changping', 'Huairou', 'Aotizhongxin']
st.sidebar.title("📍 Station Selection")
selected_station = st.sidebar.selectbox("Choose a Station", stations)
st.sidebar.markdown("---")

# Filter data by station
filtered_df = df[df['station'] == selected_station]

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["🏠 Home", "📊 Data Overview", "📈 EDA", "⚙️ Predict PM2.5"])

# --- Home ---
with tab1:
    st.markdown('<div class="non-home">', unsafe_allow_html=True)
    st.title(f"🌿 Air Quality Forecasting - {selected_station}")
    st.markdown(f"""
    Welcome to the **Air Quality Prediction Platform** for **{selected_station}**!  
    Navigate through the tabs to explore, analyze, and predict air quality. 🌍

    ### 🏙️ Station Info:
    - **Dongsi**: Dense urban core, traffic-heavy.
    - **Changping**: Suburban district, moderately polluted.
    - **Huairou**: Clean rural area, great for contrast.
    - **Aotizhongxin**: Near Olympic Sports Center, moderate traffic.
    """)
    col1, col2, col3 = st.columns(3)
    col1.metric("Avg PM2.5", f"{filtered_df['PM2.5'].mean():.2f} µg/m³")
    col2.metric("Total Records", f"{filtered_df.shape[0]}")
    col3.metric("Missing Data (%)", f"{round(filtered_df.isnull().mean().mean() * 100, 2)}%")
    st.markdown('</div>', unsafe_allow_html=True)

# --- Data Overview ---
with tab2:
    st.markdown('<div class="non-home">', unsafe_allow_html=True)
    st.title(f"📊 Data Overview - {selected_station}")
    col1, col2, col3 = st.columns(3)
    col1.metric("Rows", f"{filtered_df.shape[0]}")
    col2.metric("Columns", f"{filtered_df.shape[1]}")
    col3.metric("Missing %", f"{round(filtered_df.isnull().mean().mean() * 100, 2)}%")

    with st.expander("📋 Sample Data Table"):
        st.dataframe(filtered_df.head(20), use_container_width=True)

    with st.expander("❗ Missing Values"):
        missing = filtered_df.isnull().sum()
        st.dataframe(missing[missing > 0], use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# --- EDA ---
with tab3:
    st.markdown('<div class="non-home">', unsafe_allow_html=True)
    st.title(f"📈 Exploratory Data Analysis - {selected_station}")
    chart_type = st.selectbox("Choose a visualization", ["PM2.5 Distribution", "Correlation Heatmap", "Pairplot"])

    if chart_type == "PM2.5 Distribution":
        fig, ax = plt.subplots()
        sns.histplot(filtered_df['PM2.5'].dropna(), bins=50, kde=True, color='seagreen', ax=ax)
        ax.set_title("Distribution of PM2.5")
        st.pyplot(fig)

    elif chart_type == "Correlation Heatmap":
        corr = filtered_df.select_dtypes(include=['number']).dropna().corr()
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.heatmap(corr, annot=True, cmap='YlGnBu', fmt=".2f", ax=ax)
        ax.set_title("Correlation Heatmap")
        st.pyplot(fig)

    elif chart_type == "Pairplot":
        st.subheader("🔍 Custom Scatter and Histogram Viewer")
        numeric_cols = filtered_df.select_dtypes(include=['number']).columns.tolist()

        plot_type = st.radio("Choose plot type", ["Scatter Plot", "Histogram"], horizontal=True)

        if plot_type == "Scatter Plot":
            x_axis = st.selectbox("X-axis", numeric_cols, index=0)
            y_axis = st.selectbox("Y-axis", numeric_cols, index=1)
            fig, ax = plt.subplots()
            sns.scatterplot(data=filtered_df, x=x_axis, y=y_axis, color='seagreen', s=20, ax=ax)
            ax.set_title(f"{y_axis} vs {x_axis}")
            st.pyplot(fig)

        elif plot_type == "Histogram":
            hist_col = st.selectbox("Feature", numeric_cols, index=0)
            fig, ax = plt.subplots()
            sns.histplot(filtered_df[hist_col].dropna(), bins=30, kde=True, color='seagreen', ax=ax)
            ax.set_title(f"Distribution of {hist_col}")
            st.pyplot(fig)
    st.markdown('</div>', unsafe_allow_html=True)

# --- Prediction ---
with tab4:
    st.markdown('<div class="non-home">', unsafe_allow_html=True)
    st.title(f"⚙️ PM2.5 Prediction - {selected_station}")
    st.markdown("Use the sliders to set values for pollutants and weather, then run predictions.")

    model_option = st.selectbox("Choose a model", ["KNN", "Linear Regression"])

    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            pm10 = st.slider("PM10 (µg/m³)", 0, 1000, 100)
            so2 = st.slider("SO2 (µg/m³)", 0, 500, 15)
            no2 = st.slider("NO2 (µg/m³)", 0, 500, 20)
            co = st.slider("CO (mg/m³)", 0.0, 5.0, 1.0)
        with col2:
            o3 = st.slider("O3 (µg/m³)", 0, 500, 30)
            wspd = st.slider("Wind Speed (m/s)", 0, 20, 5)
            rain = st.slider("Rainfall (mm)", 0, 10, 0)
            temp = st.slider("Temperature (°C)", -20, 40, 15)
        with col3:
            dewp = st.slider("Dew Point (°C)", -20, 40, 5)
            pre = st.slider("Pressure (hPa)", 900, 1100, 1010)
            month = st.slider("Month", 1, 12, 6)
            hour = st.slider("Hour", 0, 23, 12)

        col_submit1, col_submit2 = st.columns(2)
        submit_predict = col_submit1.form_submit_button("🚀 Predict")
        submit_compare = col_submit2.form_submit_button("📊 Compare Models")

    input_data = [[pm10, so2, no2, co, o3, wspd, rain, temp, dewp, pre, month, hour]]

    if submit_predict:
        try:
            input_scaled = scaler.transform(input_data)
            model = model_knn if model_option == "KNN" else model_lr
            prediction = model.predict(input_scaled)
            st.success(f"🎯 Predicted PM2.5 using {model_option}: **{prediction[0]:.2f} µg/m³**")
        except Exception as e:
            st.error(f"Prediction failed: {e}")

    if submit_compare:
        try:
            input_scaled = scaler.transform(input_data)
            knn_pred = model_knn.predict(input_scaled)[0]
            lr_pred = model_lr.predict(input_scaled)[0]
            comp_df = pd.DataFrame({
                "Model": ["KNN", "Linear Regression"],
                "Predicted PM2.5": [knn_pred, lr_pred]
            }).round(2)
            st.table(comp_df)
        except Exception as e:
            st.error(f"Comparison failed: {e}")
    st.markdown('</div>', unsafe_allow_html=True)
