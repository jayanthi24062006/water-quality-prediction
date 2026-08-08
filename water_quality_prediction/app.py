import os
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Get exact directory path where app.py is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# -------------------------------------------------------------
# PAGE CONFIGURATION
# -------------------------------------------------------------
st.set_page_config(
    page_title="AquaSense Pro | Water Quality AI",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# -------------------------------------------------------------
# CUSTOM CSS FOR MODERN BRIGHT / LIGHT THEME
# -------------------------------------------------------------
st.markdown("""
<style>
    /* Bright Clean Background */
    .stApp {
        background-color: #f8fafc;
        color: #0f172a;
    }
    
    /* Main Header Styling */
    .main-title {
        text-align: center;
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(90deg, #0284c7, #2563eb);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    
    .sub-title {
        text-align: center;
        font-size: 1.1rem;
        color: #64748b;
        margin-bottom: 2rem;
        font-weight: 500;
    }

    /* Clean Bright Cards */
    .bright-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03);
        margin-bottom: 1rem;
    }

    /* Tab Headers Customization */
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
        justify-content: center;
    }

    .stTabs [data-baseweb="tab"] {
        height: 48px;
        white-space: pre-wrap;
        background-color: #ffffff;
        border-radius: 8px;
        color: #475569;
        font-weight: 600;
        border: 1px solid #cbd5e1;
        padding: 0 20px;
    }

    .stTabs [aria-selected="true"] {
        background-color: #0284c7 !important;
        color: #ffffff !important;
        border-color: #0284c7 !important;
        box-shadow: 0 4px 12px rgba(2, 132, 199, 0.25);
    }

    /* Custom Predict Button */
    div.stButton > button:first-child {
        background: linear-gradient(135deg, #0284c7, #0284c7);
        color: white;
        font-size: 1.2rem;
        font-weight: 700;
        border-radius: 10px;
        border: none;
        padding: 14px 28px;
        box-shadow: 0 4px 14px rgba(2, 132, 199, 0.3);
        transition: all 0.3s ease;
        width: 100%;
    }
    
    div.stButton > button:first-child:hover {
        background: linear-gradient(135deg, #0369a1, #0284c7);
        box-shadow: 0 6px 20px rgba(2, 132, 199, 0.45);
        transform: translateY(-2px);
    }
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------
# LOAD OR AUTO-TRAIN MODEL & SCALER WITH DYNAMIC FILE PATHS
# -------------------------------------------------------------
@st.cache_resource
def load_assets():
    model_path = os.path.join(BASE_DIR, 'water_model.pkl')
    scaler_path = os.path.join(BASE_DIR, 'scaler.pkl')
    csv_path = os.path.join(BASE_DIR, 'water_potability.csv')
    
    try:
        # Try loading pre-trained .pkl files from exact app directory
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        return model, scaler
    except Exception:
        # Fallback: Train model automatically if .pkl files are missing
        from sklearn.preprocessing import StandardScaler
        from sklearn.ensemble import RandomForestClassifier

        if not os.path.exists(csv_path):
            st.error(f"❌ File not found at path: `{csv_path}`. Make sure `water_potability.csv` is uploaded to your GitHub repository inside the `water_quality_prediction` folder alongside `app.py`!")
            st.stop()

        df = pd.read_csv(csv_path)
        df['ph'] = df['ph'].fillna(df['ph'].median())
        df['Sulfate'] = df['Sulfate'].fillna(df['Sulfate'].median())
        df['Trihalomethanes'] = df['Trihalomethanes'].fillna(df['Trihalomethanes'].median())

        X = df.drop('Potability', axis=1)
        y = df['Potability']

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_scaled, y)

        return model, scaler

model, scaler = load_assets()

# Application Title Banner
st.markdown('<div class="main-title">💧 AquaSense AI</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Next-Generation Water Potability & Safety Analytics</div>', unsafe_allow_html=True)

# -------------------------------------------------------------
# 3 SLIDES TABS NAVIGATION
# -------------------------------------------------------------
tab1, tab2, tab3 = st.tabs(["📄 Abstract", "🎯 Objectives", "💻 Application Interface"])

# =============================================================
# SLIDE 1: ABSTRACT
# =============================================================
with tab1:
    st.markdown("""
    <div class="bright-card">
        <h2 style="color: #0284c7; margin-top:0;">📌 Project Abstract</h2>
        <p style="font-size: 1.1rem; line-height: 1.7; color: #334155;">
            Access to safe drinking water is a fundamental human requirement and a key component of effective public health protection. 
            This project provides an intelligent <b>Water Quality Prediction System</b> utilizing Supervised Machine Learning algorithms 
            (Random Forest & XGBoost) to classify water safety in real time.
        </p>
        <p style="font-size: 1.1rem; line-height: 1.7; color: #334155;">
            By processing key physical and chemical characteristics—including <b>pH balance, Chloramines, Total Dissolved Solids (TDS), Turbidity, and Sulfates</b>—the system automatically determines whether water is safe for human consumption (Potable) or poses health hazards (Non-Potable).
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="bright-card" style="text-align: center;">
            <h3 style="color: #0284c7; font-size: 2rem; margin-bottom: 5px;">📊 3,276</h3>
            <p style="color: #64748b; font-weight: 500;">Water Samples Tested</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="bright-card" style="text-align: center;">
            <h3 style="color: #0284c7; font-size: 2rem; margin-bottom: 5px;">🧪 9 Features</h3>
            <p style="color: #64748b; font-weight: 500;">Physicochemical Parameters</p>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="bright-card" style="text-align: center;">
            <h3 style="color: #0284c7; font-size: 2rem; margin-bottom: 5px;">🤖 ML Model</h3>
            <p style="color: #64748b; font-weight: 500;">Random Forest Classifier</p>
        </div>
        """, unsafe_allow_html=True)

# =============================================================
# SLIDE 2: OBJECTIVES
# =============================================================
with tab2:
    st.markdown("""
    <div class="bright-card">
        <h2 style="color: #0284c7; margin-top:0;">🎯 Core Objectives</h2>
    </div>
    """, unsafe_allow_html=True)
    
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""
        <div class="bright-card" style="height: 230px;">
            <h3 style="color: #0284c7;">1. Data Analysis</h3>
            <p style="color: #334155; line-height: 1.6;">Analyze chemical parameters against World Health Organization (WHO) safety standard thresholds.</p>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="bright-card" style="height: 230px;">
            <h3 style="color: #0284c7;">2. Automated ML</h3>
            <p style="color: #334155; line-height: 1.6;">Train ensemble machine learning models to predict potability instantly without lab delays.</p>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown("""
        <div class="bright-card" style="height: 230px;">
            <h3 style="color: #0284c7;">3. User Dashboard</h3>
            <p style="color: #334155; line-height: 1.6;">Deploy an intuitive dashboard featuring front-side parameter inputs and downside prediction outcomes.</p>
        </div>
        """, unsafe_allow_html=True)

# =============================================================
# SLIDE 3: APPLICATION INTERFACE (Features Front, Results Downside)
# =============================================================
with tab3:
    st.markdown("### 🎛️ Input Water Quality Parameters")
    
    # FRONT SECTION: Features Grid Inputs
    with st.container():
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown('<div class="bright-card">', unsafe_allow_html=True)
            st.markdown("<h4 style='color: #0284c7;'>🧪 Chemical Metrics</h4>", unsafe_allow_html=True)
            ph = st.slider("pH Level", 0.0, 14.0, 7.2, help="WHO standard range: 6.5 - 8.5")
            hardness = st.number_input("Hardness (mg/L)", 0.0, 500.0, 204.0)
            solids = st.number_input("Solids - TDS (ppm)", 0.0, 50000.0, 20791.0)
            st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="bright-card">', unsafe_allow_html=True)
            st.markdown("<h4 style='color: #0284c7;'>🧼 Disinfectants & Minerals</h4>", unsafe_allow_html=True)
            chloramines = st.slider("Chloramines (ppm)", 0.0, 15.0, 7.3)
            sulfate = st.number_input("Sulfate (mg/L)", 0.0, 500.0, 368.0)
            conductivity = st.number_input("Conductivity (μS/cm)", 0.0, 1000.0, 564.0)
            st.markdown('</div>', unsafe_allow_html=True)

        with col3:
            st.markdown('<div class="bright-card">', unsafe_allow_html=True)
            st.markdown("<h4 style='color: #0284c7;'>☣️ Organic Components</h4>", unsafe_allow_html=True)
            organic_carbon = st.slider("Organic Carbon (ppm)", 0.0, 30.0, 10.3)
            trihalomethanes = st.number_input("Trihalomethanes (μg/L)", 0.0, 200.0, 86.0)
            turbidity = st.slider("Turbidity (NTU)", 0.0, 10.0, 2.9)
            st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    predict_clicked = st.button("🚀 ANALYZE WATER SAMPLE")

    # DOWNSIDE SECTION: Prediction Results
    st.markdown("---")
    st.markdown("### 📊 Prediction Analysis (Result)")

    if predict_clicked:
        input_df = pd.DataFrame([{
            'ph': ph,
            'Hardness': hardness,
            'Solids': solids,
            'Chloramines': chloramines,
            'Sulfate': sulfate,
            'Conductivity': conductivity,
            'Organic_carbon': organic_carbon,
            'Trihalomethanes': trihalomethanes,
            'Turbidity': turbidity
        }])
        
        scaled_data = scaler.transform(input_df)
        
        prediction = model.predict(scaled_data)[0]
        probs = model.predict_proba(scaled_data)[0]

        res_col1, res_col2 = st.columns([1.5, 1])

        with res_col1:
            if prediction == 1:
                st.markdown("""
                <div style="background-color: #ecfdf5; border: 2px solid #10b981; border-radius: 12px; padding: 22px; text-align: center;">
                    <h1 style="color: #047857; margin:0;">✅ WATER IS POTABLE</h1>
                    <p style="font-size: 1.15rem; color: #065f46; margin-top: 8px; font-weight: 500;">This sample is verified safe for drinking and human consumption.</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div style="background-color: #fef2f2; border: 2px solid #ef4444; border-radius: 12px; padding: 22px; text-align: center;">
                    <h1 style="color: #b91c1c; margin:0;">⚠️ WATER IS NOT POTABLE</h1>
                    <p style="font-size: 1.15rem; color: #991b1b; margin-top: 8px; font-weight: 500;">Warning: Unsafe parameter levels detected. Purification required.</p>
                </div>
                """, unsafe_allow_html=True)

        with res_col2:
            st.markdown('<div class="bright-card" style="text-align: center;">', unsafe_allow_html=True)
            st.markdown("<h4 style='color: #475569;'>Model Confidence</h4>", unsafe_allow_html=True)
            confidence = probs[1] if prediction == 1 else probs[0]
            st.progress(float(confidence))
            st.markdown(f"<h2 style='color: #0284c7; margin: 5px 0 0 0;'>{confidence*100:.1f}%</h2>", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.info("👆 Adjust the parameters above and click **'ANALYZE WATER SAMPLE'** to display results here.")
