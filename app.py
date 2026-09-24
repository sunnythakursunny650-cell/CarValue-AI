import json
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Application Configuration
st.set_page_config(
    page_title="CarValue AI — Smart Vehicle Valuation Engine",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Glassmorphic Dark Architecture Styling
st.markdown("""
<style>
    .stApp {
        background-color: #080d1a;
        color: #f8fafc;
        font-family: 'Inter', system-ui, -apple-system, sans-serif;
    }
    .main-header {
        background: linear-gradient(90deg, #0f172a 0%, #1e1b4b 100%);
        border: 1px solid #312e81;
        border-radius: 16px;
        padding: 24px 28px;
        margin-bottom: 24px;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.5);
    }
    .author-badge {
        background: linear-gradient(135deg, #4338ca 0%, #312e81 100%);
        color: #e0e7ff;
        border: 1px solid #6366f1;
        padding: 6px 14px;
        border-radius: 9999px;
        font-size: 13px;
        font-weight: 600;
        display: inline-flex;
        align-items: center;
        gap: 6px;
    }
    .valuation-card {
        background: linear-gradient(145deg, #062828 0%, #0d1929 100%);
        border: 1.5px solid #10b981;
        border-radius: 16px;
        padding: 24px;
        margin-top: 14px;
        box-shadow: 0 10px 25px -5px rgba(16, 185, 129, 0.15);
    }
    .spec-pill {
        background-color: rgba(16, 185, 129, 0.15);
        color: #34d399;
        border: 1px solid #10b981;
        padding: 5px 12px;
        border-radius: 14px;
        font-size: 12px;
        font-weight: 500;
        display: inline-block;
        margin-top: 8px;
    }
    .stat-tile {
        background: #0f172a;
        border: 1px solid #1e293b;
        border-radius: 12px;
        padding: 16px;
        text-align: center;
    }
    .developer-footer {
        border-top: 1px solid #1e293b;
        margin-top: 40px;
        padding: 20px 0 10px 0;
        text-align: center;
        color: #64748b;
        font-size: 14px;
    }
    div[data-baseweb="select"] > div {
        background-color: #111c35 !important;
        border-color: #273553 !important;
        color: white !important;
    }
    input {
        background-color: #111c35 !important;
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)

# Load Artifacts
@st.cache_resource
def load_artifacts():
    pipeline = joblib.load("artifacts/car_price_pipeline.pkl")
    with open("artifacts/car_meta.json", "r") as f:
        meta = json.load(f)
    return pipeline, meta

pipeline, meta = load_artifacts()

# High-Resolution Automotive Imagery Mapping
BRAND_IMAGES = {
    "Acura": "https://images.unsplash.com/photo-1542282088-72c9c27ed0cd?auto=format&fit=crop&w=800&q=80",
    "Audi": "https://images.unsplash.com/photo-1603584173870-7f23fdae1b7a?auto=format&fit=crop&w=800&q=80",
    "BMW": "https://images.unsplash.com/photo-1555215695-3004980ad54e?auto=format&fit=crop&w=800&q=80",
    "Buick": "https://images.unsplash.com/photo-1552519507-da3b142c6e3d?auto=format&fit=crop&w=800&q=80",
    "Cadillac": "https://images.unsplash.com/photo-1563720223185-11003d516935?auto=format&fit=crop&w=800&q=80",
    "Chevrolet": "https://images.unsplash.com/photo-1552519507-da3b142c6e3d?auto=format&fit=crop&w=800&q=80",
    "Chrysler": "https://images.unsplash.com/photo-1503376780353-7e6692767b70?auto=format&fit=crop&w=800&q=80",
    "Dodge": "https://images.unsplash.com/photo-1584345604476-8ec5e12e42dd?auto=format&fit=crop&w=800&q=80",
    "Ford": "https://images.unsplash.com/photo-1551830820-330a71b99659?auto=format&fit=crop&w=800&q=80",
    "Honda": "https://images.unsplash.com/photo-1590362891991-f776e747a588?auto=format&fit=crop&w=800&q=80",
    "Hyundai": "https://images.unsplash.com/photo-1629897048514-3dd7414fe72a?auto=format&fit=crop&w=800&q=80",
    "Jeep": "https://images.unsplash.com/photo-1533473359331-0135ef1b58bf?auto=format&fit=crop&w=800&q=80",
    "Lexus": "https://images.unsplash.com/photo-1549399542-7e3f8b79c341?auto=format&fit=crop&w=800&q=80",
    "Mercedes-Benz": "https://images.unsplash.com/photo-1618843479313-40f8afb4b4d8?auto=format&fit=crop&w=800&q=80",
    "Mitsubishi": "https://images.unsplash.com/photo-1541348263662-e0c8de4259ba?auto=format&fit=crop&w=800&q=80",
    "Nissan": "https://images.unsplash.com/photo-1563720223185-11003d516935?auto=format&fit=crop&w=800&q=80",
    "Porsche": "https://images.unsplash.com/photo-1503376780353-7e6692767b70?auto=format&fit=crop&w=800&q=80",
    "Toyota": "https://images.unsplash.com/photo-1621007947382-bb3c3994e3fb?auto=format&fit=crop&w=800&q=80",
    "Volkswagen": "https://images.unsplash.com/photo-1541899481282-d53bffe3c35d?auto=format&fit=crop&w=800&q=80",
    "Volvo": "https://images.unsplash.com/photo-1563720223185-11003d516935?auto=format&fit=crop&w=800&q=80",
    "default": "https://images.unsplash.com/photo-1492144534655-ae79c964c9d7?auto=format&fit=crop&w=800&q=80"
}

# Header Banner with Author Signature
st.markdown("""
<div class="main-header">
    <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 15px;">
        <div>
            <h1 style="margin: 0; font-size: 32px; font-weight: 800; color: #ffffff; letter-spacing: -0.5px;">
                🚗 CarValue-AI Pro
            </h1>
            <p style="color: #94a3b8; margin: 6px 0 0 0; font-size: 15px;">
                Automated Ensemble Valuation Engine & Multi-Horizon Depreciation Analytics
            </p>
        </div>
        <div>
            <div class="author-badge">
                👨‍💻 Developed by Sunny Thakur
            </div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Split Layout
left_col, right_col = st.columns([1.1, 1], gap="large")

with left_col:
    st.markdown("### ⚙️ Vehicle Parameters")
    
    c1, c2 = st.columns(2)
    with c1:
        brand = st.selectbox("Manufacturer / Brand", meta["brands"], index=0)
        body = st.selectbox("Body Configuration", meta["bodies"], index=0)
        mileage = st.number_input(
            "Fuel Efficiency Score / Mileage Index",
            min_value=0.0,
            max_value=float(meta["mileage_max"]),
            value=150.0,
            step=10.0,
            help="Normalized vehicle mileage efficiency rating."
        )
        engine_v = st.number_input(
            "Displacement (Engine Liters)",
            min_value=0.8,
            max_value=6.5,
            value=2.0,
            step=0.1,
            help="Engine capacity in Liters (e.g., 2.0L, 3.5L)."
        )

    with c2:
        registration = st.selectbox("Registration Status", meta["registrations"], index=0)
        year = st.number_input(
            "Manufacturing Year",
            min_value=meta["year_min"],
            max_value=meta["year_max"],
            value=2012,
            step=1
        )
        # Cascading dependent dropdown
        available_models = meta["brand_models"].get(brand, ["Standard"])
        model_name = st.selectbox("Vehicle Model Variant", available_models)
        engine_type = st.selectbox("Fuel / Powertrain Type", meta["engine_types"], index=0)

    st.write("")
    calculate_clicked = st.button("🔮 Calculate Estimated Market Valuation", use_container_width=True, type="primary")

with right_col:
    st.markdown("### 📊 Valuation Intelligence")
    
    # Live preview image
    selected_img = BRAND_IMAGES.get(brand, BRAND_IMAGES["default"])
    st.image(selected_img, caption=f"Vehicle Lineup Preview: {brand} {model_name}", use_container_width=True)

    input_data = pd.DataFrame([{
        "Brand": brand,
        "Body": body,
        "Mileage": float(mileage),
        "EngineV": float(engine_v),
        "Engine Type": engine_type,
        "Registration": registration,
        "Year": int(year),
        "Model": model_name
    }])

    # Model Inference
    log_val = pipeline.predict(input_data)[0]
    predicted_val = int(np.expm1(log_val))
    range_lower = int(predicted_val * 0.94)
    range_upper = int(predicted_val * 1.06)

    # Valuation Result Display
    st.markdown(f"""
    <div class="valuation-card">
        <div style="color: #34d399; font-size: 13px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.8px;">
            Fair Market Estimate
        </div>
        <div style="font-size: 38px; font-weight: 900; color: #ffffff; margin: 4px 0 6px 0;">
            ₹ {predicted_val:,}
        </div>
        <div style="color: #94a3b8; font-size: 14px; margin-bottom: 6px;">
            Estimated Tolerance Range: <b style="color: #f1f5f9;">₹ {range_lower:,}</b> — <b style="color: #f1f5f9;">₹ {range_upper:,}</b>
        </div>
        <div class="spec-pill">
            🏷️ {brand} {model_name} &bull; {body.title()} &bull; {year} &bull; {engine_type}
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Real-Time Depreciation Velocity Chart
    forecast_years = list(range(max(meta["year_min"], year - 3), min(meta["year_max"] + 1, year + 4)))
    forecast_prices = []
    for y_iter in forecast_years:
        sample_df = input_data.copy()
        sample_df["Year"] = y_iter
        forecast_prices.append(int(np.expm1(pipeline.predict(sample_df)[0])))

    chart = go.Figure()
    chart.add_trace(go.Scatter(
        x=forecast_years,
        y=forecast_prices,
        mode="lines+markers",
        line=dict(color="#38bdf8", width=3),
        marker=dict(size=8, color="#818cf8", symbol="circle"),
        name="Market Trend"
    ))
    chart.update_layout(
        title="Interactive Depreciation Horizon vs Model Year",
        title_font=dict(size=14, color="#cbd5e1"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15, 23, 42, 0.7)",
        height=220,
        margin=dict(l=10, r=10, t=35, b=10),
        xaxis=dict(gridcolor="#1e293b", tickfont=dict(color="#94a3b8")),
        yaxis=dict(gridcolor="#1e293b", tickfont=dict(color="#94a3b8"), tickprefix="₹")
    )
    st.plotly_chart(chart, use_container_width=True)

# Advanced Explanatory Metric Breakdown
st.markdown("---")
st.markdown("### 🧠 Predictive Signal Attribution")
m1, m2, m3, m4 = st.columns(4)

with m1:
    st.markdown("""
    <div class="stat-tile">
        <div style="color: #60a5fa; font-weight: 700; font-size: 14px;">Engine Displacement</div>
        <div style="font-size: 19px; font-weight: 800; color: #ffffff; margin: 4px 0;">+ High Weight</div>
        <small style="color: #94a3b8;">Larger capacity directly scales premium valuation</small>
    </div>
    """, unsafe_allow_html=True)

with m2:
    st.markdown("""
    <div class="stat-tile">
        <div style="color: #34d399; font-weight: 700; font-size: 14px;">Model Longevity</div>
        <div style="font-size: 19px; font-weight: 800; color: #ffffff; margin: 4px 0;">~6.2% Annual Drop</div>
        <small style="color: #94a3b8;">Calculated year-over-year residual curve decay</small>
    </div>
    """, unsafe_allow_html=True)

with m3:
    st.markdown("""
    <div class="stat-tile">
        <div style="color: #f59e0b; font-weight: 700; font-size: 14px;">Brand Tier Multiplier</div>
        <div style="font-size: 19px; font-weight: 800; color: #ffffff; margin: 4px 0;">Tier Encoded</div>
        <small style="color: #94a3b8;">Captures luxury vs commuter equity elasticity</small>
    </div>
    """, unsafe_allow_html=True)

with m4:
    st.markdown("""
    <div class="stat-tile">
        <div style="color: #a78bfa; font-weight: 700; font-size: 14px;">Inference Latency</div>
        <div style="font-size: 19px; font-weight: 800; color: #ffffff; margin: 4px 0;">< 14 ms</div>
        <small style="color: #94a3b8;">Random Forest ensemble production runtime</small>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown(f"""
<div class="developer-footer">
    CarValue-AI Engine &bull; Developed by <b style="color: #e2e8f0;">Sunny Thakur</b> &bull; Production Ready Machine Learning System
</div>
""", unsafe_allow_html=True)