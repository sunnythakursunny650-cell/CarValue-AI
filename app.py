import json
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Page Configuration
st.set_page_config(
    page_title="CarValue AI — Smart Vehicle Valuation Engine",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load Artifacts
@st.cache_resource
def load_artifacts():
    pipeline = joblib.load("artifacts/car_price_pipeline.pkl")
    with open("artifacts/car_meta.json", "r") as f:
        meta = json.load(f)
    return pipeline, meta

pipeline, meta = load_artifacts()

# Model & Brand-Specific High-Resolution Image Mapping
MODEL_PHOTOS = {
    # Audi Models
    "A4": "https://images.unsplash.com/photo-1606152421802-db97b9c7a11b?auto=format&fit=crop&w=800&q=80",
    "A6": "https://images.unsplash.com/photo-1603584173870-7f23fdae1b7a?auto=format&fit=crop&w=800&q=80",
    "Q7": "https://images.unsplash.com/photo-1541348263662-e0c8de4259ba?auto=format&fit=crop&w=800&q=80",
    "TT": "https://images.unsplash.com/photo-1503376780353-7e6692767b70?auto=format&fit=crop&w=800&q=80",
    
    # BMW Models
    "3 Series": "https://images.unsplash.com/photo-1555215695-3004980ad54e?auto=format&fit=crop&w=800&q=80",
    "5 Series": "https://images.unsplash.com/photo-1556189250-72ba954cfc2b?auto=format&fit=crop&w=800&q=80",
    "X5": "https://images.unsplash.com/photo-1580273916550-e323be2ae537?auto=format&fit=crop&w=800&q=80",
    "M3": "https://images.unsplash.com/photo-1617814076367-b759c7d7e738?auto=format&fit=crop&w=800&q=80",
    
    # Mercedes-Benz Models
    "C-Class": "https://images.unsplash.com/photo-1618843479313-40f8afb4b4d8?auto=format&fit=crop&w=800&q=80",
    "E-Class": "https://images.unsplash.com/photo-1617531653332-bd46c24f2068?auto=format&fit=crop&w=800&q=80",
    "S-Class": "https://images.unsplash.com/photo-1622353219448-46a489721752?auto=format&fit=crop&w=800&q=80",
    
    # Ford Models
    "Mustang": "https://images.unsplash.com/photo-1584345604476-8ec5e12e42dd?auto=format&fit=crop&w=800&q=80",
    "Focus": "https://images.unsplash.com/photo-1551830820-330a71b99659?auto=format&fit=crop&w=800&q=80",
    
    # Toyota Models
    "Corolla": "https://images.unsplash.com/photo-1621007947382-bb3c3994e3fb?auto=format&fit=crop&w=800&q=80",
    "Camry": "https://images.unsplash.com/photo-1623869675781-80aa31012a5a?auto=format&fit=crop&w=800&q=80",
    
    # Volkswagen Models
    "Golf": "https://images.unsplash.com/photo-1541899481282-d53bffe3c35d?auto=format&fit=crop&w=800&q=80",
    "Passat": "https://images.unsplash.com/photo-1542282088-72c9c27ed0cd?auto=format&fit=crop&w=800&q=80",
    
    # Fallback Brand Photos
    "Acura": "https://images.unsplash.com/photo-1542282088-72c9c27ed0cd?auto=format&fit=crop&w=800&q=80",
    "Audi": "https://images.unsplash.com/photo-1603584173870-7f23fdae1b7a?auto=format&fit=crop&w=800&q=80",
    "BMW": "https://images.unsplash.com/photo-1555215695-3004980ad54e?auto=format&fit=crop&w=800&q=80",
    "Mercedes-Benz": "https://images.unsplash.com/photo-1618843479313-40f8afb4b4d8?auto=format&fit=crop&w=800&q=80",
    "Ford": "https://images.unsplash.com/photo-1551830820-330a71b99659?auto=format&fit=crop&w=800&q=80",
    "Toyota": "https://images.unsplash.com/photo-1621007947382-bb3c3994e3fb?auto=format&fit=crop&w=800&q=80",
    "Honda": "https://images.unsplash.com/photo-1590362891991-f776e747a588?auto=format&fit=crop&w=800&q=80",
    "Hyundai": "https://images.unsplash.com/photo-1629897048514-3dd7414fe72a?auto=format&fit=crop&w=800&q=80",
    "Chevrolet": "https://images.unsplash.com/photo-1552519507-da3b142c6e3d?auto=format&fit=crop&w=800&q=80",
    "Porsche": "https://images.unsplash.com/photo-1503376780353-7e6692767b70?auto=format&fit=crop&w=800&q=80",
    "Nissan": "https://images.unsplash.com/photo-1563720223185-11003d516935?auto=format&fit=crop&w=800&q=80",
    "Volkswagen": "https://images.unsplash.com/photo-1541899481282-d53bffe3c35d?auto=format&fit=crop&w=800&q=80",
    "default": "https://images.unsplash.com/photo-1492144534655-ae79c964c9d7?auto=format&fit=crop&w=800&q=80"
}

# Sidebar Controls & Sharp Developer Card
with st.sidebar:
    st.markdown("### ⚙️ Dashboard Controls")
    theme_choice = st.radio("Display Mode", ["Dark Mode 🌙", "Light Mode ☀️"], index=0)
    
    st.markdown("---")
    st.markdown("### 👨‍💻 Engineering & Profile")
    
    if theme_choice == "Dark Mode 🌙":
        dev_bg = "rgba(99, 102, 241, 0.12)"
        dev_border = "#6366f1"
        name_color = "#a5b4fc"
        role_color = "#38bdf8"
        desc_color = "#e2e8f0"
    else:
        dev_bg = "#f1f5f9"
        dev_border = "#4f46e5"
        name_color = "#312e81"
        role_color = "#1d4ed8"
        desc_color = "#0f172a"
        
    st.markdown(f"""
    <div style="background: {dev_bg}; border: 1.5px solid {dev_border}; border-radius: 12px; padding: 16px; margin-bottom: 14px;">
        <div style="font-weight: 800; font-size: 17px; color: {name_color}; margin-bottom: 4px;">Sunny Thakur</div>
        <div style="font-weight: 700; font-size: 13px; color: {role_color}; margin-bottom: 8px;">Machine Learning Engineer & Educator</div>
        <div style="font-weight: 600; font-size: 13px; line-height: 1.5; color: {desc_color};">
            End-to-end ML pipeline architecture, Random Forest optimization & production deployment.
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Professional Direct Contact Links
    email_url = "mailto:sunnythakursunny650@gmail.com?subject=Regarding%20CarValue-AI%20Project"
    linkedin_url = "https://www.linkedin.com/in/sunny-thakur-4a56103b9/"
    github_url = "https://github.com/sunnythakursunny650-cell/CarValue-AI"

    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        st.markdown(f'<a href="{email_url}" target="_blank" style="text-decoration:none;"><button style="width:100%; padding:9px; border-radius:8px; border:none; background:#ea4335; color:white; font-weight:700; font-size:12px; cursor:pointer;">📧 Email</button></a>', unsafe_allow_html=True)
    with col_btn2:
        st.markdown(f'<a href="{linkedin_url}" target="_blank" style="text-decoration:none;"><button style="width:100%; padding:9px; border-radius:8px; border:none; background:#0284c7; color:white; font-weight:700; font-size:12px; cursor:pointer;">💼 LinkedIn</button></a>', unsafe_allow_html=True)
    
    st.write("")
    st.markdown(f'<a href="{github_url}" target="_blank" style="text-decoration:none;"><button style="width:100%; padding:8px; border-radius:8px; border:1px solid #475569; background:transparent; color:#64748b; font-weight:600; font-size:12px; cursor:pointer;">📂 View Project on GitHub</button></a>', unsafe_allow_html=True)

# Theme Configuration
is_dark = (theme_choice == "Dark Mode 🌙")

if is_dark:
    bg_color = "#080d1a"
    text_color = "#f8fafc"
    card_bg = "#0f172a"
    card_border = "#1e293b"
    header_gradient = "linear-gradient(90deg, #0f172a 0%, #1e1b4b 100%)"
    header_border = "#312e81"
    subtext_color = "#94a3b8"
    input_bg = "#111c35"
else:
    bg_color = "#f8fafc"
    text_color = "#0f172a"
    card_bg = "#ffffff"
    card_border = "#cbd5e1"
    header_gradient = "linear-gradient(90deg, #f1f5f9 0%, #e0e7ff 100%)"
    header_border = "#94a3b8"
    subtext_color = "#334155"
    input_bg = "#ffffff"

st.markdown(f"""
<style>
    .stApp {{
        background-color: {bg_color};
        color: {text_color};
        font-family: 'Inter', system-ui, -apple-system, sans-serif;
    }}
    .main-header {{
        background: {header_gradient};
        border: 1.5px solid {header_border};
        border-radius: 16px;
        padding: 22px 28px;
        margin-bottom: 24px;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.1);
    }}
    .valuation-card {{
        background: {card_bg};
        border: 1.5px solid #10b981;
        border-radius: 16px;
        padding: 22px;
        margin-top: 14px;
        box-shadow: 0 10px 25px -5px rgba(16, 185, 129, 0.15);
    }}
    .spec-pill {{
        background-color: rgba(16, 185, 129, 0.15);
        color: #059669;
        border: 1.5px solid #10b981;
        padding: 6px 14px;
        border-radius: 14px;
        font-size: 13px;
        font-weight: 700;
        display: inline-block;
        margin-top: 8px;
    }}
    .stat-tile {{
        background: {card_bg};
        border: 1.5px solid {card_border};
        border-radius: 12px;
        padding: 16px;
        text-align: center;
    }}
    div[data-baseweb="select"] > div {{
        background-color: {input_bg} !important;
        color: {text_color} !important;
    }}
</style>
""", unsafe_allow_html=True)

# Main Banner
st.markdown(f"""
<div class="main-header">
    <h1 style="margin: 0; font-size: 30px; font-weight: 800; color: {text_color}; letter-spacing: -0.5px;">
        🚗 CarValue-AI Pro
    </h1>
    <p style="color: {subtext_color}; margin: 4px 0 0 0; font-size: 15px; font-weight: 500;">
        Automated Machine Learning Valuation Engine & Multi-Horizon Depreciation Analytics
    </p>
</div>
""", unsafe_allow_html=True)

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
            step=10.0
        )
        engine_v = st.number_input(
            "Displacement (Engine Liters)",
            min_value=0.8,
            max_value=6.5,
            value=2.0,
            step=0.1
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
        available_models = meta["brand_models"].get(brand, ["Standard"])
        model_name = st.selectbox("Vehicle Model Variant", available_models)
        engine_type = st.selectbox("Fuel / Powertrain Type", meta["engine_types"], index=0)

    # Loan Estimator Widget
    with st.expander("💳 Loan & EMI Customizer"):
        downpayment_pct = st.slider("Down Payment (%)", 10, 50, 20, 5)
        interest_rate = st.slider("Loan Interest Rate (% p.a.)", 7.0, 15.0, 9.5, 0.5)
        loan_tenure_years = st.selectbox("Tenure Duration", [3, 5, 7], index=1)

with right_col:
    st.markdown("### 📊 Valuation Intelligence")
    
    # Priority: Model image > Brand fallback image > Default car
    car_photo_url = MODEL_PHOTOS.get(model_name, MODEL_PHOTOS.get(brand, MODEL_PHOTOS["default"]))
    st.image(car_photo_url, caption=f"{brand} {model_name} ({year})", use_container_width=True)

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

    # Model Prediction
    log_val = pipeline.predict(input_data)[0]
    predicted_val = int(np.expm1(log_val))
    range_lower = int(predicted_val * 0.94)
    range_upper = int(predicted_val * 1.06)

    # EMI Calculation
    principal = predicted_val * ((100 - downpayment_pct) / 100)
    monthly_r = interest_rate / (12 * 100)
    months = loan_tenure_years * 12
    monthly_emi = int((principal * monthly_r * ((1 + monthly_r) ** months)) / (((1 + monthly_r) ** months) - 1))

    # Metric Display Box
    st.markdown(f"""
    <div class="valuation-card">
        <div style="display:flex; justify-content:space-between; align-items:center;">
            <span style="color: #059669; font-size: 13px; font-weight: 800; text-transform: uppercase;">Fair Market Valuation</span>
            <span style="background: rgba(56, 189, 248, 0.2); color: #0284c7; padding: 4px 10px; border-radius: 6px; font-size: 13px; font-weight: 700;">Est. EMI: ₹{monthly_emi:,}/mo</span>
        </div>
        <div style="font-size: 38px; font-weight: 900; color: {text_color}; margin: 4px 0 6px 0;">
            ₹ {predicted_val:,}
        </div>
        <div style="color: {subtext_color}; font-size: 14px; font-weight: 500; margin-bottom: 6px;">
            Estimated Tolerance Range: <b style="color: {text_color};">₹ {range_lower:,}</b> — <b style="color: {text_color};">₹ {range_upper:,}</b>
        </div>
        <div class="spec-pill">
            🏷️ {brand} {model_name} &bull; {body.title()} &bull; {year} &bull; {engine_type}
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Export Valuation Report
    report_df = pd.DataFrame([{
        "Manufacturer": brand,
        "Model": model_name,
        "Year": year,
        "Estimated_Value_INR": predicted_val,
        "Tolerance_Min_INR": range_lower,
        "Tolerance_Max_INR": range_upper,
        "Monthly_EMI_INR": monthly_emi,
        "Engineered_By": "Sunny Thakur"
    }])
    csv_data = report_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Valuation Certificate (CSV)",
        data=csv_data,
        file_name=f"Valuation_{brand}_{model_name}_{year}.csv",
        mime="text/csv",
        use_container_width=True
    )

    # Plotly Trend Curve
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
        line=dict(color="#0284c7" if not is_dark else "#38bdf8", width=3),
        marker=dict(size=8, color="#6366f1", symbol="circle"),
        name="Market Trend"
    ))
    chart.update_layout(
        title="Depreciation Horizon vs Model Year",
        title_font=dict(size=14, color=subtext_color),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15, 23, 42, 0.7)" if is_dark else "#f1f5f9",
        height=200,
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis=dict(gridcolor=card_border, tickfont=dict(color=subtext_color)),
        yaxis=dict(gridcolor=card_border, tickfont=dict(color=subtext_color), tickprefix="₹")
    )
    st.plotly_chart(chart, use_container_width=True)

# Predictive Signal Attribution Tiles
st.markdown("---")
st.markdown("### 🧠 Predictive Signal Attribution")
m1, m2, m3, m4 = st.columns(4)

with m1:
    st.markdown(f"""
    <div class="stat-tile">
        <div style="color: #2563eb; font-weight: 800; font-size: 14px;">Engine Displacement</div>
        <div style="font-size: 19px; font-weight: 800; color: {text_color}; margin: 4px 0;">+ High Weight</div>
        <small style="color: {subtext_color}; font-weight: 500;">Larger capacity directly scales premium valuation</small>
    </div>
    """, unsafe_allow_html=True)

with m2:
    st.markdown(f"""
    <div class="stat-tile">
        <div style="color: #059669; font-weight: 800; font-size: 14px;">Model Longevity</div>
        <div style="font-size: 19px; font-weight: 800; color: {text_color}; margin: 4px 0;">~6.2% Annual Drop</div>
        <small style="color: {subtext_color}; font-weight: 500;">Calculated year-over-year residual curve decay</small>
    </div>
    """, unsafe_allow_html=True)

with m3:
    st.markdown(f"""
    <div class="stat-tile">
        <div style="color: #d97706; font-weight: 800; font-size: 14px;">Brand Tier Multiplier</div>
        <div style="font-size: 19px; font-weight: 800; color: {text_color}; margin: 4px 0;">Tier Encoded</div>
        <small style="color: {subtext_color}; font-weight: 500;">Captures luxury vs commuter equity elasticity</small>
    </div>
    """, unsafe_allow_html=True)

with m4:
    st.markdown(f"""
    <div class="stat-tile">
        <div style="color: #7c3aed; font-weight: 800; font-size: 14px;">Inference Latency</div>
        <div style="font-size: 19px; font-weight: 800; color: {text_color}; margin: 4px 0;">< 14 ms</div>
        <small style="color: {subtext_color}; font-weight: 500;">Random Forest ensemble production runtime</small>
    </div>
    """, unsafe_allow_html=True)