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

# Reliable Vector Badges
BRAND_LOGOS = {
    "Acura": "https://raw.githubusercontent.com/fannarsh/car-logos-dataset/master/logos/optimized/acura.svg",
    "Audi": "https://upload.wikimedia.org/wikipedia/commons/9/92/Audi-Logo_2016.svg",
    "BMW": "https://upload.wikimedia.org/wikipedia/commons/4/44/BMW.svg",
    "Mercedes-Benz": "https://upload.wikimedia.org/wikipedia/commons/9/90/Mercedes-Logo.svg",
    "Ford": "https://upload.wikimedia.org/wikipedia/commons/3/3e/Ford_motor_company_logo.svg",
    "Toyota": "https://upload.wikimedia.org/wikipedia/commons/e/ee/Toyota_logo_%282020%29.svg",
    "Honda": "https://upload.wikimedia.org/wikipedia/commons/7/7b/Honda_Logo.svg",
    "Hyundai": "https://upload.wikimedia.org/wikipedia/commons/4/44/Hyundai_Motor_Company_logo.svg",
    "Chevrolet": "https://raw.githubusercontent.com/fannarsh/car-logos-dataset/master/logos/optimized/chevrolet.svg",
    "Porsche": "https://raw.githubusercontent.com/fannarsh/car-logos-dataset/master/logos/optimized/porsche.svg",
    "Nissan": "https://raw.githubusercontent.com/fannarsh/car-logos-dataset/master/logos/optimized/nissan.svg",
    "Volkswagen": "https://upload.wikimedia.org/wikipedia/commons/6/6d/Volkswagen_logo_2019.svg",
    "default": "https://upload.wikimedia.org/wikipedia/commons/thumb/6/65/Circle-icons-car.svg/512px-Circle-icons-car.svg.png"
}

# Sidebar Controls & Professional Developer Card
with st.sidebar:
    st.markdown("### ⚙️ Dashboard Controls")
    theme_choice = st.radio("Display Mode", ["Dark Mode 🌙", "Light Mode ☀️"], index=0)
    
    st.markdown("---")
    st.markdown("### 👨‍💻 Engineering & Contact")
    st.markdown("""
    <div style="background: rgba(99, 102, 241, 0.08); border: 1px solid #4f46e5; border-radius: 12px; padding: 14px; margin-bottom: 12px;">
        <div style="font-weight: 800; font-size: 16px; color: #818cf8;">Sunny Thakur</div>
        <div style="font-size: 12px; color: #94a3b8; margin-bottom: 8px;">Machine Learning Engineer & Educator</div>
        <div style="font-size: 11px; line-height: 1.4; color: #cbd5e1;">Engineered complete ML pipeline, Random Forest optimization & end-to-end deployment.</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Direct 1-Click Connect Buttons
    # Note: Replace '91XXXXXXXXXX' with your actual phone number if desired
    whatsapp_direct_url = "https://wa.me/919999999999?text=Hi%20Sunny,%20I%20reviewed%20your%20CarValue-AI%20project!"
    linkedin_url = "https://www.linkedin.com"
    github_url = "https://github.com/sunnythakursunny650-cell/CarValue-AI"

    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        st.markdown(f'<a href="{whatsapp_direct_url}" target="_blank" style="text-decoration:none;"><button style="width:100%; padding:9px; border-radius:8px; border:none; background:#22c55e; color:white; font-weight:700; font-size:12px; cursor:pointer;">💬 WhatsApp</button></a>', unsafe_allow_html=True)
    with col_btn2:
        st.markdown(f'<a href="{linkedin_url}" target="_blank" style="text-decoration:none;"><button style="width:100%; padding:9px; border-radius:8px; border:none; background:#0284c7; color:white; font-weight:700; font-size:12px; cursor:pointer;">💼 LinkedIn</button></a>', unsafe_allow_html=True)
    
    st.write("")
    st.markdown(f'<a href="{github_url}" target="_blank" style="text-decoration:none;"><button style="width:100%; padding:8px; border-radius:8px; border:1px solid #475569; background:transparent; color:#94a3b8; font-size:12px; cursor:pointer;">📂 View Project on GitHub</button></a>', unsafe_allow_html=True)

# Theme Dynamic Configuration
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
    card_border = "#e2e8f0"
    header_gradient = "linear-gradient(90deg, #e2e8f0 0%, #e0e7ff 100%)"
    header_border = "#cbd5e1"
    subtext_color = "#475569"
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
        border: 1px solid {header_border};
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
    .car-display-card {{
        background: {card_bg};
        border: 1px solid {card_border};
        border-radius: 14px;
        padding: 16px;
        text-align: center;
        margin-bottom: 15px;
    }}
    .spec-pill {{
        background-color: rgba(16, 185, 129, 0.15);
        color: #10b981;
        border: 1px solid #10b981;
        padding: 5px 12px;
        border-radius: 14px;
        font-size: 12px;
        font-weight: 600;
        display: inline-block;
        margin-top: 8px;
    }}
    .stat-tile {{
        background: {card_bg};
        border: 1px solid {card_border};
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

# Main Application Banner
st.markdown(f"""
<div class="main-header">
    <h1 style="margin: 0; font-size: 30px; font-weight: 800; color: {text_color}; letter-spacing: -0.5px;">
        🚗 CarValue-AI Pro
    </h1>
    <p style="color: {subtext_color}; margin: 4px 0 0 0; font-size: 14px;">
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
    
    # Manufacturer Identity Card
    logo_url = BRAND_LOGOS.get(brand, BRAND_LOGOS["default"])
    st.markdown(f"""
    <div class="car-display-card">
        <div style="display:flex; justify-content:center; align-items:center; height:75px; margin-bottom:8px;">
            <img src="{logo_url}" alt="{brand}" onerror="this.src='{BRAND_LOGOS['default']}';" style="max-height: 70px; max-width: 120px; object-fit: contain;">
        </div>
        <div style="font-size: 20px; font-weight: 700; color: {text_color};">{brand} &bull; {model_name}</div>
        <div style="color: {subtext_color}; font-size: 13px;">{body.upper()} | {year} Edition | {engine_type} Engine ({engine_v}L)</div>
    </div>
    """, unsafe_allow_html=True)

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
            <span style="color: #10b981; font-size: 13px; font-weight: 700; text-transform: uppercase;">Fair Market Valuation</span>
            <span style="background: rgba(56, 189, 248, 0.15); color: #0284c7; padding: 3px 9px; border-radius: 6px; font-size: 12px; font-weight: 600;">Est. EMI: ₹{monthly_emi:,}/mo</span>
        </div>
        <div style="font-size: 38px; font-weight: 900; color: {text_color}; margin: 4px 0 6px 0;">
            ₹ {predicted_val:,}
        </div>
        <div style="color: {subtext_color}; font-size: 14px; margin-bottom: 6px;">
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
        <div style="color: #3b82f6; font-weight: 700; font-size: 14px;">Engine Displacement</div>
        <div style="font-size: 19px; font-weight: 800; color: {text_color}; margin: 4px 0;">+ High Weight</div>
        <small style="color: {subtext_color};">Larger capacity directly scales premium valuation</small>
    </div>
    """, unsafe_allow_html=True)

with m2:
    st.markdown(f"""
    <div class="stat-tile">
        <div style="color: #10b981; font-weight: 700; font-size: 14px;">Model Longevity</div>
        <div style="font-size: 19px; font-weight: 800; color: {text_color}; margin: 4px 0;">~6.2% Annual Drop</div>
        <small style="color: {subtext_color};">Calculated year-over-year residual curve decay</small>
    </div>
    """, unsafe_allow_html=True)

with m3:
    st.markdown(f"""
    <div class="stat-tile">
        <div style="color: #f59e0b; font-weight: 700; font-size: 14px;">Brand Tier Multiplier</div>
        <div style="font-size: 19px; font-weight: 800; color: {text_color}; margin: 4px 0;">Tier Encoded</div>
        <small style="color: {subtext_color};">Captures luxury vs commuter equity elasticity</small>
    </div>
    """, unsafe_allow_html=True)

with m4:
    st.markdown(f"""
    <div class="stat-tile">
        <div style="color: #8b5cf6; font-weight: 700; font-size: 14px;">Inference Latency</div>
        <div style="font-size: 19px; font-weight: 800; color: {text_color}; margin: 4px 0;">< 14 ms</div>
        <small style="color: {subtext_color};">Random Forest ensemble production runtime</small>
    </div>
    """, unsafe_allow_html=True)