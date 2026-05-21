import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.preprocessing import LabelEncoder

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="Student Performance Analysis",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================================
# THEME TOGGLE  (put this BEFORE the CSS block)
# =====================================================
if "theme" not in st.session_state:
    st.session_state.theme = "Light"

# Tiny helper: returns a dict of colors for the chosen theme
def get_palette(theme: str) -> dict:
    if theme == "Dark":
        return {
            "bg":            "#0b1220",   # main background
            "card_bg":       "#111b2e",   # card / panel background
            "sidebar_bg":    "#070d1a",   # sidebar background
            "border":        "#1f2a44",   # subtle border
            "text":          "#e2e8f0",   # primary text
            "muted":         "#94a3b8",   # secondary text
            "heading":       "#f8fafc",   # headings (h1, big values)
            "soft_panel":    "#0f1a2e",   # chart-note background
            "input_bg":      "#111b2e",
            "warn_bg": "#3a2a05", "warn_br": "#7c5b14", "warn_fg": "#fde68a",
            "err_bg":  "#3a0c14", "err_br":  "#7c1d2e", "err_fg":  "#fecdd3",
            "ok_bg":   "#0c2e1a", "ok_br":   "#15803d", "ok_fg":   "#bbf7d0",
            "info_bg": "#0c1f3a", "info_br": "#1e40af", "info_fg": "#bfdbfe",
            "badge_exc_bg": "#0c2e1a", "badge_exc_fg": "#86efac",
            "badge_avg_bg": "#3a2a05", "badge_avg_fg": "#fde68a",
            "badge_poor_bg":"#3a0c14", "badge_poor_fg":"#fecdd3",
            "btn_bg":  "#3b82f6", "btn_fg": "#f8fafc", "btn_hover": "#2563eb",
            "plotly_paper": "#0b1220", "plotly_plot": "#0b1220",
            "plotly_font": "#e2e8f0", "plotly_grid": "#1f2a44",
        }
    # Light (default)
    return {
        "bg":            "#f8fafc",
        "card_bg":       "#ffffff",
        "sidebar_bg":    "#0f172a",
        "border":        "#e2e8f0",
        "text":          "#334155",
        "muted":         "#94a3b8",
        "heading":       "#0f172a",
        "soft_panel":    "#f8fafc",
        "input_bg":      "#ffffff",
        "warn_bg": "#fffbeb", "warn_br": "#fde68a", "warn_fg": "#92400e",
        "err_bg":  "#fff1f2", "err_br":  "#fecdd3", "err_fg":  "#9f1239",
        "ok_bg":   "#f0fdf4", "ok_br":   "#bbf7d0", "ok_fg":   "#166534",
        "info_bg": "#eff6ff", "info_br": "#bfdbfe", "info_fg": "#1e40af",
        "badge_exc_bg": "#dcfce7", "badge_exc_fg": "#15803d",
        "badge_avg_bg": "#fef9c3", "badge_avg_fg": "#a16207",
        "badge_poor_bg":"#fee2e2", "badge_poor_fg":"#b91c1c",
        "btn_bg":  "#0f172a", "btn_fg": "#f8fafc", "btn_hover": "#1e293b",
        "plotly_paper": "#ffffff", "plotly_plot": "#ffffff",
        "plotly_font": "#374151", "plotly_grid": "#e5e7eb",
    }

P = get_palette(st.session_state.theme)

# =====================================================
# CUSTOM CSS — Professional Dark Sidebar + Clean Content
# =====================================================

st.markdown(f"""
<style>

@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Serif+Display&display=swap');

html, body, [class*="css"] {{
    font-family: 'DM Sans', sans-serif;
    color: {P['text']};
}}

.main,
[data-testid="stAppViewContainer"] {{
    background: {P['bg']};
    color: {P['text']};
}}

[data-testid="stSidebar"] {{
    background: {P['sidebar_bg']} !important;
    border-right: 1px solid {P['border']};
}}

.sidebar-title   {{ color: #f8fafc; font-size: 22px; font-weight: 700; line-height: 1.4; }}
.sidebar-subtitle{{ color: #94a3b8; font-size: 12px; margin-top: 5px; }}

[data-testid="stSidebar"] label {{
    color: #e2e8f0 !important;
    font-size: 14px !important;
    font-weight: 500 !important;
}}

/* Main content text colors */
[data-testid="stAppViewContainer"] p,
[data-testid="stAppViewContainer"] li,
[data-testid="stAppViewContainer"] span,
[data-testid="stAppViewContainer"] label,
[data-testid="stAppViewContainer"] div {{
    color: {P['text']};
}}

.page-header {{
    padding: 8px 0 4px 0;
    margin-bottom: 28px;
    border-bottom: 2px solid {P['border']};
}}
.page-header h1 {{
    font-family: 'DM Serif Display', serif;
    font-size: 32px; color: {P['heading']};
    margin: 0 0 4px 0; font-weight: 400;
}}
.page-header p {{ color: {P['muted']}; font-size: 14px; margin: 0; }}

.section-label {{
    font-size: 11px; font-weight: 600;
    letter-spacing: 1.2px; text-transform: uppercase;
    color: {P['muted']}; margin: 32px 0 14px 0;
}}

.stat-card {{
    background: {P['card_bg']};
    border: 1px solid {P['border']};
    border-radius: 12px;
    padding: 22px 20px 18px 20px;
}}
.stat-label {{
    font-size: 11px; font-weight: 600;
    text-transform: uppercase; letter-spacing: 1px;
    color: {P['muted']}; margin-bottom: 8px;
}}
.stat-value {{
    font-size: 30px; font-weight: 600;
    color: {P['heading']}; line-height: 1;
    font-family: 'DM Serif Display', serif;
}}

.result-panel {{
    background: {P['card_bg']};
    border: 1px solid {P['border']};
    border-radius: 14px;
    padding: 28px; text-align: center; margin-bottom: 16px;
}}
.result-grade {{
    font-family: 'DM Serif Display', serif;
    font-size: 64px; color: {P['heading']};
    line-height: 1; margin-bottom: 6px;
}}
.result-sublabel {{
    font-size: 12px; text-transform: uppercase;
    letter-spacing: 1px; color: {P['muted']}; font-weight: 600;
}}
.result-badge {{
    display: inline-block; margin-top: 16px;
    padding: 6px 18px; border-radius: 999px;
    font-size: 13px; font-weight: 600;
}}

.badge-excellent {{ background: {P['badge_exc_bg']};  color: {P['badge_exc_fg']}; }}
.badge-average   {{ background: {P['badge_avg_bg']};  color: {P['badge_avg_fg']}; }}
.badge-poor      {{ background: {P['badge_poor_bg']}; color: {P['badge_poor_fg']}; }}

.rec-card {{
    border-radius: 10px; padding: 14px 16px;
    margin-bottom: 10px; font-size: 13.5px;
    font-weight: 500; line-height: 1.5;
}}
.rec-warning {{ background: {P['warn_bg']}; border: 1px solid {P['warn_br']}; color: {P['warn_fg']}; }}
.rec-error   {{ background: {P['err_bg']};  border: 1px solid {P['err_br']};  color: {P['err_fg']};  }}
.rec-success {{ background: {P['ok_bg']};   border: 1px solid {P['ok_br']};   color: {P['ok_fg']};   }}
.rec-info    {{ background: {P['info_bg']}; border: 1px solid {P['info_br']}; color: {P['info_fg']}; }}

/* Force text color INSIDE recommendation cards */
.rec-warning span, .rec-error span, .rec-success span, .rec-info span {{ color: inherit !important; }}

.stButton > button {{
    background: {P['btn_bg']} !important;
    color: {P['btn_fg']} !important;
    border: none !important; border-radius: 8px !important;
    font-size: 14px !important; font-weight: 600 !important;
    height: 44px !important;
}}
.stButton > button:hover {{ background: {P['btn_hover']} !important; }}

.chart-note {{
    font-size: 13px; color: {P['muted']};
    padding: 10px 14px; background: {P['soft_panel']};
    border-left: 3px solid {P['border']};
    border-radius: 0 6px 6px 0;
    margin: -4px 0 24px 0; line-height: 1.6;
}}

.model-card {{
    background: {P['card_bg']};
    border: 1px solid {P['border']};
    border-radius: 12px; padding: 24px;
}}
.model-card h4 {{
    font-family: 'DM Serif Display', serif;
    font-size: 18px; color: {P['heading']};
}}
.model-card p, .model-card li {{ color: {P['text']}; }}
.model-score {{ font-size: 36px; font-weight: 700; color: #3b82f6; }}

.divider {{ border: none; border-top: 1px solid {P['border']}; margin: 28px 0; }}

/* Inputs (selectbox, slider labels) in dark mode */
.stSelectbox label, .stSlider label {{ color: {P['text']} !important; }}
.stSelectbox div[data-baseweb="select"] > div {{
    background: {P['input_bg']} !important;
    color: {P['text']} !important;
}}

/* Dataframe / expander */
.streamlit-expanderHeader {{ color: {P['text']} !important; }}

</style>
""", unsafe_allow_html=True)

# =====================================================
# LOAD MODELS & DATA
# =====================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@st.cache_resource
def load_models():
    reg_model = joblib.load(os.path.join(BASE_DIR, "regression_model.pkl"))
    clf_model = joblib.load(os.path.join(BASE_DIR, "classification_model.pkl"))
    return reg_model, clf_model

@st.cache_data
def load_data():
    return pd.read_csv(os.path.join(BASE_DIR, "cleaned_student_data.csv"))

reg_model, clf_model = load_models()
df = load_data()

# =====================================================
# SIDEBAR
# =====================================================

with st.sidebar:

    st.markdown(
        '<div style="padding:20px 0 24px 0;">'
        '<div class="sidebar-title">Student Performance</div>'
        '<div class="sidebar-subtitle">Analysis System · v1.0</div>'
        '</div>',
        unsafe_allow_html=True
    )

    menu = st.radio(
        "Navigation",
        [
            "Home",
            "Prediction",
            "EDA Dashboard",
            "Model Performance",
            "About"
        ]
    )
    st.markdown("<hr style='border-color:#1e293b; margin:24px 0;'>", unsafe_allow_html=True)
    new_theme = st.radio(
        "Appearance",
        ["Light", "Dark"],
        index=0 if st.session_state.theme == "Light" else 1,
        horizontal=True,
        key="theme_picker",
    )
    if new_theme != st.session_state.theme:
        st.session_state.theme = new_theme
        st.rerun()

    

# =====================================================
# HOME
# =====================================================

if menu == "Home":

    st.markdown("""
        <div class="page-header">
            <h1>Student Performance Analysis</h1>
            <p>Machine learning–based prediction and insight into academic outcomes</p>
        </div>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    cards = [
        (c1, "Total Students",    f"{len(df):,}"),
        (c2, "Avg Final Grade",   f"{df['FinalGrade'].mean():.1f}"),
        (c3, "Avg Attendance",    f"{df['AttendanceRate'].mean():.1f}%"),
        (c4, "Excellent Rate",    f"{(df['PerformanceCategory']=='Excellent').mean()*100:.1f}%"),
    ]
    for col, label, value in cards:
        with col:
            st.markdown(f"""
                <div class="stat-card">
                    <div class="stat-label">{label}</div>
                    <div class="stat-value">{value}</div>
                </div>
            """, unsafe_allow_html=True)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    st.markdown('<div class="section-label">What this system offers</div>', unsafe_allow_html=True)
    f1, f2 = st.columns(2)
    with f1:
        st.markdown("""
        **Prediction Engine**
        Enter a student's profile — attendance, study hours, parental support, prior grades — and receive a predicted final grade and performance category (Poor / Average / Excellent), along with actionable recommendations.
        """)
    with f2:
        st.markdown("""
        **Analytics Dashboard**
        Explore grade distributions, attendance patterns, category breakdowns, and feature correlations through interactive charts — all drawn from the real dataset.
        """)

# =====================================================
# PREDICTION
# =====================================================

elif menu == "Prediction":

    st.markdown("""
        <div class="page-header">
            <h1>Performance Prediction</h1>
            <p>Enter student details to generate a grade estimate and performance category</p>
        </div>
    """, unsafe_allow_html=True)

    left, right = st.columns([1.2, 1], gap="large")

    with left:
        st.markdown('<div class="section-label">Student Profile</div>', unsafe_allow_html=True)

        gender = st.selectbox("Gender", ["Male", "Female"])
        parental_support = st.selectbox("Parental Support Level", ["Low", "Medium", "High"])
        online_classes = st.selectbox("Online Classes Taken", ["No", "Yes"])

        st.markdown('<div class="section-label" style="margin-top:20px;">Academic Indicators</div>', unsafe_allow_html=True)

        col_a, col_b = st.columns(2)
        with col_a:
            attendance_rate    = st.slider("Attendance Rate (%)",      0, 100, 80)
            previous_grade     = st.slider("Previous Grade",           0, 100, 75)
            study_hours        = st.slider("Daily Study Hours",        0, 10,  4)
        with col_b:
            study_hours_week   = st.slider("Study Hours / Week",       0, 40,  15)
            extracurricular    = st.slider("Extracurricular Activities",0, 10,  2)
            attendance_percent = st.slider("Attendance Percentage (%)", 0, 100, 85)

        predict_btn = st.button("Generate Prediction", use_container_width=True)

    with right:
        st.markdown('<div class="section-label">Result</div>', unsafe_allow_html=True)

        if predict_btn:
            gender_enc     = 1 if gender == "Male" else 0
            parental_enc   = {"Low": 1, "Medium": 2, "High": 0}[parental_support]
            online_enc     = 1 if online_classes == "Yes" else 0

            features = np.array([[
                gender_enc, attendance_rate, study_hours_week,
                previous_grade, extracurricular, parental_enc,
                study_hours, attendance_percent, online_enc
            ]])

            grade_pred     = reg_model.predict(features)[0]
            class_pred     = clf_model.predict(features)[0]
            class_labels   = {0: "Average", 1: "Excellent", 2: "Poor"}
            category       = class_labels[class_pred]
            badge_cls      = f"badge-{category.lower()}"

            st.markdown(f"""
                <div class="result-panel">
                    <div class="result-sublabel">Predicted Final Grade</div>
                    <div class="result-grade">{grade_pred:.1f}</div>
                    <div><span class="result-badge {badge_cls}">{category}</span></div>
                </div>
            """, unsafe_allow_html=True)

            st.markdown('<div class="section-label">Recommendations</div>', unsafe_allow_html=True)

            recs = []
            if attendance_rate < 75:
                recs.append(("warning", "⚠", "Attendance is below 75%. Consistent attendance has a measurable impact on final outcomes."))
            if study_hours_week < 10:
                recs.append(("warning", "⚠", "Weekly study hours are low. A target of 10–15 hours tends to correlate with stronger results."))
            if category == "Poor":
                recs.append(("error", "!", "This student may be at academic risk. Consider early intervention or dedicated mentoring."))
            if category == "Excellent":
                recs.append(("success", "✓", "Strong profile. Encourage the student to sustain this momentum through the semester."))
            if not recs:
                recs.append(("info", "·", "No major concerns. The student's indicators look stable overall."))

            for kind, icon, text in recs:
                st.markdown(f'<div class="rec-card rec-{kind}"><span>{icon}</span> <span>{text}</span></div>', unsafe_allow_html=True)

        else:
            st.markdown("""
                <div style="background:#f8fafc; border:1px dashed #cbd5e1; border-radius:12px;
                            padding:48px 24px; text-align:center; color:#94a3b8;">
                    <div style="font-size:28px; margin-bottom:10px;">📋</div>
                    <div style="font-size:14px; font-weight:500;">Fill in the profile and click<br><strong>Generate Prediction</strong></div>
                </div>
            """, unsafe_allow_html=True)

# =====================================================
# EDA DASHBOARD
# =====================================================

elif menu == "EDA Dashboard":

    st.markdown("""
        <div class="page-header">
            <h1>Exploratory Data Analysis</h1>
            <p>Visual patterns and relationships across the student dataset</p>
        </div>
    """, unsafe_allow_html=True)

    BLUE   = "#3b82f6"
    GREEN  = "#22c55e"
    AMBER  = "#f59e0b"
    RED    = "#ef4444"
    CAT_COLORS = {"Excellent": GREEN, "Average": AMBER, "Poor": RED}

    LAYOUT = dict(
        plot_bgcolor=P["plotly_plot"],
        paper_bgcolor=P["plotly_paper"],
        font=dict(family="DM Sans, sans-serif", color=P["plotly_font"], size=12),
        margin=dict(t=44, b=20, l=12, r=12),
        title_font=dict(size=14, color=P["heading"]),
        xaxis=dict(gridcolor=P["plotly_grid"], zerolinecolor=P["plotly_grid"]),
        yaxis=dict(gridcolor=P["plotly_grid"], zerolinecolor=P["plotly_grid"]),
    )

    st.markdown('<div class="section-label">Final Grade Distribution</div>', unsafe_allow_html=True)
    fig1 = px.histogram(df, x="FinalGrade", nbins=20, color_discrete_sequence=[BLUE])
    fig1.update_layout(**LAYOUT, bargap=0.06)
    fig1.update_traces(marker_line_width=0)
    st.plotly_chart(fig1, use_container_width=True)
    st.markdown('<div class="chart-note">Most students scored between 75–90, suggesting generally strong academic performance across the dataset.</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-label">Performance Category Breakdown</div>', unsafe_allow_html=True)
    cat = df["PerformanceCategory"].value_counts().reset_index()
    cat.columns = ["Category", "Count"]
    fig2 = px.bar(cat, x="Category", y="Count", color="Category",
                  color_discrete_map=CAT_COLORS)
    fig2.update_layout(**LAYOUT, showlegend=False)
    st.plotly_chart(fig2, use_container_width=True)
    st.markdown('<div class="chart-note">The dataset is imbalanced — the Excellent category dominates, which affects classification model training and evaluation.</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-label">Attendance vs Final Grade</div>', unsafe_allow_html=True)
    fig3 = px.scatter(df, x="AttendanceRate", y="FinalGrade",
                      color="PerformanceCategory",
                      color_discrete_map=CAT_COLORS, opacity=0.55)
    fig3.update_layout(**LAYOUT)
    st.plotly_chart(fig3, use_container_width=True)
    st.markdown('<div class="chart-note">Higher attendance generally correlates with better grades, though the relationship is not strictly linear.</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-label">Feature Correlation Heatmap</div>', unsafe_allow_html=True)
    hmap = df.drop(columns=["StudentID", "Name"], errors="ignore").copy()
    le = LabelEncoder()
    for col in ["Gender", "ParentalSupport", "Online Classes Taken", "PerformanceCategory"]:
        if col in hmap.columns:
            hmap[col] = le.fit_transform(hmap[col].astype(str))

    fig4, ax4 = plt.subplots(figsize=(11, 7))
    fig4.patch.set_facecolor(P["plotly_paper"])
    ax4.set_facecolor(P["plotly_paper"])
    ax4.tick_params(colors=P["text"])
    for spine in ax4.spines.values():
        spine.set_color(P["border"])
    sns.heatmap(hmap.corr(), annot=True, fmt=".2f", cmap="coolwarm",
                linewidths=0.4, linecolor="#f1f5f9",
                ax=ax4, annot_kws={"size": 8.5},
                cbar_kws={"shrink": 0.8})
    ax4.set_title("Correlation Matrix", fontsize=13, color=P["heading"], pad=14, loc="left")
    plt.tight_layout()
    st.pyplot(fig4)
    st.markdown('<div class="chart-note">Most features show weak-to-moderate correlation with FinalGrade. The model benefits from combining these features rather than relying on any single predictor.</div>', unsafe_allow_html=True)

    with st.expander("View Raw Dataset (first 10 rows)"):
        st.dataframe(df.head(10), use_container_width=True)

# =====================================================
# MODEL PERFORMANCE
# =====================================================

elif menu == "Model Performance":

    st.markdown("""
        <div class="page-header">
            <h1>Model Performance</h1>
            <p>Summary of models trained and their evaluation results</p>
        </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-label">Grade Prediction</div>', unsafe_allow_html=True)
    m1, m2 = st.columns(2)

    with m1:
        st.markdown("""
            <div class="model-card">
                <h4>Random Forest Regressor</h4>
                <div class="model-score">R² −0.03</div>
                <p>Performance remained weak due to low feature-target correlation in the dataset.
                   Synthetic patterns limited the model's ability to generalise.</p>
                <ul>
                    <li>Linear Regression (baseline)</li>
                    <li>Random Forest Regressor</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)

    with m2:
        st.markdown("""
            <div class="model-card">
                <h4>Random Forest Classifier</h4>
                <div class="model-score">66% Accuracy</div>
                <p>SMOTE oversampling was applied to address class imbalance and reduce
                   majority-class bias during training.</p>
                <ul>
                    <li>Logistic Regression (baseline)</li>
                    <li>Random Forest Classifier</li>
                    <li>SMOTE for class balancing</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    

# =====================================================
# ABOUT
# =====================================================

elif menu == "About":

    st.markdown("""
        <div class="page-header">
            <h1>About this Project</h1>
            <p>Minor project · Machine Learning portfolio</p>
        </div>
    """, unsafe_allow_html=True)

    a1, a2 = st.columns([1.4, 1], gap="large")
    with a1:
        st.markdown("""
        This system was built to predict student academic performance using supervised machine learning.
        It combines a regression model (to estimate numeric final grades) and a classification model
        (to assign performance categories: Poor, Average, Excellent), wrapped in an interactive Streamlit dashboard.

        Class imbalance in the dataset was addressed using SMOTE oversampling during model training.
        All categorical features were encoded using label encoding before being passed to the models.
        """)
    with a2:
        st.markdown("""
        **Tech Stack**

        Python · Scikit-learn · Streamlit
        Pandas · NumPy · Plotly · Seaborn · Matplotlib

        **ML Techniques**

        Random Forest · Logistic Regression
        Linear Regression · SMOTE · Label Encoding
        """)
