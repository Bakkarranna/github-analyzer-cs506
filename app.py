"""
app.py
CS506 – Big Data Analytics Project
GitHub Repository Intelligence Analyzer
Muhammad Abubakar Siddique | 2023-AG-10411

Step 6: Run this to launch the Streamlit web interface.
Command: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="GitHub Repo Intelligence",
    page_icon="🔭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS (Dark mode default, light mode override) ──────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Inter:wght@300;400;600&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    /* ── Dark mode (DEFAULT) ─────────────────────────────────── */
    :root {
        --bg-main: #0d1117;
        --bg-card: linear-gradient(135deg, #161b22 0%, #1c2128 100%);
        --border-color: #30363d;
        --text-primary: #c9d1d9;
        --text-secondary: #8b949e;
        --accent: #58a6ff;
        --sidebar-bg: #161b22;
        --sidebar-border: #21262d;
        --predict-popular-bg: linear-gradient(135deg, #0d2818, #0a3a1d);
        --predict-not-bg: linear-gradient(135deg, #2d1515, #3d1a1a);
        --insight-bg: #161b22;
    }

    /* Dark mode base overrides */
    :root {
        color-scheme: dark;
    }
    .stApp {
        background-color: #0d1117 !important;
    }
    div[data-testid="stToolbar"] {
        background: #0d1117 !important;
    }

    /* ── Light mode (only when user explicitly prefers light) ── */
    @media (prefers-color-scheme: light) {
        :root {
            color-scheme: light !important;
            --bg-main: #ffffff;
            --bg-card: linear-gradient(135deg, #f6f8fa 0%, #eaeef2 100%);
            --border-color: #d0d7de;
            --text-primary: #1f2328;
            --text-secondary: #656d76;
            --accent: #0550ae;
            --sidebar-bg: #ffffff;
            --sidebar-border: #d0d7de;
            --predict-popular-bg: linear-gradient(135deg, #dafbe1, #b3f0c7);
            --predict-not-bg: linear-gradient(135deg, #ffebe9, #fddcdb);
            --insight-bg: #f6f8fa;
        }
        .stApp {
            background-color: #ffffff !important;
        }
        div[data-testid="stToolbar"] {
            background: #ffffff !important;
        }
        .main {
            background: #ffffff !important;
            color: #1f2328 !important;
        }
        div[data-testid="stSidebarContent"] {
            background: #ffffff !important;
            border-right: 1px solid #d0d7de !important;
        }
        div[data-testid="stSidebarContent"] * {
            color: #1f2328 !important;
        }
        .section-header {
            color: #0550ae !important;
            border-bottom: 1px solid #d0d7de !important;
        }
        .metric-card {
            background: #f6f8fa !important;
            border: 1px solid #d0d7de !important;
        }
        .metric-card .metric-value {
            color: #0550ae !important;
        }
        .metric-card .metric-label {
            color: #656d76 !important;
        }
        .stButton>button {
            background: #2da44e !important;
            color: #ffffff !important;
        }
        .stButton>button:hover {
            background: #218838 !important;
        }
        .insight-box {
            background: #f6f8fa !important;
            border-left: 3px solid #0550ae !important;
            color: #1f2328 !important;
        }
        .predict-result-popular {
            background: linear-gradient(135deg, #dafbe1, #b3f0c7) !important;
            border: 1px solid #2ea043 !important;
        }
        .predict-result-not {
            background: linear-gradient(135deg, #ffebe9, #fddcdb) !important;
            border: 1px solid #da3633 !important;
        }
        /* Fix all text elements in light mode */
        p, span, div, label, h1, h2, h3, h4, h5, h6 {
            color: #1f2328 !important;
        }
        div[data-testid="stMarkdownContainer"] p {
            color: #1f2328 !important;
        }
    }

    .main { background: var(--bg-main); }

    .metric-card {
        background: var(--bg-card);
        border: 1px solid var(--border-color);
        border-radius: 12px;
        padding: 20px 24px;
        text-align: center;
        transition: border-color 0.2s;
    }
    .metric-card:hover { border-color: var(--accent); }
    .metric-value {
        font-family: 'JetBrains Mono', monospace;
        font-size: 2rem;
        font-weight: 600;
        color: var(--accent);
        line-height: 1.2;
    }
    .metric-label {
        font-size: 0.78rem;
        color: var(--text-secondary);
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-top: 4px;
    }

    .predict-result-popular {
        background: var(--predict-popular-bg);
        border: 1px solid #2ea043;
        border-radius: 12px;
        padding: 24px;
        margin-top: 16px;
    }
    .predict-result-not {
        background: var(--predict-not-bg);
        border: 1px solid #da3633;
        border-radius: 12px;
        padding: 24px;
        margin-top: 16px;
    }

    .section-header {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.8rem;
        color: var(--accent);
        text-transform: uppercase;
        letter-spacing: 0.12em;
        border-bottom: 1px solid var(--sidebar-border);
        padding-bottom: 8px;
        margin-bottom: 16px;
    }

    div[data-testid="stSidebarContent"] {
        background: var(--sidebar-bg);
        border-right: 1px solid var(--sidebar-border);
    }

    .stButton>button {
        background: #238636;
        color: #ffffff;
        border: none;
        border-radius: 8px;
        padding: 10px 24px;
        font-weight: 600;
        transition: background 0.2s;
        width: 100%;
    }
    .stButton>button:hover { background: #2ea043; }

    .insight-box {
        background: var(--insight-bg);
        border-left: 3px solid var(--accent);
        border-radius: 0 8px 8px 0;
        padding: 12px 16px;
        font-size: 0.88rem;
        color: var(--text-primary);
        margin-top: 12px;
    }
</style>
""", unsafe_allow_html=True)

# ── Import backend ────────────────────────────────────────────────────────────
try:
    import backend as bk
    BACKEND_OK = True
except Exception as e:
    BACKEND_OK = False
    BACKEND_ERROR = str(e)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔭 GitHub Intelligence")
    st.markdown("**CS506 — Big Data Analytics**")
    st.markdown("*Muhammad Abubakar Siddique*")
    st.markdown("*2023-AG-10411*")
    st.markdown("---")

    page = st.radio(
        "Navigate",
        ["📊 Dashboard", "🔮 Predict Popularity", "📈 Language Insights", "🧪 Model Performance"],
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown(
        "<div style='font-size:0.75rem;color:#6e7681'>Dataset: GitHub Repositories<br>Source: Kaggle<br>Model: Random Forest / GBM</div>",
        unsafe_allow_html=True
    )


# ── Load data ─────────────────────────────────────────────────────────────────
@st.cache_data
def get_data():
    return bk.load_dataset()

@st.cache_data
def get_metrics():
    return bk.load_metrics()


def show_error(msg):
    st.error(f"⚠️ {msg}")
    st.info("Make sure you've run `preprocess.py` and `train_model.py` first.")


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 1 — DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
if page == "📊 Dashboard":
    st.markdown("# 🔭 GitHub Repository Intelligence Analyzer")
    st.markdown("*Analyzing open-source repository patterns using big data techniques.*")
    st.markdown("---")

    if not BACKEND_OK:
        show_error(BACKEND_ERROR); st.stop()

    try:
        df = get_data()
    except FileNotFoundError as e:
        show_error(str(e)); st.stop()

    kpis = bk.get_summary_kpis(df)

    # ── KPI Cards ─────────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Key Metrics</div>', unsafe_allow_html=True)
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    kpi_data = [
        (c1, f"{kpis['total_repos']:,}", "Total Repos"),
        (c2, f"{kpis['total_stars']:,}", "Total Stars"),
        (c3, str(kpis['avg_engagement']), "Avg Engagement"),
        (c4, kpis['top_language'], "Top Language"),
        (c5, f"{kpis['popular_repos_pct']}%", "Popular Repos"),
        (c6, str(kpis['avg_contributors']), "Avg Contributors"),
    ]
    for col, val, label in kpi_data:
        col.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{val}</div>
            <div class="metric-label">{label}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Row 1: Stars dist + Language bar ──────────────────────────────────────
    col_left, col_right = st.columns([1.2, 1])

    with col_left:
        st.markdown('<div class="section-header">Stars Distribution</div>', unsafe_allow_html=True)
        stars_data = bk.get_stars_distribution(df)
        if stars_data and "values" in stars_data:
            fig = px.histogram(
                x=stars_data["values"],
                nbins=60,
                color_discrete_sequence=["#58a6ff"],
                labels={"x": "Stars"},
            )
            fig.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=0, r=0, t=10, b=0),
                height=280,
                showlegend=False,
                xaxis_title="Stars",
                yaxis_title="Count",
            )
            fig.add_vline(x=stars_data["median"], line_dash="dash", line_color="#f0883e",
                          annotation_text=f"Median: {stars_data['median']}", annotation_position="top right")
            st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.markdown('<div class="section-header">Top Languages by Repo Count</div>', unsafe_allow_html=True)
        lang_df = bk.get_top_languages(df)
        if not lang_df.empty:
            fig2 = px.bar(
                lang_df.sort_values("count"),
                x="count", y="language", orientation="h",
                color="count",
                color_continuous_scale=["#0d419d", "#388bfd", "#79c0ff"],
                labels={"count": "Repositories", "language": ""},
            )
            fig2.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=0, r=0, t=10, b=0),
                height=280,
                coloraxis_showscale=False,
            )
            st.plotly_chart(fig2, use_container_width=True)

    # ── Row 2: Engagement by language + Contributor tiers ─────────────────────
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown('<div class="section-header">Avg Engagement Score by Language</div>', unsafe_allow_html=True)
        eng_df = bk.get_engagement_by_language(df)
        if not eng_df.empty:
            fig3 = px.bar(
                eng_df.sort_values("avg_engagement"),
                x="avg_engagement", y="language", orientation="h",
                color="avg_engagement",
                color_continuous_scale=["#1a3a5c", "#2ea043", "#56d364"],
            )
            fig3.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=0, r=0, t=10, b=0),
                height=280,
                coloraxis_showscale=False,
                xaxis_title="Avg Engagement",
                yaxis_title="",
            )
            st.plotly_chart(fig3, use_container_width=True)

    with col_b:
        st.markdown('<div class="section-header">Contributor Tiers</div>', unsafe_allow_html=True)
        tier_df = bk.get_contributor_tier_distribution(df)
        if not tier_df.empty:
            color_map = {
                "solo": "#6e7681",
                "small_team": "#388bfd",
                "mid_team": "#f0883e",
                "large_team": "#56d364",
                "open_source": "#bc8cff",
            }
            fig4 = px.pie(
                tier_df,
                names="tier", values="count",
                color="tier",
                color_discrete_map=color_map,
                hole=0.5,
            )
            fig4.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=0, r=0, t=10, b=0),
                height=280,
                legend=dict(orientation="v", x=1.0),
            )
            st.plotly_chart(fig4, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 2 — PREDICT
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔮 Predict Popularity":
    st.markdown("# 🔮 Predict Repository Popularity")
    st.markdown("Enter a repository's metrics and the model will predict if it will be popular.")
    st.markdown("---")

    if not BACKEND_OK:
        show_error(BACKEND_ERROR); st.stop()

    with st.form("predict_form"):
        st.markdown('<div class="section-header">Repository Metrics</div>', unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            stars = st.number_input("⭐ Stars", min_value=0, max_value=500000, value=120, step=10)
            forks = st.number_input("🍴 Forks", min_value=0, max_value=100000, value=35, step=5)
            issues = st.number_input("🐛 Open Issues", min_value=0, max_value=50000, value=18, step=1)
        with col2:
            pull_requests = st.number_input("🔀 Pull Requests", min_value=0, max_value=50000, value=22, step=1)
            contributors = st.number_input("👥 Contributors", min_value=1, max_value=5000, value=8, step=1)

        submitted = st.form_submit_button("🚀 Analyze Repository")

    if submitted:
        try:
            result = bk.predict_repository_popularity(
                int(stars), int(forks), int(issues), int(pull_requests), int(contributors)
            )
            css_class = "predict-result-popular" if result["prediction"] == 1 else "predict-result-not"
            icon = "✅" if result["prediction"] == 1 else "❌"
            color = "#56d364" if result["prediction"] == 1 else "#f85149"

            st.markdown(f"""
            <div class="{css_class}">
                <h2 style="color:{color};margin:0">{icon} {result['label']}</h2>
                <p style="color:#8b949e;margin:6px 0 16px">Confidence: <strong style="color:{color}">{result['confidence']}%</strong></p>
                <div style="display:flex;gap:24px;flex-wrap:wrap">
                    <div><span style="color:#8b949e;font-size:0.8rem">ENGAGEMENT SCORE</span><br>
                         <span style="font-family:JetBrains Mono,monospace;color:#58a6ff;font-size:1.4rem">{result['engagement_score']}</span></div>
                    <div><span style="color:#8b949e;font-size:0.8rem">PR / ISSUE RATIO</span><br>
                         <span style="font-family:JetBrains Mono,monospace;color:#58a6ff;font-size:1.4rem">{result['pr_to_issue_ratio']}</span></div>
                </div>
            </div>""", unsafe_allow_html=True)

            if result.get("insight"):
                st.markdown(f'<div class="insight-box">💡 {result["insight"]}</div>', unsafe_allow_html=True)

            # Confidence gauge
            st.markdown("<br>", unsafe_allow_html=True)
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=result["confidence"],
                number={"suffix": "%", "font": {"color": color, "size": 40}},
                gauge={
                    "axis": {"range": [0, 100], "tickcolor": "#6e7681"},
                    "bar": {"color": color},
                    "bgcolor": "#161b22",
                    "bordercolor": "#30363d",
                    "steps": [
                        {"range": [0, 40], "color": "#2d1515"},
                        {"range": [40, 70], "color": "#1f2d1a"},
                        {"range": [70, 100], "color": "#0d2818"},
                    ],
                },
                title={"text": "Prediction Confidence", "font": {"color": "#8b949e", "size": 14}},
            ))
            fig.update_layout(
                height=260,
                paper_bgcolor="rgba(0,0,0,0)",
                font_color="#c9d1d9",
                margin=dict(l=30, r=30, t=20, b=10),
            )
            st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            show_error(str(e))


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 3 — LANGUAGE INSIGHTS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📈 Language Insights":
    st.markdown("# 📈 Language Insights")
    st.markdown("Which programming languages dominate GitHub and achieve the highest popularity rates?")
    st.markdown("---")

    if not BACKEND_OK:
        show_error(BACKEND_ERROR); st.stop()

    try:
        df = get_data()
    except FileNotFoundError as e:
        show_error(str(e)); st.stop()

    pop_df = bk.get_popularity_by_language(df)

    if not pop_df.empty:
        st.markdown('<div class="section-header">Popularity Rate by Language (%)</div>', unsafe_allow_html=True)
        fig = px.bar(
            pop_df,
            x="language", y="popularity_rate",
            color="popularity_rate",
            color_continuous_scale=["#0d419d", "#388bfd", "#56d364"],
            text="popularity_rate",
        )
        fig.update_traces(texttemplate="%{text}%", textposition="outside")
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=380,
            margin=dict(l=0, r=0, t=10, b=0),
            coloraxis_showscale=False,
            xaxis_title="",
            yaxis_title="Popularity Rate (%)",
        )
        st.plotly_chart(fig, use_container_width=True)

    # Scatter: Stars vs Forks colored by language
    st.markdown('<div class="section-header">Stars vs Forks (Top 8 Languages)</div>', unsafe_allow_html=True)
    if all(c in df.columns for c in ["stars", "forks", "language"]):
        top8 = df["language"].value_counts().head(8).index.tolist()
        scatter_df = df[(df["language"].isin(top8)) & (df["stars"] < df["stars"].quantile(0.98))]
        fig2 = px.scatter(
            scatter_df.sample(min(3000, len(scatter_df)), random_state=42),
            x="stars", y="forks",
            color="language",
            opacity=0.6,
            hover_data=["contributors"] if "contributors" in df.columns else None,
        )
        fig2.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=380,
            margin=dict(l=0, r=0, t=10, b=0),
        )
        st.plotly_chart(fig2, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 4 — MODEL PERFORMANCE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🧪 Model Performance":
    st.markdown("# 🧪 Model Performance")
    st.markdown("Evaluation results of the trained machine learning model.")
    st.markdown("---")

    if not BACKEND_OK:
        show_error(BACKEND_ERROR); st.stop()

    metrics = get_metrics()

    if not metrics:
        show_error("metrics.json not found. Run train_model.py first.")
        st.stop()

    # Metric KPIs
    st.markdown('<div class="section-header">Model Results</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    kpi_items = [
        (c1, f"{metrics.get('test_accuracy', 0)*100:.1f}%", "Test Accuracy"),
        (c2, f"{metrics.get('roc_auc', 0)*100:.1f}%", "AUC-ROC"),
        (c3, f"{metrics.get('cv_accuracy', 0)*100:.1f}%", "CV Accuracy"),
        (c4, metrics.get('best_model', 'N/A'), "Best Model"),
    ]
    for col, val, label in kpi_items:
        col.markdown(f"""
        <div class="metric-card">
            <div class="metric-value" style="font-size:1.6rem">{val}</div>
            <div class="metric-label">{label}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown('<div class="section-header">Confusion Matrix</div>', unsafe_allow_html=True)
        cm = metrics.get("confusion_matrix", [[0, 0], [0, 0]])
        fig_cm = px.imshow(
            cm,
            text_auto=True,
            color_continuous_scale=["#0d1117", "#388bfd"],
            labels=dict(x="Predicted", y="Actual"),
            x=["Not Popular", "Popular"],
            y=["Not Popular", "Popular"],
        )
        fig_cm.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            height=320,
            margin=dict(l=0, r=0, t=10, b=0),
            coloraxis_showscale=False,
        )
        st.plotly_chart(fig_cm, use_container_width=True)

    with col_right:
        st.markdown('<div class="section-header">Model Comparison</div>', unsafe_allow_html=True)
        models_data = {
            "Model": ["Random Forest", "Gradient Boosting"],
            "Accuracy": [metrics.get("rf_accuracy", 0), metrics.get("gb_accuracy", 0)],
            "AUC-ROC": [metrics.get("rf_auc", 0), metrics.get("gb_auc", 0)],
        }
        comp_df = pd.DataFrame(models_data)
        fig_comp = go.Figure()
        fig_comp.add_trace(go.Bar(
            name="Accuracy", x=comp_df["Model"], y=comp_df["Accuracy"],
            marker_color="#388bfd",
        ))
        fig_comp.add_trace(go.Bar(
            name="AUC-ROC", x=comp_df["Model"], y=comp_df["AUC-ROC"],
            marker_color="#56d364",
        ))
        fig_comp.update_layout(
            barmode="group",
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=320,
            margin=dict(l=0, r=0, t=10, b=0),
            yaxis=dict(range=[0, 1]),
        )
        st.plotly_chart(fig_comp, use_container_width=True)