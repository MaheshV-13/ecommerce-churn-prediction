"""
E-Commerce Churn Prediction - Streamlit Dashboard
==================================================
Purpose: Interactive dashboard for churn prediction and customer segmentation
Input: models/*.pkl, models/*.json, data/processed/*.csv
Pages:
  1. Executive Overview  - Business stakeholder view
  2. Model Diagnostics   - Technical / lecturer view
  3. Prediction Engine   - Interactive demo
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import joblib
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="E-Commerce Churn Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent.parent.parent
DATA_DIR   = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"

# ── Load data (cached) ────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    features = pd.read_csv(
        DATA_DIR / "customers_features.csv",
        parse_dates=['first_purchase_date', 'last_purchase_date']
    )
    segments = pd.read_csv(DATA_DIR / "customer_segments.csv")
    profiles = pd.read_csv(DATA_DIR / "segment_profiles.csv")
    return features, segments, profiles

@st.cache_data
def load_metrics():
    with open(MODELS_DIR / "evaluation_metrics.json") as f:
        return json.load(f)

@st.cache_data
def load_feature_names():
    with open(MODELS_DIR / "feature_names.json") as f:
        return json.load(f)['feature_names']

@st.cache_resource
def load_models():
    lr  = joblib.load(MODELS_DIR / "logistic_regression.pkl")
    rf  = joblib.load(MODELS_DIR / "random_forest.pkl")
    xgb = joblib.load(MODELS_DIR / "xgboost.pkl")
    return lr, rf, xgb

features_df, segments_df, profiles_df = load_data()
metrics       = load_metrics()
feature_names = load_feature_names()
lr_model, rf_model, xgb_model = load_models()

# ── Consistent colour palette ─────────────────────────────────────────────────
SEGMENT_COLORS = {
    "Champions":           "#2ECC71",
    "Loyal Customers":     "#3498DB",
    "Potential Loyalists": "#F39C12",
    "At Risk":             "#E74C3C"
}

# ── Sidebar navigation ────────────────────────────────────────────────────────
st.sidebar.title("📊 Churn Dashboard")
st.sidebar.markdown("**TEB 2043 Data Science**")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Navigate",
    ["📄 Executive Overview", "🔬 Model Diagnostics", "🎯 Prediction Engine"]
)
st.sidebar.markdown("---")
st.sidebar.caption("UCI Online Retail II · 3,463 customers · Jan 2026")

# Page 1: Executive Overview
if page == "📄 Executive Overview":

    st.title("📄 Executive Overview")
    st.markdown("*Business-level summary of customer health and segment strategy.*")
    st.markdown("---")

    # ── KPI Cards ─────────────────────────────────────────────────────────────
    total_customers = len(segments_df)
    churn_rate      = segments_df['churned'].mean()
    revenue_at_risk = segments_df.loc[segments_df['churned'] == 1, 'monetary_gross'].sum()
    active_revenue  = segments_df.loc[segments_df['churned'] == 0, 'monetary_gross'].sum()

    k1, k2, k3, k4 = st.columns(4)
    k1.metric(
    "Total Customers",
    f"{total_customers:,}",
    help="Total number of eligible customers: tenure ≥ 90 days AND frequency ≥ 2 purchases in the observation window."
    )
    k2.metric(
        "Overall Churn Rate",
        f"{churn_rate:.1%}",
        help="Churn is defined as zero purchases in the 6-month outcome window (Jun–Dec 2011). Returns alone do not count as retention."
    )
    k3.metric(
        "Revenue at Risk",
        f"£{revenue_at_risk:,.0f}",
        delta=f"−{churn_rate:.1%} of base",
        delta_color="inverse",
        help="Sum of gross monetary value (observation window purchases) for all customers who churned. Represents the revenue base that failed to return."
    )
    k4.metric(
        "Active Revenue",
        f"£{active_revenue:,.0f}",
        help="Sum of gross monetary value for all customers who made at least one purchase in the outcome window."
    )

    st.markdown("---")

    # ── Row 1: Donut chart + 3D RFM Scatter ───────────────────────────────────
    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Segment Breakdown")
        donut = px.pie(
            profiles_df,
            values='customer_count',
            names='segment_name',
            hole=0.55,
            color='segment_name',
            color_discrete_map=SEGMENT_COLORS
        )
        donut.update_traces(textposition='outside', textinfo='percent+label')
        donut.update_layout(
            showlegend=False,
            margin=dict(t=20, b=20, l=20, r=20),
            height=380
        )
        st.plotly_chart(donut, width='stretch')

    with col2:
        st.subheader("RFM Segment Map (3D Interactive)")
        fig3d = px.scatter_3d(
            segments_df,
            x='recency',
            y='frequency',
            z='monetary_gross',
            color='segment_name',
            color_discrete_map=SEGMENT_COLORS,
            opacity=0.7,
            labels={
                'recency':       'Recency (days)',
                'frequency':     'Frequency',
                'monetary_gross':'Monetary (£)'
            },
            height=420
        )
        fig3d.update_traces(marker=dict(size=3))
        fig3d.update_layout(margin=dict(t=20, b=0, l=0, r=0))
        st.plotly_chart(fig3d, width='stretch')

    st.markdown("---")

    # ── At Risk export ────────────────────────────────────────────────────────────
    at_risk_df = segments_df[segments_df['segment_name'] == 'At Risk'][[
        'customer_id', 'segment_name', 'recency', 'frequency', 'monetary_gross', 'churned'
    ]].copy()
    at_risk_df.columns = [
        'Customer ID', 'Segment', 'Recency (days)', 'Frequency', 'Monetary (£)', 'Churned'
    ]

    @st.cache_data
    def convert_df(df):
        return df.to_csv(index=False).encode('utf-8')

    csv_data = convert_df(at_risk_df)

    st.download_button(
        label=f"⬇️ Export At Risk Customers ({len(at_risk_df):,} records)",
        data=csv_data,
        file_name="at_risk_customers.csv",
        mime="text/csv",
        help="Downloads the full list of 'At Risk' customers with their RFM values. Ready for import into an email CRM or re-engagement campaign tool."
    )

    # ── Row 2: Profiles table + Radar charts ──────────────────────────────────
    col3, col4 = st.columns([1, 1])

    with col3:
        st.subheader("Segment Profiles")

        display_df = profiles_df[[
            'segment_name', 'customer_count', 'churn_rate',
            'recency_mean', 'frequency_mean', 'monetary_mean'
        ]].copy()
        display_df.columns = [
            'Segment', 'Customers', 'Churn Rate',
            'Avg Recency', 'Avg Frequency', 'Avg Monetary (£)'
        ]
        display_df['Churn Rate']       = display_df['Churn Rate'].apply(lambda x: f"{x:.1%}")
        display_df['Avg Recency']      = display_df['Avg Recency'].apply(lambda x: f"{x:.0f}d")
        display_df['Avg Monetary (£)'] = display_df['Avg Monetary (£)'].apply(lambda x: f"£{x:,.0f}")
        display_df['Customers']        = display_df['Customers'].apply(lambda x: f"{x:,}")

        st.dataframe(display_df, width='stretch', hide_index=True)

        st.markdown("**Strategic Recommendations**")
        for _, row in profiles_df.iterrows():
            color = SEGMENT_COLORS.get(row['segment_name'], '#888888')
            st.markdown(
                f"<span style='color:{color}'>●</span> "
                f"**{row['segment_name']}**: {row['recommendation']}",
                unsafe_allow_html=True
            )

    with col4:
        st.subheader("Persona Radar Charts")

        radar_metrics = ['recency_mean', 'frequency_mean', 'monetary_mean', 'churn_rate']
        radar_labels  = ['Recency (inv)', 'Frequency', 'Monetary', 'Churn Risk']

        # Min-max normalise each metric across segments to [0, 1]
        norm = profiles_df[radar_metrics].copy().reset_index(drop=True)
        for col in radar_metrics:
            rng = norm[col].max() - norm[col].min()
            norm[col] = (norm[col] - norm[col].min()) / rng if rng > 0 else 0.5
        # Invert recency: lower days = more recent = better
        norm['recency_mean'] = 1 - norm['recency_mean']

        radar_fig = go.Figure()
        profiles_reset = profiles_df.reset_index(drop=True)

        for i, row in profiles_reset.iterrows():
            vals = [norm.loc[i, c] for c in radar_metrics]
            vals += [vals[0]]  # close the polygon
            radar_fig.add_trace(go.Scatterpolar(
                r=vals,
                theta=radar_labels + [radar_labels[0]],
                fill='toself',
                name=row['segment_name'],
                line_color=SEGMENT_COLORS.get(row['segment_name'], '#888888'),
                opacity=0.6
            ))

        radar_fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            showlegend=True,
            height=420,
            margin=dict(t=30, b=30, l=30, r=30)
        )
        st.plotly_chart(radar_fig, width='stretch')


# Page 2: Model Diagnostics
elif page == "🔬 Model Diagnostics":

    st.title("🔬 Model Diagnostics")
    st.markdown("*Technical validation of model performance for peer and lecturer review.*")
    st.markdown("---")

    model_names = list(metrics.keys())
    display_names = {
        'logistic_regression': 'Logistic Regression',
        'random_forest':        'Random Forest',
        'xgboost':              'XGBoost'
    }

    # ── ROC-AUC metric cards ───────────────────────────────────────────────────
    st.subheader("Test ROC-AUC Scores")
    c1, c2, c3 = st.columns(3)
    card_cols = [c1, c2, c3]

    for i, name in enumerate(model_names):
        cv_auc   = metrics[name]['cv_roc_auc']
        test_auc = metrics[name]['roc_auc']
        delta    = test_auc - cv_auc
        card_cols[i].metric(
        label=display_names[name],
        value=f"{test_auc:.4f}",
        delta=f"CV: {cv_auc:.4f}  ({delta:+.4f})",
        delta_color="normal",
        help="Test ROC-AUC: scored on the held-out 20% test set. CV ROC-AUC: mean across 5 stratified folds during training. A negative delta indicates mild overfitting."
    )

    st.markdown("---")

    # ── Model comparison bar chart ─────────────────────────────────────────────
    st.subheader("Model Comparison")

    bar_data = []
    for name in model_names:
        for metric_key, label in [
            ('roc_auc',   'ROC-AUC'),
            ('f1_score',  'F1-Score'),
            ('precision', 'Precision'),
            ('recall',    'Recall')
        ]:
            bar_data.append({
                'Model':  display_names[name],
                'Metric': label,
                'Score':  metrics[name][metric_key]
            })

    bar_df  = pd.DataFrame(bar_data)
    bar_fig = px.bar(
        bar_df,
        x='Metric',
        y='Score',
        color='Model',
        barmode='group',
        text_auto='.3f',
        color_discrete_sequence=['#3498DB', '#2ECC71', '#E74C3C'],
        height=400
    )
    bar_fig.update_layout(
        yaxis=dict(range=[0.5, 0.85]),
        legend=dict(orientation='h', yanchor='bottom', y=1.02),
        margin=dict(t=40, b=20)
    )
    st.plotly_chart(bar_fig, width='stretch')

    st.markdown("---")

    # ── Confusion matrices ─────────────────────────────────────────────────────
    st.subheader("Confusion Matrices")

    cm_cols = st.columns(3)
    for i, name in enumerate(model_names):
        cm = np.array(metrics[name]['confusion_matrix'])
        cm_fig = px.imshow(
            cm,
            text_auto=True,
            color_continuous_scale='Blues',
            labels=dict(x='Predicted', y='Actual'),
            x=['Not Churned', 'Churned'],
            y=['Not Churned', 'Churned'],
            title=display_names[name],
            height=320
        )
        cm_fig.update_layout(
            coloraxis_showscale=False,
            margin=dict(t=50, b=20, l=20, r=20)
        )
        cm_cols[i].plotly_chart(cm_fig, width='stretch')

    st.markdown("---")

    # ── Feature importance ─────────────────────────────────────────────────────
    st.subheader("Feature Importance (Random Forest — Champion Model)")

    fi_df = pd.DataFrame({
        'Feature':    feature_names,
        'Importance': rf_model.feature_importances_
    }).sort_values('Importance', ascending=True)

    fi_fig = px.bar(
        fi_df,
        x='Importance',
        y='Feature',
        orientation='h',
        text_auto='.3f',
        color='Importance',
        color_continuous_scale='Teal',
        height=520
    )
    fi_fig.update_layout(
        coloraxis_showscale=False,
        margin=dict(t=20, b=20, l=20, r=20)
    )
    st.plotly_chart(fi_fig, width='stretch')


# Page 3: Prediction Engine
elif page == "🎯 Prediction Engine":

    st.title("🎯 Prediction Engine")
    st.markdown("*Live churn probability scored against any customer in the dataset.*")
    st.markdown("---")

    # ── Customer selector ──────────────────────────────────────────────────────
    all_ids     = sorted(segments_df['customer_id'].unique().tolist())
    selected_id = st.selectbox("Select Customer ID", all_ids)

    # ── Pull and prepare customer data ────────────────────────────────────────
    drop_cols    = ['customer_id', 'first_purchase_date', 'last_purchase_date', 'churned']
    customer_row = features_df[features_df['customer_id'] == selected_id]
    X_customer   = customer_row.drop(columns=drop_cols)

    segment_row  = segments_df[segments_df['customer_id'] == selected_id].iloc[0]
    segment_name = segment_row['segment_name']
    actual_churn = int(segment_row['churned'])

    # ── Predictions from all 3 models ─────────────────────────────────────────
    # LR pipeline handles scaling internally — all models receive raw X_customer
    rf_prob  = rf_model.predict_proba(X_customer)[0, 1]
    lr_prob  = lr_model.predict_proba(X_customer)[0, 1]
    xgb_prob = xgb_model.predict_proba(X_customer)[0, 1]

    st.markdown("---")

    # ── Primary: Gauge + customer info ────────────────────────────────────────
    col_gauge, col_info = st.columns([1, 1])

    with col_gauge:
        st.subheader("Churn Risk Gauge")
        gauge_color = (
            "#E74C3C" if rf_prob >= 0.6 else
            "#F39C12" if rf_prob >= 0.4 else
            "#2ECC71"
        )
        gauge_fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=rf_prob * 100,
            number={'suffix': '%', 'font': {'size': 48}},
            delta={'reference': 50, 'suffix': '%'},
            title={'text': "Random Forest (Champion Model)", 'font': {'size': 16}},
            gauge={
                'axis': {'range': [0, 100], 'ticksuffix': '%'},
                'bar':  {'color': gauge_color},
                'steps': [
                    {'range': [0,  40],  'color': '#D5F5E3'},
                    {'range': [40, 60],  'color': '#FCF3CF'},
                    {'range': [60, 100], 'color': '#FADBD8'}
                ],
                'threshold': {
                    'line':      {'color': 'black', 'width': 3},
                    'thickness': 0.75,
                    'value':     50
                }
            }
        ))
        gauge_fig.update_layout(height=320, margin=dict(t=40, b=20, l=30, r=30))
        st.plotly_chart(gauge_fig, width='stretch')

    with col_info:
        st.subheader("Customer Profile")

        seg_color = SEGMENT_COLORS.get(segment_name, '#888888')
        st.markdown(
            f"<h3 style='color:{seg_color}'>● {segment_name}</h3>",
            unsafe_allow_html=True
        )

        verdict = (
            "🔴 High Churn Risk"   if rf_prob >= 0.6 else
            "🟡 Moderate Risk"     if rf_prob >= 0.4 else
            "🟢 Low Churn Risk"
        )
        st.markdown(f"**Risk Assessment:** {verdict}")
        st.markdown(
            f"**Actual Outcome:** "
            f"{'🔴 Churned' if actual_churn else '🟢 Retained'}"
        )

        st.markdown("---")
        st.markdown("**RFM Summary**")
        m1, m2, m3 = st.columns(3)
        m1.metric(
            "Recency",
            f"{segment_row['recency']:.0f}d",
            help="Days since this customer's last transaction as of the observation end date (2011-05-31). Lower = more recently active."
        )
        m2.metric(
            "Frequency",
            f"{segment_row['frequency']:.0f}",
            help="Number of unique invoices (purchase events) in the 18-month observation window. Minimum is 2 due to cohort filter."
        )
        m3.metric(
            "Monetary",
            f"£{segment_row['monetary_gross']:,.0f}",
            help="Gross monetary value: sum of all positive transaction amounts in the observation window. Excludes returns."
        )

        profile_row = profiles_df[profiles_df['segment_name'] == segment_name]
        if not profile_row.empty:
            st.markdown("---")
            st.markdown("**Recommended Action**")
            st.info(profile_row.iloc[0]['recommendation'])

    st.markdown("---")

    # ── Secondary: All 3 models comparison ────────────────────────────────────
    st.subheader("Model Comparison — All Predictions")

    mc1, mc2, mc3 = st.columns(3)
    mc1.metric("🥇 Random Forest",    f"{rf_prob:.1%}", delta="Champion Model")
    mc2.metric("Logistic Regression", f"{lr_prob:.1%}",
               delta=f"{lr_prob - rf_prob:+.1%} vs RF", delta_color="inverse")
    mc3.metric("XGBoost",             f"{xgb_prob:.1%}",
               delta=f"{xgb_prob - rf_prob:+.1%} vs RF", delta_color="inverse")

    compare_fig = px.bar(
        pd.DataFrame({
            'Model':      ['Random Forest', 'Logistic Regression', 'XGBoost'],
            'Churn Prob': [rf_prob, lr_prob, xgb_prob],
            'Champion':   ['Yes', 'No', 'No']
        }),
        x='Model',
        y='Churn Prob',
        color='Champion',
        color_discrete_map={'Yes': '#E74C3C', 'No': '#BDC3C7'},
        text_auto='.1%',
        height=340
    )
    compare_fig.add_hline(
        y=0.5,
        line_dash='dash',
        line_color='black',
        annotation_text='Decision Threshold (50%)'
    )
    compare_fig.update_layout(
        yaxis=dict(range=[0, 1], tickformat='.0%'),
        showlegend=False,
        margin=dict(t=20, b=20)
    )
    st.plotly_chart(compare_fig, width='stretch')

    st.markdown("---")

    # ── Full feature row expander ──────────────────────────────────────────────
    with st.expander("View Full Feature Row for This Customer"):
        st.dataframe(X_customer, width='stretch')