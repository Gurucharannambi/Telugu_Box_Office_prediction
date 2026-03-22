import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (mean_absolute_error, r2_score, mean_squared_error,
                              confusion_matrix, accuracy_score)
import warnings
warnings.filterwarnings("ignore")

# ─── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Telugu Box Office Predictor",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem; font-weight: 800;
        background: linear-gradient(90deg, #f5a623, #e94560);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }
    .sub-title { font-size: 0.95rem; color: #aaaaaa; margin-top: -10px; }
    .metric-box {
        background: #1e1e2e; border-radius: 10px;
        padding: 12px 14px; border-left: 4px solid #f5a623;
        margin-bottom: 8px;
    }
    .metric-label { font-size: 0.72rem; color: #aaa; margin: 0; font-weight: 600; }
    .metric-value { font-size: 1.3rem; font-weight: 800; color: #f5a623; margin: 2px 0; }
    .metric-desc  { font-size: 0.68rem; color: #888; margin: 0; }
    .predict-box {
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        border: 2px solid #f5a623; border-radius: 16px;
        padding: 28px; text-align: center;
    }
    .verdict-badge {
        display: inline-block; padding: 6px 20px;
        border-radius: 20px; font-weight: 700; font-size: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# ─── Data & Model ──────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("telugu_movies.csv")
    return df[df["worldwide_collection_crores"] > 0].copy().reset_index(drop=True)

@st.cache_resource
def train_models(df):
    d = df.copy()
    encoders = {}
    for raw in ["hero", "director", "genre", "release_season"]:
        le = LabelEncoder()
        d[raw + "_enc"] = le.fit_transform(d[raw].astype(str))
        encoders[raw] = le

    d["is_sequel"]    = d["is_sequel"].map({"Yes": 1, "No": 0})
    d["has_big_music"] = d["has_big_music"].map({"Yes": 1, "No": 0})

    feats = ["year","budget_crores","screens","is_sequel","has_big_music",
             "hero_enc","director_enc","genre_enc","release_season_enc"]
    X      = d[feats]
    y_reg  = d["worldwide_collection_crores"]
    y_cls  = d["verdict"]

    X_tr, X_te, yr_tr, yr_te = train_test_split(X, y_reg, test_size=0.2, random_state=42)
    _,    _,    yc_tr, yc_te = train_test_split(X, y_cls, test_size=0.2, random_state=42)

    # Regression
    reg = GradientBoostingRegressor(n_estimators=300, learning_rate=0.08, max_depth=4, random_state=42)
    reg.fit(X_tr, yr_tr)
    yr_pred = reg.predict(X_te)

    metrics = {
        "r2":   round(r2_score(yr_te, yr_pred), 4),
        "mae":  round(mean_absolute_error(yr_te, yr_pred), 2),
        "rmse": round(np.sqrt(mean_squared_error(yr_te, yr_pred)), 2),
        "mse":  round(mean_squared_error(yr_te, yr_pred), 2),
    }

    # Classification (verdict)
    le_v = LabelEncoder()
    yc_tr_e = le_v.fit_transform(yc_tr)
    yc_te_e = le_v.transform(yc_te)
    clf = GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, max_depth=3, random_state=42)
    clf.fit(X_tr, yc_tr_e)
    yc_pred_e = clf.predict(X_te)
    acc = round(accuracy_score(yc_te_e, yc_pred_e) * 100, 1)
    cm  = confusion_matrix(yc_te_e, yc_pred_e)

    # Feature importance
    fi = pd.DataFrame({
        "Feature":    ["Year","Budget","Screens","Is Sequel","Big Music","Hero","Director","Genre","Season"],
        "Importance": reg.feature_importances_
    }).sort_values("Importance", ascending=False)

    # All predictions for comparison page
    all_pred = reg.predict(X)
    avp = pd.DataFrame({
        "Movie":            d["movie_name"].values,
        "Year":             d["year"].values,
        "Hero":             d["hero"].values,
        "Verdict":          d["verdict"].values,
        "Actual (₹ Cr)":    np.round(y_reg.values, 1),
        "Predicted (₹ Cr)": np.round(all_pred, 1),
    })
    avp["Error (₹ Cr)"] = np.round(avp["Actual (₹ Cr)"] - avp["Predicted (₹ Cr)"], 1)
    avp["Accuracy %"]   = (100 - abs(avp["Error (₹ Cr)"]) / avp["Actual (₹ Cr)"] * 100).round(1).clip(0, 100)

    return reg, encoders, le_v, metrics, acc, cm, le_v.classes_, fi, avp, d

def get_verdict(c):
    if c >= 400:  return "🏆 Blockbuster", "#27ae60"
    elif c >= 150: return "✅ Hit",          "#2ecc71"
    elif c >= 80:  return "👍 Average Hit",  "#f39c12"
    elif c >= 40:  return "😐 Average",      "#e67e22"
    else:          return "❌ Flop",          "#e74c3c"

df = load_data()
reg, encoders, le_v, metrics, acc, cm, cls_labels, fi, avp, df2 = train_models(df)

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR — Metrics + Confusion Matrix (always visible)
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🎬 Telugu Box Office")
    page = st.radio("Go to", [
        "🔮 Predict Box Office",
        "📊 Movie Trends",
        "🔍 Actual vs Predicted"
    ], label_visibility="collapsed")

    st.divider()
    st.markdown("### 📐 Model Performance Metrics")

    st.markdown(f"""
    <div class="metric-box">
        <p class="metric-label">R² Score</p>
        <p class="metric-value">{metrics['r2']}</p>
        <p class="metric-desc">Model explains <b>{int(metrics['r2']*100)}%</b> of box office variation.<br>Range: 0 to 1 — higher is better.</p>
    </div>
    <div class="metric-box">
        <p class="metric-label">MAE — Mean Absolute Error</p>
        <p class="metric-value">₹ {metrics['mae']} Cr</p>
        <p class="metric-desc">On average, prediction is off by ₹{metrics['mae']} Cr</p>
    </div>
    <div class="metric-box">
        <p class="metric-label">RMSE — Root Mean Squared Error</p>
        <p class="metric-value">₹ {metrics['rmse']} Cr</p>
        <p class="metric-desc">Punishes large errors more than MAE. Lower = better.</p>
    </div>
    <div class="metric-box">
        <p class="metric-label">MSE — Mean Squared Error</p>
        <p class="metric-value">₹ {metrics['mse']} Cr²</p>
        <p class="metric-desc">Squared average error. Used in model training.</p>
    </div>
    <div class="metric-box">
        <p class="metric-label">Verdict Classifier Accuracy</p>
        <p class="metric-value">{acc}%</p>
        <p class="metric-desc">Correctly predicted Hit / Flop / Blockbuster etc.</p>
    </div>
    """, unsafe_allow_html=True)

    st.divider()
    st.markdown("### 🔲 Confusion Matrix")
    st.caption("Shows how well the model predicts movie verdicts")

    short = {"Blockbuster":"Blkbstr","Hit":"Hit","Average Hit":"AvgHit",
             "Average":"Avg","Below Average":"BelAvg","Flop":"Flop"}
    slabels = [short.get(l, l) for l in cls_labels]

    fig_cm = px.imshow(cm, x=slabels, y=slabels,
                       color_continuous_scale="YlOrRd", text_auto=True,
                       aspect="auto",
                       labels=dict(x="Predicted", y="Actual"))
    fig_cm.update_layout(
        margin=dict(l=0, r=0, t=5, b=0),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white", size=9), coloraxis_showscale=False, height=230
    )
    fig_cm.update_xaxes(tickangle=-30, tickfont=dict(size=8))
    fig_cm.update_yaxes(tickfont=dict(size=8))
    st.plotly_chart(fig_cm, use_container_width=True)
    st.caption("✅ Diagonal = correct  |  ❌ Off-diagonal = wrong")

    st.divider()
    st.caption(f"📽 {len(df)} Telugu movies · 2009–2025")
    st.caption("🤖 Gradient Boosting · scikit-learn")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — PREDICT
# ══════════════════════════════════════════════════════════════════════════════
if page == "🔮 Predict Box Office":
    st.markdown('<h1 class="main-title">🔮 Telugu Box Office Predictor</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">Enter movie details to predict worldwide collection</p>', unsafe_allow_html=True)
    st.divider()

    c1, c2, c3 = st.columns(3)
    with c1:
        st.subheader("🎭 Cast & Crew")
        hero     = st.selectbox("Lead Hero",  sorted(df["hero"].unique()))
        director = st.selectbox("Director",   sorted(df["director"].unique()))
        genre    = st.selectbox("Genre",      sorted(df["genre"].unique()))
    with c2:
        st.subheader("📅 Release Info")
        year      = st.slider("Release Year", 2024, 2030, 2025)
        season    = st.selectbox("Release Season", ["Sankranti","Summer","Independence Day","Dussehra","Christmas","Regular"])
        is_sequel = st.radio("Is it a Sequel?", ["No","Yes"], horizontal=True)
    with c3:
        st.subheader("💰 Production")
        budget    = st.slider("Budget (₹ Crores)", 5, 700, 80)
        screens   = st.slider("Number of Screens", 500, 12000, 3000)
        has_music = st.radio("Has Hit Music?", ["Yes","No"], horizontal=True)

    st.divider()

    if st.button("🎬 Predict Box Office Collection", use_container_width=True, type="primary"):
        def safe_enc(le, v):
            return le.transform([v])[0] if v in list(le.classes_) else 0

        X_in = np.array([[year, budget, screens,
                          1 if is_sequel=="Yes" else 0,
                          1 if has_music=="Yes" else 0,
                          safe_enc(encoders["hero"], hero),
                          safe_enc(encoders["director"], director),
                          safe_enc(encoders["genre"], genre),
                          safe_enc(encoders["release_season"], season)]])


        pred = max(5, round(reg.predict(X_in)[0], 1))
        vlab, vcol = get_verdict(pred)

        _, mid, _ = st.columns([1, 2, 1])
        with mid:
            st.markdown(f"""
            <div class="predict-box">
                <p style="color:#aaa;font-size:0.85rem;margin:0">Predicted Worldwide Collection</p>
                <h1 style="font-size:3rem;margin:8px 0;color:#f5a623">₹ {pred:.1f} Cr</h1>
                <span class="verdict-badge" style="background:{vcol}22;color:{vcol};border:1.5px solid {vcol}">
                    {vlab}
                </span>
                <hr style="border-color:#333;margin:16px 0">
                <div style="display:flex;justify-content:space-around">
                    <div><p style="color:#aaa;font-size:0.8rem;margin:0">Opening Weekend</p>
                         <p style="color:#fff;font-weight:700;font-size:1.1rem">₹ {pred*0.35:.1f} Cr</p></div>
                    <div><p style="color:#aaa;font-size:0.8rem;margin:0">India Net Est.</p>
                         <p style="color:#fff;font-weight:700;font-size:1.1rem">₹ {pred*0.65:.1f} Cr</p></div>
                    <div><p style="color:#aaa;font-size:0.8rem;margin:0">ROI Estimate</p>
                         <p style="color:#fff;font-weight:700;font-size:1.1rem">{((pred/budget)-1)*100:.0f}%</p></div>
                </div>
            </div>""", unsafe_allow_html=True)

        st.divider()
        similar = df[df["genre"]==genre].sort_values("worldwide_collection_crores", ascending=False).head(8)
        if len(similar):
            fig = px.bar(similar, x="movie_name", y="worldwide_collection_crores",
                         color="worldwide_collection_crores", color_continuous_scale="Oranges",
                         title=f"Similar {genre} Movies — for comparison",
                         labels={"movie_name":"Movie","worldwide_collection_crores":"Collection (₹ Cr)"})
            fig.add_hline(y=pred, line_dash="dash", line_color="#e94560",
                          annotation_text=f"Your Prediction: ₹{pred:.0f} Cr", annotation_font_color="#e94560")
            fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                              font_color="white", showlegend=False, xaxis_tickangle=-30)
            st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — TRENDS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Movie Trends":
    st.markdown('<h1 class="main-title">📊 Telugu Movie Trends</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">Explore box office patterns by year, hero, director, genre and season</p>', unsafe_allow_html=True)
    st.divider()

    k1,k2,k3,k4 = st.columns(4)
    k1.metric("🎬 Total Movies",    len(df))
    k2.metric("💰 Highest Grosser", f"₹{df['worldwide_collection_crores'].max():.0f} Cr")
    k3.metric("🏆 Blockbusters",    len(df[df['verdict']=='Blockbuster']))
    k4.metric("📉 Flops",           len(df[df['verdict']=='Flop']))
    st.divider()

    t1,t2,t3,t4 = st.tabs(["📅 Year-wise","🦸 Heroes","🎬 Directors","🎭 Genre & Season"])

    with t1:
        yr = df.groupby("year").agg(
            Total=("worldwide_collection_crores","sum"),
            Avg=("worldwide_collection_crores","mean"),
            Movies=("movie_name","count")).reset_index()
        fig = px.bar(yr, x="year", y="Total", color="Avg", color_continuous_scale="Oranges",
                     title="Total Box Office by Year (₹ Cr)",
                     labels={"Total":"Total Collection (₹ Cr)","year":"Year"})
        fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font_color="white")
        st.plotly_chart(fig, use_container_width=True)
        a,b = st.columns(2)
        with a:
            f2 = px.line(yr, x="year", y="Avg", markers=True, title="Avg Collection Per Movie (₹ Cr)",
                         color_discrete_sequence=["#f5a623"])
            f2.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font_color="white")
            st.plotly_chart(f2, use_container_width=True)
        with b:
            vyr = df.groupby(["year","verdict"]).size().reset_index(name="count")
            f3 = px.bar(vyr, x="year", y="count", color="verdict", barmode="stack",
                        title="Verdict Distribution by Year",
                        color_discrete_map={"Blockbuster":"#27ae60","Hit":"#2ecc71",
                                            "Average Hit":"#f39c12","Average":"#e67e22",
                                            "Below Average":"#e74c3c","Flop":"#c0392b"})
            f3.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font_color="white")
            st.plotly_chart(f3, use_container_width=True)

    with t2:
        hs = df.groupby("hero").agg(Movies=("movie_name","count"),
                                     Total=("worldwide_collection_crores","sum"),
                                     Avg=("worldwide_collection_crores","mean")).reset_index()\
               .sort_values("Total", ascending=False).head(15)
        fh = px.bar(hs, x="hero", y="Total", color="Avg", color_continuous_scale="YlOrRd",
                    title="Top 15 Heroes by Total Box Office (₹ Cr)")
        fh.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                          font_color="white", xaxis_tickangle=-40)
        st.plotly_chart(fh, use_container_width=True)
        fh2 = px.scatter(hs, x="Movies", y="Avg", size="Total", text="hero",
                         color="Total", color_continuous_scale="Oranges",
                         title="Hero Strike Rate — Movies vs Avg Collection")
        fh2.update_traces(textposition="top center")
        fh2.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font_color="white")
        st.plotly_chart(fh2, use_container_width=True)

    with t3:
        ds = df.groupby("director").agg(Movies=("movie_name","count"),
                                         Total=("worldwide_collection_crores","sum"),
                                         Avg=("worldwide_collection_crores","mean")).reset_index()\
               .query("Movies >= 2").sort_values("Avg", ascending=False).head(15)
        fd = px.bar(ds, x="director", y="Avg", color="Movies", color_continuous_scale="Blues",
                    title="Top Directors by Avg Collection — Min 2 Movies")
        fd.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                          font_color="white", xaxis_tickangle=-40)
        st.plotly_chart(fd, use_container_width=True)

    with t4:
        a,b = st.columns(2)
        with a:
            gs = df.groupby("genre")["worldwide_collection_crores"].mean().reset_index()
            gs.columns = ["genre","avg"]
            fg = px.bar(gs.sort_values("avg",ascending=False), x="genre", y="avg",
                        color="avg", color_continuous_scale="Reds", title="Avg Box Office by Genre")
            fg.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                              font_color="white", xaxis_tickangle=-40)
            st.plotly_chart(fg, use_container_width=True)
        with b:
            ss = df.groupby("release_season")["worldwide_collection_crores"].mean().reset_index()
            ss.columns = ["season","avg"]
            fs = px.bar(ss.sort_values("avg",ascending=False), x="season", y="avg",
                        color="avg", color_continuous_scale="Oranges", title="Avg Box Office by Season")
            fs.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font_color="white")
            st.plotly_chart(fs, use_container_width=True)
        fbv = px.scatter(df, x="budget_crores", y="worldwide_collection_crores",
                         color="verdict", size="screens", hover_data=["movie_name","hero","year"],
                         title="Budget vs Collection — Bubble size = No. of Screens",
                         color_discrete_map={"Blockbuster":"#27ae60","Hit":"#2ecc71",
                                             "Average Hit":"#f39c12","Average":"#e67e22",
                                             "Below Average":"#e74c3c","Flop":"#c0392b"})
        fbv.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font_color="white")
        st.plotly_chart(fbv, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — ACTUAL vs PREDICTED
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔍 Actual vs Predicted":
    st.markdown('<h1 class="main-title">🔍 Actual vs Predicted</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">Compare ML predictions against real-world box office results</p>', unsafe_allow_html=True)
    st.divider()

    f1,f2,f3 = st.columns(3)
    with f1: sel_year    = st.multiselect("Filter by Year",    sorted(df["year"].unique(), reverse=True))
    with f2: sel_hero    = st.multiselect("Filter by Hero",    sorted(df["hero"].unique()))
    with f3: sel_verdict = st.multiselect("Filter by Verdict", sorted(df["verdict"].unique()))

    filtered = avp.copy()
    if sel_year:    filtered = filtered[filtered["Year"].isin(sel_year)]
    if sel_hero:    filtered = filtered[filtered["Hero"].isin(sel_hero)]
    if sel_verdict: filtered = filtered[filtered["Verdict"].isin(sel_verdict)]

    s1,s2,s3 = st.columns(3)
    s1.metric("Movies Shown",            len(filtered))
    s2.metric("Avg Prediction Accuracy", f"{filtered['Accuracy %'].mean():.1f}%")
    s3.metric("Mean Error",              f"₹{filtered['Error (₹ Cr)'].abs().mean():.1f} Cr")

    # Bar chart top 20
    top20 = filtered.sort_values("Actual (₹ Cr)", ascending=False).head(20)
    fig_bar = go.Figure()
    fig_bar.add_trace(go.Bar(name="Actual",    x=top20["Movie"], y=top20["Actual (₹ Cr)"],    marker_color="#f5a623"))
    fig_bar.add_trace(go.Bar(name="Predicted", x=top20["Movie"], y=top20["Predicted (₹ Cr)"], marker_color="#e94560"))
    fig_bar.update_layout(barmode="group", title="Actual vs Predicted — Top 20 Movies",
                           plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                           font_color="white", xaxis_tickangle=-45,
                           legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    st.plotly_chart(fig_bar, use_container_width=True)

    # Accuracy scatter
    mx = max(filtered["Actual (₹ Cr)"].max(), filtered["Predicted (₹ Cr)"].max())
    fig_sc = px.scatter(filtered, x="Actual (₹ Cr)", y="Predicted (₹ Cr)",
                        color="Accuracy %", size="Accuracy %",
                        hover_data=["Movie","Hero","Year"],
                        color_continuous_scale="RdYlGn",
                        title="Accuracy Scatter — Greener = More Accurate")
    fig_sc.add_trace(go.Scatter(x=[0,mx], y=[0,mx], mode="lines", name="Perfect Prediction",
                                 line=dict(color="white", dash="dash")))
    fig_sc.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font_color="white")
    st.plotly_chart(fig_sc, use_container_width=True)

    # Feature Importance
    st.subheader("🔑 What Drives Box Office the Most?")
    fig_fi = px.bar(fi, x="Importance", y="Feature", orientation="h",
                    color="Importance", color_continuous_scale="Oranges",
                    title="Feature Importance — Which inputs matter most?")
    fig_fi.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                          font_color="white", yaxis={"categoryorder":"total ascending"})
    st.plotly_chart(fig_fi, use_container_width=True)

    # Table — plain, no matplotlib needed
    st.subheader("📋 Full Comparison Table")
    show = filtered[["Movie","Year","Hero","Verdict",
                      "Actual (₹ Cr)","Predicted (₹ Cr)","Error (₹ Cr)","Accuracy %"]]\
               .sort_values("Actual (₹ Cr)", ascending=False).reset_index(drop=True)
    st.dataframe(show, use_container_width=True, height=480)
