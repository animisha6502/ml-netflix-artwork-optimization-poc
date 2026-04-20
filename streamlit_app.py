"""
Netflix Thumbnail CTR Optimizer — Demo App
Run with: streamlit run streamlit_app.py
"""

import os
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Netflix Thumbnail Optimizer",
    page_icon="🎬",
    layout="wide",
)

# ── Load data ──────────────────────────────────────────────────────────────────
REPO_ROOT   = os.path.dirname(os.path.abspath(__file__))
OUTPUTS_DIR = os.path.join(REPO_ROOT, "outputs")

@st.cache_data
def load_data():
    preds    = pd.read_csv(os.path.join(OUTPUTS_DIR, "netflix_ctr_predictions.csv"))
    summary  = pd.read_csv(os.path.join(OUTPUTS_DIR, "dashboard_summary.csv"))
    best     = pd.read_csv(os.path.join(OUTPUTS_DIR, "best_segment_per_title.csv"))
    return preds, summary, best

try:
    df_preds, df_summary, df_best = load_data()
except FileNotFoundError:
    st.error(
        "⚠️ Output files not found. Please run the notebook first to generate "
        "`outputs/netflix_ctr_predictions.csv` and related files."
    )
    st.stop()

# ── Helpers ────────────────────────────────────────────────────────────────────
SEGMENT_LABELS = {
    "drama_viewer":  "Drama Viewers",
    "action_viewer": "Action Viewers",
    "family_viewer": "Family Viewers",
}
SEGMENT_COLORS = {
    "drama_viewer":  "#E50914",   # Netflix red
    "action_viewer": "#F5A623",   # warm amber
    "family_viewer": "#4A90D9",   # calm blue
}

def segment_label(s):
    return SEGMENT_LABELS.get(s, s)

def ctr_bar_chart(rows):
    """Draw a horizontal bar chart for a single title's segment CTRs."""
    fig, ax = plt.subplots(figsize=(6, 2.4))
    fig.patch.set_facecolor("#141414")
    ax.set_facecolor("#141414")

    segments = ["action_viewer", "drama_viewer", "family_viewer"]
    labels   = [SEGMENT_LABELS[s] for s in segments]
    values   = [rows.loc[rows["user_segment"] == s, "predicted_ctr"].values[0]
                if s in rows["user_segment"].values else 0.0
                for s in segments]
    colors   = [SEGMENT_COLORS[s] for s in segments]

    bars = ax.barh(labels, values, color=colors, height=0.5)
    ax.set_xlim(0, max(values) * 1.35)
    ax.set_xlabel("Predicted Click-Through Rate", color="white", fontsize=9)
    ax.tick_params(colors="white", labelsize=9)
    for spine in ax.spines.values():
        spine.set_visible(False)
    for bar, val in zip(bars, values):
        ax.text(val + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", color="white", fontsize=9)
    plt.tight_layout()
    return fig

def genre_heatmap(df_summary):
    pivot = df_summary.pivot(index="genre", columns="user_segment", values="avg_predicted_ctr")
    pivot = pivot.rename(columns=SEGMENT_LABELS)
    pivot = pivot.sort_values("Drama Viewers", ascending=False)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    fig.patch.set_facecolor("#141414")
    ax.set_facecolor("#141414")

    sns.heatmap(
        pivot, annot=True, fmt=".3f", cmap="YlOrRd",
        linewidths=0.4, linecolor="#2a2a2a",
        ax=ax, cbar_kws={"label": "Avg Predicted CTR"},
    )
    ax.set_title("Avg Predicted CTR by Genre & Viewer Segment", color="white",
                 fontsize=12, fontweight="bold", pad=10)
    ax.set_xlabel("Viewer Segment", color="white", fontsize=10)
    ax.set_ylabel("Genre", color="white", fontsize=10)
    ax.tick_params(colors="white", labelsize=8)
    ax.collections[0].colorbar.ax.yaxis.label.set_color("white")
    ax.collections[0].colorbar.ax.tick_params(colors="white")
    plt.tight_layout()
    return fig

# ── Sidebar ────────────────────────────────────────────────────────────────────
st.sidebar.image(
    "https://upload.wikimedia.org/wikipedia/commons/0/08/Netflix_2015_logo.svg",
    width=120,
)
st.sidebar.markdown("## 🎬 Thumbnail Optimizer")
st.sidebar.markdown(
    "This tool predicts which **viewer segment** is most likely to click on "
    "a Netflix title based on its genre and visual quality signals."
)
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Navigate",
    ["🔍 Title Lookup", "📊 Genre Heatmap", "🏆 Top Picks by Segment"],
)
st.sidebar.markdown("---")
st.sidebar.caption("CMU ML Course · Spring 2026")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — Title Lookup
# ══════════════════════════════════════════════════════════════════════════════
if page == "🔍 Title Lookup":
    st.title("🔍 Title Lookup")
    st.markdown(
        "Search for any Netflix title to see which viewer type is most likely to click — "
        "and why that matters for thumbnail selection."
    )

    all_titles = sorted(df_preds["title"].unique())
    selected   = st.selectbox("Search for a Netflix title", all_titles, index=0)

    rows = df_preds[df_preds["title"] == selected].copy()

    if rows.empty:
        st.warning("No data found for this title.")
    else:
        genre      = rows["genre_bucket"].iloc[0]
        best_seg   = rows.loc[rows["predicted_ctr"].idxmax(), "user_segment"]
        best_ctr   = rows["predicted_ctr"].max()
        worst_seg  = rows.loc[rows["predicted_ctr"].idxmin(), "user_segment"]
        worst_ctr  = rows["predicted_ctr"].min()
        gap        = best_ctr - worst_ctr

        # ── Metric cards ──
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Genre",         genre)
        col2.metric("Best Audience", segment_label(best_seg))
        col3.metric("Peak CTR",      f"{best_ctr:.3f}")
        col4.metric("Segment Gap",   f"{gap:.3f}",
                    delta="Show different thumbnails" if gap > 0.05 else "Similar across audiences",
                    delta_color="normal" if gap > 0.05 else "off")

        st.markdown("---")
        lcol, rcol = st.columns([1.1, 1])

        with lcol:
            st.subheader("Predicted CTR by Viewer Segment")
            fig = ctr_bar_chart(rows)
            st.pyplot(fig, use_container_width=False)

        with rcol:
            st.subheader("What this means")
            best_label  = segment_label(best_seg)
            worst_label = segment_label(worst_seg)

            if gap > 0.08:
                st.success(
                    f"**Strong personalization signal.**  \n"
                    f"*{selected}* resonates most with **{best_label}** "
                    f"(CTR {best_ctr:.3f}) but much less with **{worst_label}** "
                    f"(CTR {worst_ctr:.3f}).  \n\n"
                    f"Netflix should show a **different thumbnail** to each segment — "
                    f"e.g. an action-forward image for action viewers, and an emotional "
                    f"scene for drama viewers."
                )
            elif gap > 0.04:
                st.info(
                    f"**Moderate personalization opportunity.**  \n"
                    f"*{selected}* performs best with **{best_label}**, "
                    f"with a {gap:.3f} gap vs {worst_label}.  \n\n"
                    f"A thumbnail swap could lift CTR for the underperforming segment."
                )
            else:
                st.warning(
                    f"**Broad appeal title.**  \n"
                    f"*{selected}* scores similarly across all viewer types "
                    f"(gap: {gap:.3f}).  \n\n"
                    f"A single thumbnail likely works fine — no personalization needed."
                )

            st.markdown("---")
            st.markdown("**Full breakdown**")
            display = rows[["user_segment", "predicted_ctr"]].copy()
            display["user_segment"]  = display["user_segment"].map(SEGMENT_LABELS)
            display["predicted_ctr"] = display["predicted_ctr"].map(lambda x: f"{x:.4f}")
            display.columns = ["Viewer Segment", "Predicted CTR"]
            st.dataframe(display.reset_index(drop=True), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — Genre Heatmap
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Genre Heatmap":
    st.title("📊 Genre × Viewer Segment Heatmap")
    st.markdown(
        "This heatmap shows the **average predicted CTR** for every genre-segment combination. "
        "Darker cells = higher engagement. Use this to spot which genres have the biggest "
        "personalization gaps."
    )

    fig = genre_heatmap(df_summary)
    st.pyplot(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("Key Takeaways")
    tcol1, tcol2, tcol3 = st.columns(3)

    # Find genre with biggest gap across segments
    pivot = df_summary.pivot(index="genre", columns="user_segment", values="avg_predicted_ctr")
    pivot["gap"] = pivot.max(axis=1) - pivot.min(axis=1)
    top_gap_genre = pivot["gap"].idxmax()
    top_gap_val   = pivot["gap"].max()

    # Find highest overall CTR genre
    top_ctr_genre = pivot.drop(columns="gap").max(axis=1).idxmax()
    top_ctr_val   = pivot.drop(columns="gap").max(axis=1).max()

    # Find most consistent genre (smallest gap)
    stable_genre = pivot["gap"].idxmin()
    stable_val   = pivot["gap"].min()

    tcol1.metric("Biggest Personalization Gap", top_gap_genre, f"gap = {top_gap_val:.3f}")
    tcol2.metric("Highest Overall Engagement",  top_ctr_genre, f"peak CTR = {top_ctr_val:.3f}")
    tcol3.metric("Most Consistent Genre",       stable_genre,  f"gap = {stable_val:.3f}")

    st.markdown("---")
    st.subheader("Browse by Genre")
    selected_genre = st.selectbox("Select a genre", sorted(df_preds["genre_bucket"].unique()))
    genre_titles = (
        df_preds[df_preds["genre_bucket"] == selected_genre]
        .groupby(["title", "user_segment"])["predicted_ctr"]
        .mean()
        .unstack()
        .rename(columns=SEGMENT_LABELS)
        .reset_index()
    )
    genre_titles["Best Audience"] = genre_titles[list(SEGMENT_LABELS.values())].idxmax(axis=1)
    genre_titles = genre_titles.sort_values("Drama Viewers", ascending=False).head(20)
    st.dataframe(genre_titles.reset_index(drop=True), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — Top Picks by Segment
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🏆 Top Picks by Segment":
    st.title("🏆 Top Picks by Viewer Segment")
    st.markdown(
        "For each viewer type, these are the Netflix titles with the highest predicted "
        "click-through rate — useful for thumbnail A/B test prioritization."
    )

    tabs = st.tabs(["Action Viewers", "Drama Viewers", "Family Viewers"])

    for tab, seg_key in zip(tabs, ["action_viewer", "drama_viewer", "family_viewer"]):
        with tab:
            seg_label = SEGMENT_LABELS[seg_key]
            top = (
                df_preds[df_preds["user_segment"] == seg_key]
                .sort_values("predicted_ctr", ascending=False)
                .drop_duplicates("title")
                .head(15)[["title", "genre_bucket", "predicted_ctr"]]
                .reset_index(drop=True)
            )
            top.index += 1
            top.columns = ["Title", "Genre", "Predicted CTR"]
            top["Predicted CTR"] = top["Predicted CTR"].map(lambda x: f"{x:.4f}")

            st.markdown(f"**Top 15 titles most likely to be clicked by {seg_label}**")
            st.dataframe(top, use_container_width=True)

            # Mini bar chart — top 10
            top_num = (
                df_preds[df_preds["user_segment"] == seg_key]
                .sort_values("predicted_ctr", ascending=False)
                .drop_duplicates("title")
                .head(10)
            )
            fig, ax = plt.subplots(figsize=(7, 3.5))
            fig.patch.set_facecolor("#141414")
            ax.set_facecolor("#141414")
            color = SEGMENT_COLORS[seg_key]
            ax.barh(
                top_num["title"].str[:35].tolist()[::-1],
                top_num["predicted_ctr"].tolist()[::-1],
                color=color, height=0.6,
            )
            ax.set_xlabel("Predicted CTR", color="white", fontsize=9)
            ax.tick_params(colors="white", labelsize=8)
            for spine in ax.spines.values():
                spine.set_visible(False)
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
