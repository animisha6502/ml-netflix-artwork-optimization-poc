"""
Netflix Thumbnail CTR Optimizer — Final Version
Features:
  1. Thumbnail Scorer  — upload any image, PIL scoring + local semantic analysis
                         (OpenCV face detection + VADER sentiment), adjusted AVA
                         score, predicted CTR, and an explanation
  2. TMDB Image Ranker — fetch all posters for a title, score + rank by CTR
  3. A/B Simulator     — compare two titles head-to-head, project monthly clicks

Run:
  export TMDB_API_KEY=your_tmdb_key
  streamlit run streamlit_app_final.py
  (No LLM API key needed — semantic analysis runs fully locally.)
"""

import os, io, json, requests
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter
import cv2 as _cv2

# ── NLTK / VADER (local, no API) ─────────────────────────────────────────────
try:
    import nltk as _nltk
    _nltk.download("vader_lexicon", quiet=True)
    from nltk.sentiment import SentimentIntensityAnalyzer as _SIA
    _VADER = _SIA()
    _NLP_OK = True
except Exception:
    _NLP_OK = False

_FACE_CASCADE = _cv2.CascadeClassifier(
    _cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Netflix Thumbnail Optimizer",
    page_icon="🎬",
    layout="wide",
)

# ── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  .stApp, [data-testid="stAppViewContainer"] { background:#141414; }
  [data-testid="stSidebar"] { background:#0d0d0d; border-right:1px solid #1e1e1e; }

  h1  { color:#ffffff !important; font-size:28px !important; font-weight:700 !important; margin-bottom:4px !important; }
  h4  { color:#ffffff !important; font-size:16px !important; font-weight:600 !important; }
  p, li, span, div { color:#cccccc; }
  .stMarkdown p { color:#888888; font-size:14px; }

  [data-testid="metric-container"] {
    background:#1e1e1e; border:1px solid #2a2a2a; border-radius:8px; padding:16px 20px;
  }
  [data-testid="stMetricLabel"] > div { color:#888 !important; font-size:11px !important;
    text-transform:uppercase; letter-spacing:.8px; }
  [data-testid="stMetricValue"]       { color:#fff !important; font-size:24px !important;
    font-weight:700 !important; }
  [data-testid="stMetricDelta"] > div { font-size:12px !important; }

  .stButton > button {
    background:#E50914 !important; color:#fff !important;
    border:none; border-radius:6px; font-weight:700; padding:10px 28px; font-size:14px;
  }
  .stButton > button:hover { background:#b2070f !important; }

  div[data-baseweb="select"] > div           { background:#1e1e1e !important; color:#fff;
    border-color:#2a2a2a !important; }
  div[data-baseweb="select"] [role="option"] { background:#1e1e1e; }

  .stTabs [data-baseweb="tab-list"] { background:#1e1e1e; border-radius:8px; gap:4px; padding:4px; }
  .stTabs [data-baseweb="tab"]      { color:#888; border-radius:6px; padding:8px 18px; }
  .stTabs [aria-selected="true"]    { background:#2a2a2a !important; color:#fff !important; }

  .stDataFrame thead th { background:#252525 !important; color:#888 !important;
    font-size:11px; text-transform:uppercase; }
  .stDataFrame tbody td { color:#ffffff !important; font-size:13px; }
  .stDataFrame          { border:1px solid #2a2a2a; border-radius:8px; }

  [data-testid="stFileUploaderDropzone"] {
    background:#1e1e1e; border:2px dashed #2a2a2a; border-radius:10px;
  }
  .llm-card {
    background:#1a1a2e; border:1px solid #2a2a4a; border-radius:10px;
    padding:18px 20px; margin-top:12px;
  }
  .llm-tag {
    display:inline-block; background:#2a2a4a; color:#a0a0ff;
    border-radius:4px; padding:2px 8px; font-size:11px; margin:3px 2px;
    font-weight:600; letter-spacing:.4px;
  }
  .llm-tag.positive { background:#1a3a2a; color:#46D369; }
  .llm-tag.negative { background:#3a1a1a; color:#E50914; }
  .llm-tag.neutral  { background:#2a2a2a; color:#888888; }
  .adj-pill {
    display:inline-block; border-radius:20px; padding:4px 14px;
    font-size:13px; font-weight:700; margin-left:8px;
  }
  hr { border-color:#2a2a2a !important; margin:20px 0 !important; }
  #MainMenu, footer, header { visibility:hidden; }
</style>
""", unsafe_allow_html=True)


# ── Data ─────────────────────────────────────────────────────────────────────
REPO_ROOT   = os.path.dirname(os.path.abspath(__file__))
OUTPUTS_DIR = os.path.join(REPO_ROOT, "outputs")

@st.cache_data
def load_data():
    preds   = pd.read_csv(os.path.join(OUTPUTS_DIR, "netflix_ctr_predictions.csv"))
    summary = pd.read_csv(os.path.join(OUTPUTS_DIR, "dashboard_summary.csv"))
    return preds, summary

try:
    df_preds, df_summary = load_data()
except FileNotFoundError:
    st.error("Output files not found. Run the training notebook first.")
    st.stop()


# ── Config ───────────────────────────────────────────────────────────────────
API_URL  = "http://54.209.170.252:5000/predict"
TMDB_KEY = os.environ.get("TMDB_API_KEY", "")
LLM_OK   = _NLP_OK   # always True when nltk + opencv are installed

SEGS = {
    "action_viewer": "Action Viewers",
    "drama_viewer":  "Drama Viewers",
    "family_viewer": "Family Viewers",
}
COLORS = {
    "action_viewer": "#F5A623",
    "drama_viewer":  "#E50914",
    "family_viewer": "#4A90D9",
}
GENRES = sorted(df_preds["genre_bucket"].unique().tolist())

# Genre-level average AVA scores (from training notebook)
GENRE_AVA = {
    "Action & Adventure":    5.498,
    "Children & Family Movies": 5.380,
    "Comedies":              5.241,
    "Documentaries":         5.408,
    "Dramas":                5.465,
    "Independent Movies":    5.341,
    "International Movies":  5.472,
    "Romantic Movies":       5.391,
    "Thrillers":             5.332,
}


# ══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

# ── PIL aesthetic scoring ─────────────────────────────────────────────────────
def score_image(img: Image.Image) -> tuple:
    """
    Compute AVA aesthetic score (1–10) from raw pixel statistics.
    Returns (ava_score, breakdown_dict).
    """
    arr  = np.array(img.convert("RGB")).astype(float)
    gray = 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]

    # Brightness — peaks at 50% luminance
    b_norm  = arr.mean() / 255
    b_score = max(0.0, 1 - 2 * abs(b_norm - 0.5))

    # Contrast — std of luminance
    c_score = min(gray.std() / 128, 1.0)

    # Colorfulness (Hasler & Süsstrunk, 2003)
    R, G, B = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    rg, yb  = R - G, 0.5 * (R + G) - B
    cf      = (np.sqrt(rg.std() ** 2 + yb.std() ** 2)
               + 0.3 * np.sqrt(rg.mean() ** 2 + yb.mean() ** 2))
    col_score = min(cf / 100, 1.0)

    # Sharpness — Laplacian edge energy
    edges       = img.convert("L").filter(ImageFilter.FIND_EDGES)
    sharp_score = min(np.array(edges).std() / 30, 1.0)

    raw       = 0.30 * b_score + 0.30 * c_score + 0.20 * col_score + 0.20 * sharp_score
    ava_score = round(1 + raw * 9, 2)   # AVA scale: 1–10

    breakdown = {
        "Brightness":   round(b_score * 10, 1),
        "Contrast":     round(c_score * 10, 1),
        "Colorfulness": round(col_score * 10, 1),
        "Sharpness":    round(sharp_score * 10, 1),
    }
    return ava_score, breakdown


def score_from_url(url: str):
    """Download image from URL and run PIL scoring. Returns ((ava_score, breakdown), img) or (None, None)."""
    try:
        resp = requests.get(url, timeout=8)
        img  = Image.open(io.BytesIO(resp.content)).convert("RGB")
        return score_image(img), img
    except Exception:
        return None, None


# ── Local semantic analysis (OpenCV + VADER, no API) ─────────────────────────

# Genre → base emotional tone mapping
_GENRE_EMOTION = {
    "Action & Adventure":       "intense",
    "Thrillers":                "intense",
    "Dramas":                   "neutral",
    "Comedies":                 "joyful",
    "Children & Family Movies": "warm",
    "Romantic Movies":          "warm",
    "Documentaries":            "neutral",
    "Independent Movies":       "neutral",
    "International Movies":     "neutral",
}

@st.cache_data(show_spinner=False)
def analyze_thumbnail_llm(img_bytes: bytes, genre: str, segment: str,
                          title: str = "") -> tuple:
    """
    Local semantic analysis — no API calls, fully offline.

    Signals derived from:
      • OpenCV Haar-cascade face detection   → face_present, face_count
      • PIL pixel statistics                 → color_mood
      • Canny edge density                   → text_presence (proxy)
      • Face position vs rule-of-thirds      → composition
      • Genre mapping + VADER on title       → emotion
    Returns (signals_dict, explanation_str).
    """
    try:
        arr  = np.array(Image.open(io.BytesIO(img_bytes)).convert("RGB"))
        gray = _cv2.cvtColor(arr, _cv2.COLOR_RGB2GRAY)
        h, w = arr.shape[:2]

        # ── 1. Face detection ────────────────────────────────────────────────
        # Pass 1: frontal faces — permissive params catch mid-distance subjects
        faces = _FACE_CASCADE.detectMultiScale(
            gray, scaleFactor=1.05, minNeighbors=3, minSize=(20, 20)
        )
        # Pass 2: profile faces as fallback
        if len(faces) == 0:
            _profile = _cv2.CascadeClassifier(
                _cv2.data.haarcascades + "haarcascade_profileface.xml"
            )
            faces = _profile.detectMultiScale(
                gray, scaleFactor=1.05, minNeighbors=3, minSize=(20, 20)
            )
        # Pass 3: HOG full-body detector — catches performers / standing figures
        if len(faces) == 0:
            hog = _cv2.HOGDescriptor()
            hog.setSVMDetector(_cv2.HOGDescriptor_getDefaultPeopleDetector())
            people, weights = hog.detectMultiScale(
                arr, winStride=(8, 8), padding=(4, 4), scale=1.05
            )
            if len(people) > 0:
                # Apply Non-Maximum Suppression to collapse overlapping boxes
                # (HOG often emits 3-5 overlapping rects for the same person)
                boxes = [[x, y, x + w, y + h] for (x, y, w, h) in people]
                boxes_np = np.array(boxes, dtype=float)
                scores_np = np.array(weights, dtype=float).flatten()

                x1, y1, x2, y2 = boxes_np[:, 0], boxes_np[:, 1], boxes_np[:, 2], boxes_np[:, 3]
                areas = (x2 - x1 + 1) * (y2 - y1 + 1)
                order = scores_np.argsort()[::-1]
                keep  = []
                while order.size > 0:
                    i = order[0]
                    keep.append(i)
                    xx1 = np.maximum(x1[i], x1[order[1:]])
                    yy1 = np.maximum(y1[i], y1[order[1:]])
                    xx2 = np.minimum(x2[i], x2[order[1:]])
                    yy2 = np.minimum(y2[i], y2[order[1:]])
                    inter = np.maximum(0, xx2 - xx1 + 1) * np.maximum(0, yy2 - yy1 + 1)
                    iou   = inter / (areas[i] + areas[order[1:]] - inter)
                    order = order[np.where(iou <= 0.45)[0] + 1]

                faces = people[keep]   # deduplicated detections

        n_faces = len(faces)
        face_present = n_faces > 0
        face_count   = "none" if n_faces == 0 else "one" if n_faces == 1 else "multiple"

        # ── 2. Color mood ────────────────────────────────────────────────────
        arr_f  = arr.astype(float)
        r_mean = arr_f[:, :, 0].mean()
        b_mean = arr_f[:, :, 2].mean()
        bright = arr_f.mean()
        rg     = arr_f[:, :, 0] - arr_f[:, :, 1]
        yb     = 0.5 * (arr_f[:, :, 0] + arr_f[:, :, 1]) - arr_f[:, :, 2]
        colorfulness = (np.sqrt(rg.std()**2 + yb.std()**2)
                        + 0.3 * np.sqrt(rg.mean()**2 + yb.mean()**2))
        if bright < 55:
            color_mood = "dark"
        elif colorfulness > 50:
            color_mood = ("warm" if r_mean > b_mean + 15
                          else "cool" if b_mean > r_mean + 15 else "vibrant")
        else:
            color_mood = "muted"

        # ── 3. Text presence ──────────────────────────────────────────────────
        # Strategy: thumbnails put text in the bottom third. Check there first.
        # Large title text creates dense, uniform horizontal edges in that region.
        bottom = gray[int(h * 0.6):, :]                        # bottom 40%
        edges_bottom = _cv2.Canny(bottom, 30, 120)
        bottom_density = edges_bottom.mean() / 255.0

        # Also check full-image edge density for overlay text spread across frame
        edges_full  = _cv2.Canny(gray, 50, 150)
        full_density = edges_full.mean() / 255.0

        # Heavy: dense edges in bottom region OR high density across full image
        if bottom_density > 0.08 or full_density > 0.10:
            text_presence = "heavy"
        elif bottom_density > 0.04 or full_density > 0.05:
            text_presence = "minimal"
        else:
            text_presence = "none"

        # ── 4. Composition ───────────────────────────────────────────────────
        if face_present:
            # Strong if any face falls near a rule-of-thirds vertical line
            third_xs = {w // 3, 2 * w // 3}
            near_third = any(
                abs((fx + fw // 2) - tx) < w // 6
                for (fx, fy, fw, fh) in faces
                for tx in third_xs
            )
            composition = "strong" if near_third else "moderate"
        else:
            # No face — assess asymmetry: balanced asymmetry = intentional design
            left_bright  = gray[:, : w // 2].mean()
            right_bright = gray[:, w // 2 :].mean()
            top_bright   = gray[: h // 2, :].mean()
            bot_bright   = gray[h // 2 :, :].mean()
            asym = max(abs(left_bright - right_bright), abs(top_bright - bot_bright))
            composition = "strong" if asym > 20 else "moderate" if asym > 8 else "weak"

        # ── 5. Emotion: genre base + VADER refinement on title ────────────────
        emotion = _GENRE_EMOTION.get(genre, "neutral")
        if title and _NLP_OK:
            vs       = _VADER.polarity_scores(title)
            compound = vs["compound"]
            if compound >= 0.4 and emotion not in ("intense",):
                emotion = "joyful" if emotion == "warm" else "warm"
            elif compound <= -0.4 and emotion not in ("warm", "joyful"):
                emotion = "intense" if emotion == "neutral" else "mysterious"

        # ── 6. Template explanation ──────────────────────────────────────────
        face_desc  = (
            "A face is" if face_present and face_count == "one"
            else "Multiple faces are" if face_count == "multiple"
            else "No faces are"
        )
        comp_desc  = {
            "strong":   "strong composition with a clear focal point",
            "moderate": "moderate composition",
            "weak":     "weak composition lacking a clear subject",
        }.get(composition, "moderate composition")
        text_desc  = {
            "none":    "no text overlay keeps focus on the visuals",
            "minimal": "minimal text overlay",
            "heavy":   "heavy text overlay may draw attention away from the image",
        }.get(text_presence, "minimal text overlay")
        seg_label  = SEGS.get(segment, segment)
        tone_fit   = {
            "action_viewer": {"intense": "strong", "mysterious": "good",
                              "neutral": "moderate", "warm": "weak", "joyful": "weak"},
            "drama_viewer":  {"warm": "strong", "mysterious": "strong",
                              "intense": "good", "neutral": "moderate", "joyful": "moderate"},
            "family_viewer": {"joyful": "strong", "warm": "strong",
                              "neutral": "moderate", "mysterious": "weak", "intense": "weak"},
        }.get(segment, {}).get(emotion, "moderate")

        explanation = (
            f"{face_desc} detected in this thumbnail, with {comp_desc} and {text_desc}. "
            f"The {color_mood} color palette signals a {emotion} tone — a {tone_fit} fit "
            f"for {seg_label}."
        )

        signals = {
            "face_present":   face_present,
            "face_count":     face_count,
            "emotion":        emotion,
            "text_presence":  text_presence,
            "composition":    composition,
            "color_mood":     color_mood,
        }
        return signals, explanation

    except Exception as e:
        return None, f"__error__:{type(e).__name__}: {str(e)[:300]}"


def compute_llm_adjustment(signals: dict, segment: str) -> float:
    """
    Convert local semantic signals into a deterministic AVA score adjustment.
    Rules are transparent and segment-aware. Clamped to ±1.5.
    """
    adj = 0.0

    # 1. Face presence — faces strongly drive click decisions
    if signals.get("face_present"):
        adj += 0.4
    if signals.get("face_count") == "multiple":
        adj += 0.1   # ensemble cast thumbnails perform slightly better

    # 2. Emotion × viewer segment affinity
    emotion_map = {
        "action_viewer": {
            "intense":    +0.40,
            "mysterious": +0.25,
            "joyful":     -0.10,
            "warm":       -0.10,
            "neutral":     0.00,
        },
        "drama_viewer": {
            "warm":       +0.30,
            "mysterious": +0.25,
            "intense":    +0.15,
            "joyful":     +0.10,
            "neutral":     0.00,
        },
        "family_viewer": {
            "joyful":     +0.40,
            "warm":       +0.35,
            "neutral":     0.00,
            "mysterious": -0.15,
            "intense":    -0.30,
        },
    }
    adj += emotion_map.get(segment, {}).get(signals.get("emotion", "neutral"), 0.0)

    # 3. Text overlay — heavy text reduces visual appeal for streaming thumbnails
    text_adj = {"none": +0.10, "minimal": 0.00, "heavy": -0.50}
    adj += text_adj.get(signals.get("text_presence", "minimal"), 0.00)

    # 4. Composition quality
    comp_adj = {"strong": +0.40, "moderate": 0.00, "weak": -0.40}
    adj += comp_adj.get(signals.get("composition", "moderate"), 0.00)

    return round(max(-1.5, min(1.5, adj)), 2)


def img_to_bytes(img: Image.Image, quality: int = 85) -> bytes:
    """Convert PIL image to JPEG bytes for LLM transmission."""
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


# ── EC2 API ───────────────────────────────────────────────────────────────────
def live_ctr(genre: str, seg: str, ava_score: float = None,
             is_us: int = 1, is_movie: int = 1, decade: float = 2010):
    payload = {
        "genre_bucket":   genre,
        "user_segment":   seg,
        "is_us":          is_us,
        "is_movie":       is_movie,
        "release_decade": decade,
    }
    if ava_score is not None:
        payload["ava_score"] = ava_score
    try:
        r = requests.post(API_URL, json=payload, timeout=5)
        return r.json().get("predicted_ctr")
    except Exception:
        return None


def fallback_ctr(genre: str, seg: str, ava_score: float = None) -> float:
    """Use precomputed CSV when EC2 is unavailable. Optionally scale by AVA."""
    rows = df_preds[
        (df_preds["genre_bucket"] == genre) &
        (df_preds["user_segment"] == seg)
    ]["predicted_ctr"]
    base = float(rows.mean()) if not rows.empty else 0.30
    if ava_score is not None:
        base *= ava_score / GENRE_AVA.get(genre, 5.4)
    return round(base, 4)


# ── TMDB ─────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600)
def tmdb_posters(title: str):
    """Returns list of {url, path} dicts (up to 6 posters)."""
    if not TMDB_KEY:
        return []
    try:
        results = requests.get(
            "https://api.themoviedb.org/3/search/multi",
            params={"api_key": TMDB_KEY, "query": title}, timeout=5,
        ).json().get("results", [])
        if not results:
            return []
        item   = results[0]
        mtype  = item.get("media_type", "movie")
        images = requests.get(
            f"https://api.themoviedb.org/3/{mtype}/{item['id']}/images",
            params={"api_key": TMDB_KEY}, timeout=5,
        ).json().get("posters", [])
        return [
            {"url": f"https://image.tmdb.org/t/p/w300{p['file_path']}", "path": p["file_path"]}
            for p in images[:6]
        ]
    except Exception:
        return []


# ── Charting ──────────────────────────────────────────────────────────────────
def seg_bar(data: dict, highlight: str = None, height: float = 2.2):
    """Horizontal bar chart of CTR by segment."""
    fig, ax = plt.subplots(figsize=(6, height))
    fig.patch.set_facecolor("#141414")
    ax.set_facecolor("#141414")
    seg_order = ["action_viewer", "drama_viewer", "family_viewer"]
    labels    = [SEGS[s] for s in seg_order]
    values    = [data.get(s, 0.0) for s in seg_order]
    colors    = [("#ffffff" if s == highlight else COLORS[s]) for s in seg_order]
    bars      = ax.barh(labels, values, color=colors, height=0.45)
    ax.set_xlim(0, max(values) * 1.3 if max(values) > 0 else 0.5)
    ax.set_xlabel("Predicted CTR", color="#555", fontsize=8)
    ax.tick_params(colors="#888", labelsize=9)
    for spine in ax.spines.values():
        spine.set_visible(False)
    for bar, val in zip(bars, values):
        ax.text(val + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", color="white", fontsize=9, fontweight="bold")
    plt.tight_layout(pad=0.5)
    return fig


def score_bar_html(label: str, value: float, max_val: float = 10.0) -> str:
    pct = value / max_val * 100
    col = "#46D369" if pct > 66 else "#F5A623" if pct > 33 else "#E50914"
    return f"""
    <div style="margin-bottom:10px">
      <div style="display:flex;justify-content:space-between;margin-bottom:4px">
        <span style="color:#888;font-size:12px">{label}</span>
        <span style="color:#fff;font-size:12px;font-weight:700">{value}/10</span>
      </div>
      <div style="background:#2a2a2a;border-radius:4px;height:6px">
        <div style="background:{col};width:{pct:.1f}%;height:6px;border-radius:4px"></div>
      </div>
    </div>"""


def llm_tag(label: str, value: str, sentiment: str = "neutral") -> str:
    return (f'<span class="llm-tag {sentiment}">'
            f'<span style="color:#666;font-size:10px">{label}: </span>{value}'
            f'</span>')


# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.image(
    "https://upload.wikimedia.org/wikipedia/commons/0/08/Netflix_2015_logo.svg",
    width=100,
)
st.sidebar.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
st.sidebar.markdown(
    "<span style='color:#fff;font-size:15px;font-weight:700'>Thumbnail Optimizer</span>",
    unsafe_allow_html=True,
)
st.sidebar.markdown(
    "<span style='color:#555;font-size:12px'>Personalized artwork CTR prediction</span>",
    unsafe_allow_html=True,
)
st.sidebar.markdown("---")

page = st.sidebar.radio("", [
    "🖼️  Thumbnail Scorer",
    "🎬  TMDB Image Ranker",
    "🧪  A/B Simulator",
], label_visibility="collapsed")

st.sidebar.markdown("---")

# LLM status indicator
if LLM_OK:
    st.sidebar.markdown(
        "<div style='background:#1a3a2a;border:1px solid #2a4a3a;border-radius:6px;"
        "padding:8px 12px;font-size:11px'>"
        "<span style='color:#46D369;font-weight:700'>● Local Analysis Active</span><br>"
        "<span style='color:#555'>OpenCV + VADER · no API needed</span></div>",
        unsafe_allow_html=True,
    )
else:
    st.sidebar.markdown(
        "<div style='background:#2a2a2a;border:1px solid #333;border-radius:6px;"
        "padding:8px 12px;font-size:11px'>"
        "<span style='color:#888;font-weight:700'>○ Local Analysis Unavailable</span><br>"
        "<span style='color:#555'>pip install nltk opencv-python</span></div>",
        unsafe_allow_html=True,
    )

st.sidebar.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
# st.sidebar.markdown(
#     "<span style='color:#444;font-size:11px'>CMU 95-828 · Spring 2026 · Team 27</span>",
#     unsafe_allow_html=True,
# )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — THUMBNAIL SCORER
# ══════════════════════════════════════════════════════════════════════════════
if page == "🖼️  Thumbnail Scorer":
    st.title("Thumbnail Scorer")
    st.markdown(
        "Upload any thumbnail to get an aesthetic quality score, local semantic "
        "analysis (face detection + sentiment), and a predicted CTR per viewer segment — before the image goes live."
    )
    st.markdown("---")

    # ── Inputs ──────────────────────────────────────────────────────────────
    c1, c2, c3 = st.columns([1.5, 1, 1])
    with c1:
        genre_s = st.selectbox("Genre", GENRES, key="ts_genre")
    with c2:
        seg_s = st.selectbox(
            "Primary viewer segment", list(SEGS.keys()),
            format_func=lambda s: SEGS[s], key="ts_seg",
        )
    with c3:
        is_movie_s = st.toggle("Movie (not TV show)", value=True, key="ts_movie")

    uploaded = st.file_uploader(
        "Upload thumbnail", type=["jpg", "jpeg", "png", "webp"],
        label_visibility="collapsed",
    )

    if not uploaded:
        st.markdown(
            "<div style='background:#1e1e1e;border:1px dashed #2a2a2a;border-radius:10px;"
            "padding:40px;text-align:center;color:#444;font-size:14px'>"
            "Drop a thumbnail here to analyse it</div>",
            unsafe_allow_html=True,
        )
        st.stop()

    img           = Image.open(uploaded).convert("RGB")
    ava_raw, bk   = score_image(img)
    genre_avg     = GENRE_AVA.get(genre_s, 5.4)
    img_bytes_val = img_to_bytes(img)

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

    # ── Run LLM (if available) ───────────────────────────────────────────────
    signals, explanation = None, None
    ava_adj              = 0.0
    llm_ran              = False

    llm_error_msg = None
    if LLM_OK:
        with st.spinner("Analyzing thumbnail locally…"):
            signals, explanation = analyze_thumbnail_llm(img_bytes_val, genre_s, seg_s)
        if signals:
            ava_adj = compute_llm_adjustment(signals, seg_s)
            llm_ran = True
        elif isinstance(explanation, str) and explanation.startswith("__error__:"):
            llm_error_msg = explanation.replace("__error__:", "")
            explanation   = None

    ava_final = round(min(9.0, max(2.0, ava_raw + ava_adj)), 2)

    # ── Layout: image | scores | CTR ─────────────────────────────────────────
    img_col, score_col, ctr_col = st.columns([1, 1.3, 1.4])

    with img_col:
        st.image(img, caption="Uploaded thumbnail", use_container_width=True)

    with score_col:
        st.markdown("#### Aesthetic Score")

        quality = ("Excellent" if ava_final >= 7 else "Good" if ava_final >= 5.5
                   else "Average" if ava_final >= 4 else "Poor")
        qcolor  = ("#46D369" if ava_final >= 7 else "#F5A623" if ava_final >= 5.5
                   else "#888" if ava_final >= 4 else "#E50914")

        # Adjustment pill
        if llm_ran and ava_adj != 0:
            adj_color = "#46D369" if ava_adj > 0 else "#E50914"
            adj_label = f"+{ava_adj}" if ava_adj > 0 else str(ava_adj)
            adj_html  = (
                f'<span class="adj-pill" style="background:{adj_color}22;'
                f'color:{adj_color};border:1px solid {adj_color}44">'
                f'AI adj {adj_label}</span>'
            )
        else:
            adj_html = ""

        # Build HTML as a flat string — multiline f-strings break Streamlit's
        # markdown parser (newlines inside divs render as text, not HTML)
        score_card = (
            f'<div style="background:#1e1e1e;border:1px solid #2a2a2a;border-radius:8px;padding:16px 20px;margin-bottom:16px">'
            f'<div style="color:{qcolor};font-size:42px;font-weight:700;text-align:center">{ava_final}</div>'
            f'<div style="color:#888;font-size:12px;text-align:center;margin-top:4px">'
            f'out of 10 &nbsp;·&nbsp;<span style="color:{qcolor}">{quality}</span>{adj_html}</div>'
            f'<div style="color:#555;font-size:11px;text-align:center;margin-top:6px">'
            f'PIL raw: {ava_raw} &nbsp;|&nbsp; Genre avg: {genre_avg:.2f}</div>'
            f'</div>'
        )
        bars_html = "".join(score_bar_html(k, v) for k, v in bk.items())
        st.markdown(score_card + bars_html, unsafe_allow_html=True)

    with ctr_col:
        st.markdown("#### Predicted CTR by Segment")
        ctrs   = {}
        api_ok = False
        for seg in SEGS:
            val = live_ctr(genre_s, seg, ava_score=ava_final, is_movie=int(is_movie_s))
            if val is not None:
                ctrs[seg] = val
                api_ok    = True
            else:
                ctrs[seg] = fallback_ctr(genre_s, seg, ava_score=ava_final)

        # EC2 may be stopped; fallback to precomputed CSV is seamless — no UI warning needed

        st.pyplot(seg_bar(ctrs, highlight=seg_s), use_container_width=False)

        best_seg = max(ctrs, key=ctrs.get)
        delta_vs_avg = ava_final - genre_avg

        if delta_vs_avg > 0.5:
            st.success(
                f"**Above-average quality.** Best for **{SEGS[best_seg]}** "
                f"(CTR {ctrs[best_seg]:.3f}). "
                f"Score {ava_final} beats genre avg of {genre_avg:.2f}."
            )
        elif delta_vs_avg > -0.5:
            st.info(
                f"**Average quality.** Best for **{SEGS[best_seg]}**. "
                f"Score {ava_final} is on par with the genre average ({genre_avg:.2f})."
            )
        else:
            st.warning(
                f"**Below average.** Score {ava_final} is below the {genre_s} "
                f"genre average ({genre_avg:.2f}). "
                f"Consider higher contrast or more vibrant colors."
            )

    # ── Local Semantic Analysis card ────────────────────────────────────────
    if llm_ran and signals:
        st.markdown("---")
        st.markdown("#### Local Semantic Analysis")

        # Determine tag sentiments
        face_sent  = "positive" if signals.get("face_present") else "neutral"
        emo        = signals.get("emotion", "neutral")
        emo_sent   = ("positive" if emo in ("intense", "warm", "joyful", "mysterious")
                      else "neutral")
        txt        = signals.get("text_presence", "minimal")
        txt_sent   = "positive" if txt == "none" else "neutral" if txt == "minimal" else "negative"
        comp       = signals.get("composition", "moderate")
        comp_sent  = "positive" if comp == "strong" else "neutral" if comp == "moderate" else "negative"
        adj_sent   = "positive" if ava_adj > 0 else "negative" if ava_adj < 0 else "neutral"
        adj_label  = f"+{ava_adj}" if ava_adj > 0 else str(ava_adj)

        tags_html = " ".join([
            llm_tag("Face",        ("Yes" if signals.get("face_present") else "No") +
                                   (f' ({signals.get("face_count", "")})', "")[signals.get("face_count") == "none"],
                    face_sent),
            llm_tag("Emotion",     emo.capitalize(), emo_sent),
            llm_tag("Text",        txt.capitalize(), txt_sent),
            llm_tag("Composition", comp.capitalize(), comp_sent),
            llm_tag("Color",       signals.get("color_mood", "—").capitalize(), "neutral"),
            llm_tag("AVA Adjust",  adj_label, adj_sent),
        ])

        st.markdown(
            f'<div class="llm-card">'
            f'<div style="margin-bottom:12px">{tags_html}</div>'
            f'<p style="color:#cccccc;font-size:14px;margin:0;line-height:1.6">'
            f'{explanation}</p>'
            f'</div>',
            unsafe_allow_html=True,
        )

        # Score decomposition table
        with st.expander("How the AI adjustment was calculated"):
            rows_exp = []
            if signals.get("face_present"):
                rows_exp.append(("Face detected", "+0.40"))
            if signals.get("face_count") == "multiple":
                rows_exp.append(("Multiple faces", "+0.10"))
            emo_val = {
                "action_viewer": {"intense": 0.40, "mysterious": 0.25, "joyful": -0.10,
                                  "warm": -0.10, "neutral": 0.00},
                "drama_viewer":  {"warm": 0.30, "mysterious": 0.25, "intense": 0.15,
                                  "joyful": 0.10, "neutral": 0.00},
                "family_viewer": {"joyful": 0.40, "warm": 0.35, "neutral": 0.00,
                                  "mysterious": -0.15, "intense": -0.30},
            }.get(seg_s, {}).get(emo, 0.0)
            if emo_val != 0:
                rows_exp.append((f'Emotion "{emo}" for {SEGS[seg_s]}',
                                 f"{'+' if emo_val > 0 else ''}{emo_val:.2f}"))
            txt_v = {"none": 0.10, "minimal": 0.00, "heavy": -0.50}.get(txt, 0.0)
            if txt_v != 0:
                rows_exp.append((f'Text overlay "{txt}"',
                                 f"{'+' if txt_v > 0 else ''}{txt_v:.2f}"))
            comp_v = {"strong": 0.40, "moderate": 0.00, "weak": -0.40}.get(comp, 0.0)
            if comp_v != 0:
                rows_exp.append((f'Composition "{comp}"',
                                 f"{'+' if comp_v > 0 else ''}{comp_v:.2f}"))
            rows_exp.append(("Total adjustment (clamped ±1.5)", adj_label))

            tbl_df = pd.DataFrame(rows_exp, columns=["Signal", "AVA Δ"])
            st.dataframe(tbl_df, hide_index=True, use_container_width=True)

    elif LLM_OK and not llm_ran:
        if llm_error_msg:
            st.error(f"**Semantic analysis failed:** {llm_error_msg}")
        else:
            st.warning("Semantic analysis returned no result — check your API key or try again.")
    elif not LLM_OK:
        st.markdown(
            "<div style='background:#1e1e1e;border:1px solid #2a2a2a;border-radius:8px;"
            "padding:14px 18px;color:#555;font-size:13px'>"
            "Run <code>pip install nltk opencv-python-headless</code> and restart to enable local semantic analysis."
            "</div>",
            unsafe_allow_html=True,
        )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — TMDB IMAGE RANKER
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🎬  TMDB Image Ranker":
    st.title("TMDB Image Ranker")
    st.markdown(
        "Fetch all available posters for a Netflix title, score each one, "
        "and rank them by predicted CTR for your target viewer segment."
    )
    st.markdown("---")

    if not TMDB_KEY:
        st.warning(
            "**TMDB API key not set.**  \n"
            "```\nexport TMDB_API_KEY=your_key\nstreamlit run streamlit_app_final.py\n```  \n"
            "Get a free key at [themoviedb.org/settings/api](https://www.themoviedb.org/settings/api)"
        )
        st.stop()

    # ── Inputs ───────────────────────────────────────────────────────────────
    c1, c2 = st.columns([2, 1])
    with c1:
        title_r  = st.selectbox("Select a Netflix title", sorted(df_preds["title"].unique()))
    with c2:
        seg_r    = st.selectbox(
            "Rank for segment", list(SEGS.keys()),
            format_func=lambda s: SEGS[s], key="ir_seg",
        )

    genre_r    = (df_preds[df_preds["title"] == title_r]["genre_bucket"].iloc[0]
                  if title_r in df_preds["title"].values else "Dramas")
    is_movie_r = st.toggle("Movie (not TV show)", value=True, key="ir_movie")

    use_llm_r = LLM_OK and st.checkbox(
        "Enable local semantic analysis per poster",
        value=LLM_OK,
    )

    run_r = st.button("Fetch & Rank Posters", type="primary")

    if run_r:
        with st.spinner(f"Fetching posters for *{title_r}* from TMDB…"):
            posters = tmdb_posters(title_r)

        if not posters:
            st.error("No posters found on TMDB for this title. Try a different title.")
            st.stop()

        results = []
        prog    = st.progress(0, text="Scoring images…")

        for i, poster in enumerate(posters):
            scored, img_obj = score_from_url(poster["url"])
            if scored is None:
                prog.progress((i + 1) / len(posters))
                continue

            ava_s, breakdown = scored
            ava_final_r      = ava_s
            signals_r        = None
            explanation_r    = None

            # Local semantic analysis
            if use_llm_r and img_obj:
                prog.progress((i + 1) / len(posters),
                              text=f"Analyzing poster {i+1}/{len(posters)}…")
                img_bytes_r = img_to_bytes(img_obj)
                signals_r, explanation_r = analyze_thumbnail_llm(
                    img_bytes_r, genre_r, seg_r
                )
                if signals_r:
                    ava_adj_r    = compute_llm_adjustment(signals_r, seg_r)
                    ava_final_r  = round(min(9.0, max(2.0, ava_s + ava_adj_r)), 2)
            else:
                prog.progress((i + 1) / len(posters),
                              text=f"Scoring poster {i+1}/{len(posters)}…")

            ctr_r = live_ctr(genre_r, seg_r, ava_score=ava_final_r,
                             is_movie=int(is_movie_r))
            if ctr_r is None:
                ctr_r = fallback_ctr(genre_r, seg_r, ava_score=ava_final_r)

            results.append({
                "url":         poster["url"],
                "img":         img_obj,
                "ava_raw":     ava_s,
                "ava_final":   ava_final_r,
                "breakdown":   breakdown,
                "ctr":         ctr_r,
                "signals":     signals_r,
                "explanation": explanation_r,
            })

        prog.empty()

        if not results:
            st.error("Could not score any posters.")
            st.stop()

        results.sort(key=lambda x: x["ctr"], reverse=True)
        winner = results[0]

        st.markdown(f"#### Results — ranked for **{SEGS[seg_r]}**")
        st.markdown("---")

        # Winner callout
        score_source = "Locally adjusted" if use_llm_r else "PIL"
        st.success(
            f"**Best thumbnail:** aesthetic score {winner['ava_final']:.2f}/10 "
            f"({score_source}), predicted CTR **{winner['ctr']:.4f}** "
            f"for {SEGS[seg_r]}."
        )

        # LLM explanation for winner
        if winner.get("explanation"):
            st.markdown(
                f'<div class="llm-card">'
                f'<span style="color:#a0a0ff;font-size:11px;font-weight:700;'
                f'letter-spacing:.5px">Semantic Analysis · WINNER</span>'
                f'<p style="color:#cccccc;font-size:14px;margin:8px 0 0;line-height:1.6">'
                f'{winner["explanation"]}</p>'
                f'</div>',
                unsafe_allow_html=True,
            )

        st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

        # Poster grid
        n_cols  = min(len(results), 3)
        columns = st.columns(n_cols)

        for i, res in enumerate(results):
            with columns[i % n_cols]:
                border = "2px solid #E50914" if i == 0 else "1px solid #2a2a2a"
                rank   = "🥇 Best" if i == 0 else f"#{i + 1}"
                st.markdown(
                    f"<div style='border:{border};border-radius:8px;overflow:hidden;"
                    f"margin-bottom:8px'>",
                    unsafe_allow_html=True,
                )
                st.image(res["img"], use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)
                st.markdown(f"**{rank}**")
                st.metric("Predicted CTR",   f"{res['ctr']:.4f}")
                st.metric("Aesthetic Score", f"{res['ava_final']:.2f}/9")

                # Compact semantic tags
                if res.get("signals"):
                    sig = res["signals"]
                    emo = sig.get("emotion", "—")
                    cmp = sig.get("composition", "—")
                    txt = sig.get("text_presence", "—")
                    st.markdown(
                        f'<div style="margin-top:4px">'
                        f'<span class="llm-tag">{emo}</span>'
                        f'<span class="llm-tag">{cmp} comp</span>'
                        f'<span class="llm-tag">text: {txt}</span>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

                st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

        # Full comparison table
        st.markdown("---")
        st.markdown("#### Full Comparison")
        tbl_rows = []
        for i, r in enumerate(results):
            row = {
                "Rank":            i + 1,
                "PIL Score":       r["ava_raw"],
                "Final Score":     r["ava_final"],
                "Brightness":      r["breakdown"]["Brightness"],
                "Contrast":        r["breakdown"]["Contrast"],
                "Colorfulness":    r["breakdown"]["Colorfulness"],
                "Sharpness":       r["breakdown"]["Sharpness"],
                f"CTR ({SEGS[seg_r]})": r["ctr"],
            }
            if use_llm_r and r.get("signals"):
                row["Emotion"]     = r["signals"].get("emotion", "—")
                row["Composition"] = r["signals"].get("composition", "—")
                row["Text"]        = r["signals"].get("text_presence", "—")
            tbl_rows.append(row)

        st.dataframe(pd.DataFrame(tbl_rows).set_index("Rank"), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — A/B SIMULATOR
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🧪  A/B Simulator":
    st.title("A/B Simulator")
    st.markdown(
        "Compare two thumbnail strategies for the same title and viewer segment. "
        "See which wins and how many additional clicks per month that translates to."
    )
    st.markdown("---")

    title_ab = st.selectbox("Select a title", sorted(df_preds["title"].unique()))
    rows_ab  = df_preds[df_preds["title"] == title_ab].copy()

    c1, c2, c3 = st.columns(3)
    with c1:
        seg_a = st.selectbox(
            "Thumbnail A — target audience",
            list(SEGS.keys()), format_func=lambda s: SEGS[s], key="ab_sa",
        )
    with c2:
        seg_b = st.selectbox(
            "Thumbnail B — target audience",
            list(SEGS.keys()), index=1, format_func=lambda s: SEGS[s], key="ab_sb",
        )
    with c3:
        imp = st.slider(
            "Daily impressions", 100_000, 5_000_000, 500_000, step=100_000,
            format="%d",
        )

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    if st.button("Run Simulation", type="primary"):
        def get_ctr_ab(seg):
            m = rows_ab.loc[rows_ab["user_segment"] == seg, "predicted_ctr"]
            return float(m.values[0]) if not m.empty else 0.30

        ctr_a, ctr_b = get_ctr_ab(seg_a), get_ctr_ab(seg_b)
        clicks_a     = int(ctr_a * imp)
        clicks_b     = int(ctr_b * imp)
        winner       = "A" if ctr_a >= ctr_b else "B"
        win_ctr      = max(ctr_a, ctr_b)
        lose_ctr     = min(ctr_a, ctr_b)
        extra_day    = abs(clicks_a - clicks_b)
        extra_month  = extra_day * 30
        lift_pct     = abs(ctr_a - ctr_b) / lose_ctr * 100 if lose_ctr > 0 else 0

        st.markdown("---")
        m1, m2, m3 = st.columns(3)
        m1.metric(f"Thumbnail A  ({SEGS[seg_a]})", f"{ctr_a:.4f}",
                  f"{clicks_a:,} clicks/day")
        m2.metric(f"Thumbnail B  ({SEGS[seg_b]})", f"{ctr_b:.4f}",
                  f"{clicks_b:,} clicks/day")
        m3.metric("Winner", f"Thumbnail {winner}", f"+{lift_pct:.1f}% CTR lift")

        # Comparison bar chart
        fig, ax = plt.subplots(figsize=(6, 2.0))
        fig.patch.set_facecolor("#141414")
        ax.set_facecolor("#141414")
        ylabels = [f"B  ({SEGS[seg_b]})", f"A  ({SEGS[seg_a]})"]
        yvals   = [ctr_b, ctr_a]
        ycolors = [
            COLORS[seg_b] if winner == "B" else "#333",
            COLORS[seg_a] if winner == "A" else "#333",
        ]
        bars = ax.barh(ylabels, yvals, color=ycolors, height=0.4)
        ax.set_xlim(0, max(yvals) * 1.3)
        ax.set_xlabel("Predicted CTR", color="#555", fontsize=8)
        ax.tick_params(colors="#888", labelsize=9)
        for spine in ax.spines.values():
            spine.set_visible(False)
        for bar, val in zip(bars, yvals):
            ax.text(val + 0.003, bar.get_y() + bar.get_height() / 2,
                    f"{val:.4f}", va="center", color="white", fontsize=9, fontweight="bold")
        plt.tight_layout(pad=0.5)
        st.pyplot(fig, use_container_width=False)

        st.success(
            f"**Thumbnail {winner} wins.** CTR {win_ctr:.4f} vs {lose_ctr:.4f} — "
            f"that's **{extra_day:,} additional clicks/day** "
            f"and **~{extra_month:,} additional clicks/month** "
            f"at {imp:,} daily impressions."
        )

        # Monthly projection chart
        days    = list(range(1, 31))
        delta_a = [ctr_a * imp * d for d in days]
        delta_b = [ctr_b * imp * d for d in days]

        fig2, ax2 = plt.subplots(figsize=(8, 3))
        fig2.patch.set_facecolor("#141414")
        ax2.set_facecolor("#141414")
        ax2.plot(days, [v / 1e6 for v in delta_a], color=COLORS[seg_a],
                 linewidth=2, label=f"A ({SEGS[seg_a]})")
        ax2.plot(days, [v / 1e6 for v in delta_b], color=COLORS[seg_b],
                 linewidth=2, label=f"B ({SEGS[seg_b]})", linestyle="--")
        ax2.fill_between(
            days,
            [v / 1e6 for v in delta_a],
            [v / 1e6 for v in delta_b],
            alpha=0.12,
            color="#ffffff",
        )
        ax2.set_xlabel("Days", color="#555", fontsize=9)
        ax2.set_ylabel("Cumulative Clicks (M)", color="#555", fontsize=9)
        ax2.tick_params(colors="#666", labelsize=9)
        ax2.legend(facecolor="#1e1e1e", labelcolor="white", fontsize=9, framealpha=0.8)
        ax2.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, _: f"{x:.1f}M")
        )
        for spine in ax2.spines.values():
            spine.set_visible(False)
        plt.tight_layout(pad=0.5)

        st.markdown("#### Cumulative click projection — 30 days")
        st.pyplot(fig2, use_container_width=True)
