import os
import ast
import re
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from dejacode import Searcher, recommend_next, attach_solution_stats, get_code_snippet, DIFF_ORDER

# ----------------- PAGE CONFIG -----------------
st.set_page_config(
    page_title="Deja Code | Problem Search & Recommendations",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------- STYLING SYSTEM -----------------
st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap" rel="stylesheet">

<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');

    /* =========================================================================
       TYPOGRAPHY CONFIGURATION (3-Role Font System)
       To switch font pairing easily, update the CSS variables in this :root block:
       
       Active Setup:
       - Heading: Space Grotesk
       - Body / UI: Inter
       - Monospace: JetBrains Mono
       
       Alternative Pairings (switchable in one place):
       - Option 1 (Friendly SaaS):   'Plus Jakarta Sans' / 'DM Sans' / 'Fira Code'
       - Option 2 (Futuristic):      'Sora' / 'Manrope' / 'IBM Plex Mono'
       - Option 3 (Soft / Beginner): 'Outfit' / 'Nunito Sans' / 'Source Code Pro'
       ========================================================================= */
    :root {
        --font-heading: 'Space Grotesk', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        --font-body: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        --font-mono: 'JetBrains Mono', ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace;
    }

    /* === 1. GLOBAL LAYOUT & NEUTRAL CANVAS === */
    html, body, [data-testid="stAppViewContainer"] {
        background-color: #F5F6FA !important;
        color: #1E293B !important;
        font-family: var(--font-body) !important;
        -webkit-font-smoothing: antialiased;
        -moz-osx-font-smoothing: grayscale;
    }
    
    /* Cap max content width and trim excessive top margin */
    .block-container {
        max-width: 1200px !important;
        padding-top: 1.6rem !important;
        padding-bottom: 2.5rem !important;
        padding-left: 2rem !important;
        padding-right: 2rem !important;
    }

    /* === 2. SIDEBAR DARK GRADIENT & ACCENTS === */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0F172A 0%, #1E1B4B 100%) !important;
        border-right: 1px solid rgba(255, 255, 255, 0.08) !important;
    }
    section[data-testid="stSidebar"] > div {
        background: transparent !important;
        padding-top: 1.8rem !important;
    }
    section[data-testid="stSidebar"] hr {
        border-color: rgba(255, 255, 255, 0.12) !important;
        margin: 1.2rem 0 !important;
    }
    
    /* Sidebar Translucent Metric Panels with Teal Left Border */
    section[data-testid="stSidebar"] div[data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.06) !important;
        border: 1px solid rgba(255, 255, 255, 0.12) !important;
        border-left: 4px solid #14B8A6 !important;
        border-radius: 10px !important;
        padding: 10px 14px !important;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.15) !important;
        margin-bottom: 10px !important;
    }
    section[data-testid="stSidebar"] div[data-testid="stMetricLabel"] * {
        color: #94A3B8 !important;
        font-family: var(--font-body) !important;
        font-size: 0.72rem !important;
        font-weight: 600 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.06em !important;
    }
    section[data-testid="stSidebar"] div[data-testid="stMetricValue"] * {
        color: #F8FAFC !important;
        font-family: var(--font-heading) !important;
        font-size: 1.45rem !important;
        font-weight: 700 !important;
        letter-spacing: -0.02em !important;
        font-variant-numeric: tabular-nums !important;
    }

    /* === 3. TYPOGRAPHY & TYPE SCALE (SCOPED TO MAIN AREA) === */
    /* Headings: Space Grotesk, bold with tight letter-spacing */
    .main h1, .main h2, .main h3, .main h4, .main h5, .main h6,
    .main .stMarkdown h1, .main .stMarkdown h2, .main .stMarkdown h3, .main .stMarkdown h4,
    [data-testid="stMain"] h1, [data-testid="stMain"] h2, [data-testid="stMain"] h3,
    [data-testid="stAppViewContainer"] > section:not([data-testid="stSidebar"]) h1,
    [data-testid="stAppViewContainer"] > section:not([data-testid="stSidebar"]) h2 {
        color: #0F172A !important;
        font-family: var(--font-heading) !important;
        font-weight: 700 !important;
        letter-spacing: -0.02em !important;
    }
    
    .page-title {
        font-family: var(--font-heading) !important;
        font-size: 1.85rem !important;
        font-weight: 800 !important;
        letter-spacing: -0.025em !important;
        line-height: 1.2 !important;
        color: #0F172A !important;
        margin: 0 !important;
    }
    .page-subtitle {
        font-family: var(--font-body) !important;
        font-size: 0.93rem !important;
        color: #64748B !important;
        margin-top: 4px !important;
        margin-bottom: 0 !important;
        line-height: 1.55 !important;
    }
    .section-heading {
        font-family: var(--font-heading) !important;
        font-size: 1.12rem !important;
        font-weight: 700 !important;
        color: #0F172A !important;
        letter-spacing: -0.015em !important;
        line-height: 1.35 !important;
    }

    /* Body Text: Inter, comfortable line-height 1.55 */
    .main p, .main span, .main label,
    .main .stMarkdown p,
    [data-testid="stMain"] p, [data-testid="stMain"] span,
    [data-testid="stAppViewContainer"] > section:not([data-testid="stSidebar"]) p,
    [data-testid="stWidgetLabel"] label,
    [data-testid="stWidgetLabel"] p {
        color: #334155 !important;
        font-family: var(--font-body) !important;
        line-height: 1.55 !important;
    }
    .main .stCaption, [data-testid="stMain"] .stCaption {
        color: #64748B !important;
        font-family: var(--font-body) !important;
        font-size: 0.85rem !important;
        line-height: 1.5 !important;
    }

    /* === SIDEBAR TYPOGRAPHY SPECIFIC OVERRIDES === */
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] h4,
    section[data-testid="stSidebar"] .stMarkdown h1,
    section[data-testid="stSidebar"] .stMarkdown h2,
    section[data-testid="stSidebar"] .stMarkdown h3,
    section[data-testid="stSidebar"] .stMarkdown h4,
    section[data-testid="stSidebar"] .sidebar-title {
        color: #FFFFFF !important;
        font-family: var(--font-heading) !important;
        letter-spacing: -0.025em !important;
    }
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] .stMarkdown p,
    section[data-testid="stSidebar"] .sidebar-subtitle {
        color: #CBD5E1 !important;
        font-family: var(--font-body) !important;
        line-height: 1.5 !important;
    }
    section[data-testid="stSidebar"] .sidebar-footer {
        color: #94A3B8 !important;
        font-family: var(--font-body) !important;
        font-size: 0.74rem !important;
        line-height: 1.5 !important;
    }
    section[data-testid="stSidebar"] .sidebar-footer a {
        color: #818CF8 !important;
        text-decoration: none !important;
        font-weight: 600 !important;
        font-family: var(--font-body) !important;
    }
    section[data-testid="stSidebar"] .stCaption {
        color: #94A3B8 !important;
        font-family: var(--font-body) !important;
    }

    /* === 4. PILL TABS NAVIGATION === */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px !important;
        background-color: transparent !important;
        padding: 4px 0 16px 0 !important;
        border-bottom: none !important;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #FFFFFF !important;
        border: 1px solid #E2E8F0 !important;
        border-radius: 22px !important;
        padding: 8px 22px !important;
        font-family: var(--font-body) !important;
        font-weight: 500 !important;
        font-size: 0.90rem !important;
        color: #64748B !important;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04) !important;
        transition: all 0.15s ease-in-out !important;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: #1E293B !important;
        border-color: #CBD5E1 !important;
    }
    .stTabs [aria-selected="true"] {
        background-color: #6366F1 !important;
        color: #FFFFFF !important;
        font-family: var(--font-body) !important;
        font-weight: 600 !important;
        border-color: #6366F1 !important;
        box-shadow: 0 2px 8px rgba(99, 102, 241, 0.3) !important;
    }
    /* Suppress default underline bar */
    .stTabs [data-baseweb="tab-highlight"],
    .stTabs [data-baseweb="tab-border"] {
        display: none !important;
    }

    /* === 5. WHITE CARDS, PANELS & CONTAINERS === */
    .white-card {
        background: #FFFFFF;
        border: 1px solid #E3E7F0;
        border-radius: 12px;
        padding: 16px 20px;
        margin-bottom: 12px;
        box-shadow: 0 1px 4px rgba(0, 0, 0, 0.04);
        transition: border-color 0.15s ease, box-shadow 0.15s ease;
    }
    .white-card:hover {
        border-color: #CBD5E1;
        box-shadow: 0 3px 8px rgba(0, 0, 0, 0.06);
    }
    
    .control-card {
        background: #FFFFFF;
        border: 1px solid #E3E7F0;
        border-radius: 12px;
        padding: 18px 20px;
        box-shadow: 0 1px 4px rgba(0, 0, 0, 0.03);
    }
    .control-card-header {
        font-family: var(--font-body) !important;
        font-size: 0.75rem !important;
        font-weight: 700 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.08em !important;
        color: #475569 !important;
        margin-bottom: 12px;
        padding-bottom: 8px;
        border-bottom: 1px solid #F1F5F9;
    }

    /* Problem titles in result cards: Semi-bold Space Grotesk */
    .problem-title {
        font-family: var(--font-heading) !important;
        font-size: 1.02rem !important;
        font-weight: 600 !important;
        letter-spacing: -0.01em !important;
        line-height: 1.35 !important;
    }
    .problem-code {
        font-family: var(--font-mono) !important;
        font-size: 0.80rem !important;
        font-weight: 500 !important;
        color: #64748B !important;
        letter-spacing: -0.01em !important;
    }

    /* === 6. INPUTS & FORM CONTROLS === */
    .stTextInput input,
    .stTextArea textarea {
        background-color: #FFFFFF !important;
        border: 1px solid #E2E8F0 !important;
        border-radius: 10px !important;
        color: #0F172A !important;
        font-family: var(--font-body) !important;
        font-size: 0.92rem !important;
        line-height: 1.5 !important;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.03) !important;
        padding: 10px 14px !important;
        transition: border-color 0.15s ease, box-shadow 0.15s ease !important;
    }
    .stTextInput input:focus,
    .stTextArea textarea:focus {
        border-color: #6366F1 !important;
        box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.18) !important;
        outline: none !important;
    }
    div[data-baseweb="select"] > div {
        background-color: #FFFFFF !important;
        border: 1px solid #E2E8F0 !important;
        border-radius: 10px !important;
        font-family: var(--font-body) !important;
    }
    div[data-baseweb="select"] > div:focus-within {
        border-color: #6366F1 !important;
        box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.18) !important;
    }
    
    /* Primary Action Buttons: Body font, medium weight 500 */
    button, button[kind="primary"], .stButton > button {
        font-family: var(--font-body) !important;
        font-weight: 500 !important;
        font-size: 0.92rem !important;
        letter-spacing: 0.01em !important;
    }
    button[kind="primary"] {
        background-color: #6366F1 !important;
        color: #FFFFFF !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 10px 20px !important;
        box-shadow: 0 2px 6px rgba(99, 102, 241, 0.25) !important;
        transition: background-color 0.15s ease !important;
    }
    button[kind="primary"]:hover {
        background-color: #4F46E5 !important;
    }

    /* Slider Font Harmonization (Body font with Tabular Numerals) */
    div[data-testid="stSlider"] * {
        font-family: var(--font-body) !important;
        font-variant-numeric: tabular-nums !important;
    }
    div[data-testid="stSlider"] div[data-testid="stTickBarMin"],
    div[data-testid="stSlider"] div[data-testid="stTickBarMax"] {
        font-family: var(--font-body) !important;
        font-size: 0.78rem !important;
        color: #64748B !important;
        font-variant-numeric: tabular-nums !important;
    }

    /* === 7. MULTISELECT CHIPS (NO TRUNCATION) === */
    div[data-baseweb="tag"] {
        background-color: #EEF2FF !important;
        border: 1px solid #C7D2FE !important;
        border-radius: 16px !important;
        max-width: 100% !important;
        height: auto !important;
        padding: 4px 10px !important;
        margin: 3px !important;
    }
    div[data-baseweb="tag"] span {
        color: #4338CA !important;
        font-family: var(--font-body) !important;
        font-size: 0.82rem !important;
        font-weight: 500 !important;
        white-space: normal !important;
        text-overflow: unset !important;
        overflow: visible !important;
    }
    div[data-baseweb="tag"] svg {
        fill: #6366F1 !important;
    }

    /* === 8. DIFFICULTY BADGES & SCORE PILLS === */
    /* Difficulty Badges: Body font, uppercase, small, slight letter-spacing */
    .badge-pill {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 14px;
        font-family: var(--font-body) !important;
        font-size: 0.72rem !important;
        font-weight: 600 !important;
        letter-spacing: 0.05em !important;
        text-transform: uppercase !important;
    }
    .badge-beginner  { background: #DCFCE7; color: #15803D; border: 1px solid #BBF7D0; }
    .badge-easy      { background: #E0F2FE; color: #0369A1; border: 1px solid #BAE6FD; }
    .badge-medium    { background: #FEF3C7; color: #B45309; border: 1px solid #FDE68A; }
    .badge-hard      { background: #FEE2E2; color: #B91C1C; border: 1px solid #FECACA; }
    .badge-challenge { background: #F3E8FF; color: #7E22CE; border: 1px solid #E9D5FF; }

    /* Score Pills: Mono font, tabular numbers */
    .score-pill {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 14px;
        font-family: var(--font-mono) !important;
        font-size: 0.74rem !important;
        font-weight: 500 !important;
        letter-spacing: -0.01em !important;
        font-variant-numeric: tabular-nums !important;
        background: #EEF2FF;
        color: #4338CA;
        border: 1px solid #C7D2FE;
    }

    /* === 9. MODE COMPARISON METRIC CARDS === */
    .comp-metric-card {
        background: #FFFFFF;
        border-radius: 12px;
        padding: 14px 18px;
        border: 1px solid #E3E7F0;
        box-shadow: 0 1px 4px rgba(0, 0, 0, 0.04);
    }
    .comp-metric-1 { border-top: 3px solid #6366F1; } /* Indigo */
    .comp-metric-2 { border-top: 3px solid #14B8A6; } /* Teal */
    .comp-metric-3 { border-top: 3px solid #F59E0B; } /* Amber */
    .comp-metric-label {
        color: #64748B;
        font-family: var(--font-body) !important;
        font-size: 0.74rem !important;
        font-weight: 600 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.06em !important;
    }
    .comp-metric-val {
        color: #0F172A;
        font-family: var(--font-heading) !important;
        font-size: 1.7rem !important;
        font-weight: 700 !important;
        margin-top: 2px;
        letter-spacing: -0.02em !important;
        font-variant-numeric: tabular-nums !important;
    }

    /* Side-by-Side Result Card Accents */
    .mode-card {
        background: #FFFFFF;
        border: 1px solid #E3E7F0;
        border-radius: 10px;
        padding: 12px 16px;
        margin-bottom: 10px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.03);
    }
    .mode-card-kw   { border-left: 4px solid #94A3B8; }
    .mode-card-sem  { border-left: 4px solid #14B8A6; }
    .mode-card-both { border-left: 4px solid #6366F1; }

    /* === 10. PROBLEM PREVIEW SECTION LABELS, CODE BLOCKS & EXPANDERS === */
    div[data-testid="stExpander"] {
        background: #FFFFFF !important;
        border-radius: 10px !important;
        border: 1px solid #E3E7F0 !important;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.03) !important;
        margin-top: 8px !important;
    }
    div[data-testid="stExpander"] summary {
        padding: 8px 14px !important;
    }
    div[data-testid="stExpander"] summary * {
        color: #475569 !important;
        font-family: var(--font-body) !important;
        font-size: 0.85rem !important;
        font-weight: 600 !important;
    }
    
    /* Small uppercase labels for preview */
    .preview-label {
        font-family: var(--font-body) !important;
        font-size: 0.72rem !important;
        font-weight: 700 !important;
        letter-spacing: 0.08em !important;
        text-transform: uppercase !important;
        color: #64748B !important;
        margin-top: 14px;
        margin-bottom: 4px;
        padding-bottom: 2px;
        border-bottom: 1px solid #F1F5F9;
    }

    /* Code Blocks & Monospace Snippets: JetBrains Mono */
    code, pre, pre code,
    div[data-testid="stCodeBlock"] *,
    div[data-testid="stCodeBlock"] pre,
    div[data-testid="stCodeBlock"] code,
    .stMarkdown pre,
    .stMarkdown code,
    .stCode * {
        font-family: var(--font-mono) !important;
        font-size: 0.84rem !important;
        line-height: 1.5 !important;
    }
</style>
""", unsafe_allow_html=True)

# ----------------- DATA LOADER -----------------
@st.cache_resource(show_spinner="Loading problem index and models...")
def load_app_data():
    data_path = "out/problems_clustered.csv"
    emb_path = "out/emb.npy"
    coords_path = "out/coords_2d.npy"
    if not (os.path.exists(data_path) and os.path.exists(emb_path)):
        return None, None, None, None, {}
    
    df = pd.read_csv(data_path)
    def safe_parse(v):
        if isinstance(v, list): return v
        try:
            val = ast.literal_eval(str(v))
            return val if isinstance(val, list) else []
        except Exception:
            return []
    df["tags"] = df["tags"].map(safe_parse)
    df["doc"] = df["title"].astype(str) + ". " + df["text"].astype(str)
    
    df = attach_solution_stats(df, "data/solutions.csv")
    emb = np.load(emb_path)
    coords = np.load(coords_path) if os.path.exists(coords_path) else None
    searcher = Searcher(df, emb)

    raw_map = {}
    if os.path.exists("data/questions.csv"):
        try:
            raw_q = pd.read_csv("data/questions.csv", usecols=["QCode", "statement"])
            raw_map = dict(zip(raw_q["QCode"].astype(str), raw_q["statement"].astype(str)))
        except Exception:
            pass

    return df, emb, coords, searcher, raw_map

df, emb, coords, searcher, raw_map = load_app_data()

# ----------------- LATEX & STATEMENT FORMATTER -----------------
def render_problem_latex(raw_text):
    """Format problem statement with KaTeX LaTeX math, small uppercase labels, and clean code blocks."""
    if not raw_text or pd.isna(raw_text):
        return "No statement text available."
    t = str(raw_text)

    # 1. Strip web boilerplate & scraped artifacts
    t = re.sub(r'<[^>]+>', ' ', t)
    t = re.sub(r'(?i)all submissions for this problem are available\.?', '', t)
    t = re.sub(r'(?i)read problems? statements? in [^.]*?(?:as well|here)\s*\.?|read problems? statements? in [^.]*\.', '', t)
    t = re.sub(r'\$\([^\)]*\)\.html\([^\)]*\);?', '', t)
    t = re.sub(r'\$\.ajax\(\{.*?\}\);?', '', t, flags=re.DOTALL)
    t = re.sub(r'Here is where you can show off your computer programming skills.*', '', t, flags=re.DOTALL)

    # 2. Format Sample Input / Output code blocks
    t = re.sub(r'(?i)\b(?:sample\s*input|example\s*input)\s*:?\s*\n+((?:[ \t]*\S+[^\n]*\n*)+?)(?=\n*(?:sample\s*output|output\s*:|explanation|\Z))', 
               lambda m: '\n<div class="preview-label">SAMPLE INPUT</div>\n\n```\n' + m.group(1).strip() + '\n```\n', t)
    t = re.sub(r'(?i)\b(?:sample\s*output|example\s*output)\s*:?\s*\n+((?:[ \t]*\S+[^\n]*\n*)+?)(?=\n*(?:explanation|constraints|\Z))', 
               lambda m: '\n<div class="preview-label">SAMPLE OUTPUT</div>\n\n```\n' + m.group(1).strip() + '\n```\n', t)

    # 3. Small uppercase section labels
    t = re.sub(r'(?m)^\s*(?:Input\s*format|Input)\s*:?\s*$', '\n<div class="preview-label">INPUT FORMAT</div>\n', t, flags=re.I)
    t = re.sub(r'(?m)^\s*(?:Output\s*format|Output)\s*:?\s*$', '\n<div class="preview-label">OUTPUT FORMAT</div>\n', t, flags=re.I)
    t = re.sub(r'(?m)^\s*(?:Constraints|Test\s*data)\s*:?\s*$', '\n<div class="preview-label">CONSTRAINTS</div>\n', t, flags=re.I)
    t = re.sub(r'(?m)^\s*(?:Explanation)\s*:?\s*$', '\n<div class="preview-label">EXPLANATION</div>\n', t, flags=re.I)

    # 4. Convert math inequalities and notation to KaTeX LaTeX
    def double_ineq(m):
        lo, v, hi = m.group(1), m.group(2).strip(), m.group(3)
        v = re.sub(r'\b([A-Za-z])(\d)\b', r'\1_\2', v)
        return f"${lo} \\le {v} \\le {hi}$"
    t = re.sub(r'(\d+)\s*(?:≤|<=)\s*([A-Za-z0-9_, ]+?)\s*(?:≤|<=)\s*(\d+(?:\^\{\d+\}|\^\d+)?)', double_ineq, t)

    def single_le(m):
        v, val = m.group(1).strip(), m.group(2)
        v = re.sub(r'\b([A-Za-z])(\d)\b', r'\1_\2', v)
        return f"${v} \\le {val}$"
    t = re.sub(r'([A-Za-z][A-Za-z0-9_]*)\s*(?:≤|<=)\s*(\d+(?:\^\{\d+\}|\^\d+)?)', single_le, t)

    def single_ge(m):
        v, val = m.group(1).strip(), m.group(2)
        v = re.sub(r'\b([A-Za-z])(\d)\b', r'\1_\2', v)
        return f"${v} \\ge {val}$"
    t = re.sub(r'([A-Za-z][A-Za-z0-9_]*)\s*(?:≥|>=)\s*(\d+(?:\^\{\d+\}|\^\d+)?)', single_ge, t)

    t = re.sub(r'(?<![A-Za-z0-9_\$\\])\b([NnMmA-Z])(\d)\b(?![A-Za-z0-9_\$])', r'$\1_\2$', t)
    t = re.sub(r'(?<!\$)\b10\^(\d+)\b(?!\$)', r'$10^{\1}$', t)
    t = re.sub(r'\bO\(([NnMKk][\w\s\*\+\-\^\(\)\/]*)\)', r'$\\mathcal{O}(\1)$', t)

    t = re.sub(r'\n{3,}', '\n\n', t)
    return t.strip()

# ----------------- SIDEBAR -----------------
with st.sidebar:
    st.markdown("""
    <div style="margin-bottom: 1.2rem;">
        <h2 class="sidebar-title" style="color: #FFFFFF !important; margin: 0; font-size: 1.6rem; font-weight: 800; letter-spacing: -0.02em;">Deja Code</h2>
        <p class="sidebar-subtitle" style="color: #CBD5E1 !important; margin-top: 4px; font-size: 0.82rem; font-weight: 500;">Competitive Programming Search & Recs</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<div style="height: 1px; background: rgba(255,255,255,0.12); margin-bottom: 1.2rem;"></div>', unsafe_allow_html=True)
    
    if df is not None:
        st.metric("Problems Indexed", f"{len(df):,}")
        clusters_count = df["cluster"].nunique() if "cluster" in df.columns else 0
        st.metric("Concept Clusters", f"{clusters_count}")
        st.metric("Total Submissions Indexed", f"{df['total_submissions'].sum():,}")
    
    st.markdown("""
    <div class="sidebar-footer" style="margin-top: 2.2rem; padding-top: 1.2rem; border-top: 1px solid rgba(255,255,255,0.1); font-size: 0.74rem; color: #94A3B8 !important; line-height: 1.5;">
        Engineered with Sentence-Transformers, scikit-learn, UMAP, and Streamlit.<br>
        <a href="https://github.com/visshalb2007/dejacode" target="_blank" style="color: #818CF8 !important; text-decoration: none; font-weight: 600;">GitHub Repository</a>
    </div>
    """, unsafe_allow_html=True)

# ----------------- MAIN HEADER (NON-REDUNDANT) -----------------
st.markdown("""
<div style="margin-bottom: 1.4rem;">
    <h1 class="page-title">Problem Search & Recommendations</h1>
    <p class="page-subtitle">Algorithmic similarity discovery, topic clustering, and adaptive difficulty progression for CodeChef</p>
</div>
""", unsafe_allow_html=True)

if searcher is None:
    st.error("Index artifacts not found in out/. Please run python dejacode.py --data data/questions.csv --k 12 first.")
    st.stop()

# ----------------- TABS -----------------
tabs = st.tabs([
    "Search & Discovery",
    "Mode Comparison",
    "Cluster Manifold",
    "Recommendations"
])

# ----------------- TAB 1: Search & Discovery -----------------
with tabs[0]:
    sample_queries = [
        "-- Select an example query or type your own below --",
        "Shortest path in a weighted directed acyclic graph with obstacles",
        "Given an array find maximum subarray sum with at most k elements",
        "Number of ways to tile a grid using dominoes and trominoes dynamic programming",
        "Find the longest common subsequence of two strings",
        "Count inversions in an array using merge sort or Fenwick tree",
        "Nim game with piles of stones game theory optimal strategy"
    ]
    
    col_left, col_right = st.columns([2, 1])
    
    with col_left:
        selected_sample = st.selectbox("Preset Example Queries", sample_queries)
        default_q = "" if selected_sample.startswith("--") else selected_sample
        q = st.text_area(
            "Problem description, story, or algorithm keywords:",
            value=default_q,
            height=140,
            placeholder="Paste a problem statement or describe the underlying algorithmic task..."
        )
        search_clicked = st.button("Search Problems", type="primary", use_container_width=True)

    with col_right:
        st.markdown('<div class="control-card">', unsafe_allow_html=True)
        st.markdown('<div class="control-card-header">Search Controls</div>', unsafe_allow_html=True)
        mode = st.radio("Search Mode", ["hybrid", "semantic", "keyword"], index=0)
        alpha = 0.65
        if mode == "hybrid":
            alpha = st.slider("Dense / Sparse Weight (alpha)", 0.0, 1.0, 0.65, 0.05, help="1.0 = Pure Semantic Dense, 0.0 = Pure Sparse Keyword")
        k_results = st.slider("Max Results", 3, 20, 5)
        diff_filter = st.selectbox("Difficulty Filter", ["All"] + DIFF_ORDER)
        st.markdown('</div>', unsafe_allow_html=True)

    if q.strip():
        scores = searcher.scores(q, mode=mode, alpha=alpha)
        ranked_idx = np.argsort(-scores)
        
        results_df = df.iloc[ranked_idx].copy()
        results_df["score"] = scores[ranked_idx]
        
        if diff_filter != "All":
            results_df = results_df[results_df["difficulty"].astype(str).str.lower().str.strip() == diff_filter.lower()]
        
        top_k = results_df.head(k_results)
        
        st.markdown(f'<div class="section-heading">Top {len(top_k)} Matching Problems ({mode} mode)</div>', unsafe_allow_html=True)
        
        for _, row in top_k.iterrows():
            diff = str(row.get("difficulty", "medium")).lower()
            badge_class = f"badge-{diff}" if diff in DIFF_ORDER else "badge-medium"
            codechef_link = f"https://www.codechef.com{row['link']}" if str(row.get("link", "")).startswith("/") else f"https://www.codechef.com/problems/{row.get('qcode', '')}"
            tags_str = ", ".join(row["tags"]) if isinstance(row["tags"], list) and row["tags"] else "None"
            
            with st.container():
                st.markdown(f"""
                <div class="white-card">
                    <div style="display: flex; justify-content: space-between; align-items: flex-start;">
                        <div>
                            <h4 style="margin: 0;">
                                <a href="{codechef_link}" target="_blank" class="problem-title" style="text-decoration: none; color: #4338CA;">{row.get('title', 'Unknown')}</a>
                                <span class="problem-code">({row.get('qcode', 'N/A')})</span>
                            </h4>
                        </div>
                        <div style="display: flex; align-items: center; gap: 8px;">
                            <span class="badge-pill {badge_class}">{diff.upper()}</span>
                            <span class="score-pill">Score: {row['score']:.3f}</span>
                        </div>
                    </div>
                    <div style="margin-top: 8px; font-size: 0.84rem; color: #64748B;">
                        <strong>Tags:</strong> {tags_str} &nbsp;|&nbsp; 
                        <strong>Cluster:</strong> {row.get('cluster_label', 'N/A')}
                    </div>
                </div>
                """, unsafe_allow_html=True)
                with st.expander("Preview Problem Statement"):
                    stmt_raw = raw_map.get(str(row.get("qcode", ""))) or row.get("text", "No statement text available.")
                    st.markdown(render_problem_latex(stmt_raw), unsafe_allow_html=True)

# ----------------- TAB 2: Mode Comparison -----------------
with tabs[1]:
    st.caption("Directly contrast Sparse Keyword (TF-IDF) against Dense Semantic (all-MiniLM-L6-v2) on the same input.")

    comp_query = st.text_input(
        "Enter a test query for comparison:",
        value="Chef wants to find shortest path between cities avoiding tolls",
        key="comp_query_input"
    )
    k_comp = st.slider("Results to Compare", 3, 15, 6, key="k_comp_slider")

    if comp_query.strip():
        kw_scores = searcher.scores(comp_query, mode="keyword")
        sem_scores = searcher.scores(comp_query, mode="semantic")

        kw_idx = np.argsort(-kw_scores)[:k_comp]
        sem_idx = np.argsort(-sem_scores)[:k_comp]

        kw_df = df.iloc[kw_idx].copy(); kw_df["score"] = kw_scores[kw_idx]
        sem_df = df.iloc[sem_idx].copy(); sem_df["score"] = sem_scores[sem_idx]

        kw_codes = set(kw_df["qcode"])
        sem_codes = set(sem_df["qcode"])
        common = kw_codes & sem_codes
        jaccard = len(common) / len(kw_codes | sem_codes) if (kw_codes | sem_codes) else 0.0

        # Three metric cards with custom colored top borders
        m_col1, m_col2, m_col3 = st.columns(3)
        with m_col1:
            st.markdown(f"""
            <div class="comp-metric-card comp-metric-1">
                <div class="comp-metric-label">Shared Problems in Top-K</div>
                <div class="comp-metric-val">{len(common)} / {k_comp}</div>
            </div>
            """, unsafe_allow_html=True)
        with m_col2:
            st.markdown(f"""
            <div class="comp-metric-card comp-metric-2">
                <div class="comp-metric-label">Jaccard Similarity Overlap</div>
                <div class="comp-metric-val">{jaccard:.2f}</div>
            </div>
            """, unsafe_allow_html=True)
        with m_col3:
            st.markdown(f"""
            <div class="comp-metric-card comp-metric-3">
                <div class="comp-metric-label">Novel Semantic Discoveries</div>
                <div class="comp-metric-val">{len(sem_codes - kw_codes)}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('<div style="height: 18px;"></div>', unsafe_allow_html=True)
        c_left, c_right = st.columns(2)

        with c_left:
            st.markdown('<div class="section-heading" style="margin: 0 0 4px 0 !important; font-size: 1.02rem !important;">Sparse Keyword (TF-IDF)</div>', unsafe_allow_html=True)
            st.caption("Matches exact word tokens (often distracted by narrative words like Chef, city, game).")
            for rank, (_, row) in enumerate(kw_df.iterrows(), 1):
                is_shared = row["qcode"] in common
                card_class = "mode-card mode-card-both" if is_shared else "mode-card mode-card-kw"
                shared_tag = " <span class='score-pill' style='margin-left: 6px;'>In Both</span>" if is_shared else ""
                diff = str(row.get("difficulty", "medium")).lower()
                badge_class = f"badge-{diff}" if diff in DIFF_ORDER else "badge-medium"
                link = f"https://www.codechef.com{row['link']}" if str(row.get("link", "")).startswith("/") else f"https://www.codechef.com/problems/{row.get('qcode', '')}"
                
                st.markdown(f"""
                <div class="{card_class}">
                    <div style="font-weight: 600;">
                        #{rank}. <a href="{link}" target="_blank" class="problem-title" style="text-decoration: none; color: #1E293B;">{row['title']}</a> 
                        <span class="problem-code">({row['qcode']})</span>{shared_tag}
                    </div>
                    <div style="font-size: 0.82rem; color: #64748B; margin-top: 6px; display: flex; align-items: center; gap: 8px;">
                        <span class="badge-pill {badge_class}">{diff.upper()}</span>
                        <span class="score-pill">Score: {row['score']:.3f}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

        with c_right:
            st.markdown('<div class="section-heading" style="margin: 0 0 4px 0 !important; font-size: 1.02rem !important;">Dense Semantic (MiniLM-L6-v2)</div>', unsafe_allow_html=True)
            st.caption("Matches conceptual problem meaning and algorithm abstraction across disparate stories.")
            for rank, (_, row) in enumerate(sem_df.iterrows(), 1):
                is_shared = row["qcode"] in common
                card_class = "mode-card mode-card-both" if is_shared else "mode-card mode-card-sem"
                shared_tag = " <span class='score-pill' style='margin-left: 6px;'>In Both</span>" if is_shared else " <span class='badge-pill badge-beginner' style='margin-left: 6px;'>Semantic Match</span>"
                diff = str(row.get("difficulty", "medium")).lower()
                badge_class = f"badge-{diff}" if diff in DIFF_ORDER else "badge-medium"
                link = f"https://www.codechef.com{row['link']}" if str(row.get("link", "")).startswith("/") else f"https://www.codechef.com/problems/{row.get('qcode', '')}"

                st.markdown(f"""
                <div class="{card_class}">
                    <div style="font-weight: 600;">
                        #{rank}. <a href="{link}" target="_blank" class="problem-title" style="text-decoration: none; color: #1E293B;">{row['title']}</a> 
                        <span class="problem-code">({row['qcode']})</span>{shared_tag}
                    </div>
                    <div style="font-size: 0.82rem; color: #64748B; margin-top: 6px; display: flex; align-items: center; gap: 8px;">
                        <span class="badge-pill {badge_class}">{diff.upper()}</span>
                        <span class="score-pill">Score: {row['score']:.3f}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

# ----------------- TAB 3: Cluster Manifold -----------------
with tabs[2]:
    st.caption("Nonlinear 2D projection (UMAP) of dense problem embeddings with discovered thematic clusters.")
    
    if coords is not None and "cluster_label" in df.columns:
        plot_df = df.copy()
        plot_df["x"] = coords[:, 0]
        plot_df["y"] = coords[:, 1]
        plot_df["tags_display"] = plot_df["tags"].apply(lambda t: ", ".join(t[:5]) if isinstance(t, list) else "")
        
        c_filter1, c_filter2, c_filter3 = st.columns([2, 2, 2])
        with c_filter1:
            color_choice = st.selectbox("Color By", ["Cluster", "Difficulty"])
        with c_filter2:
            cluster_opts = ["All Clusters"] + sorted(plot_df["cluster_label"].dropna().unique().tolist())
            selected_cluster = st.selectbox("Filter Cluster", cluster_opts)
        with c_filter3:
            diff_opts = ["All Difficulties"] + [d.capitalize() for d in DIFF_ORDER]
            selected_diff = st.selectbox("Filter Difficulty", diff_opts)
            
        filtered_plot = plot_df.copy()
        if selected_cluster != "All Clusters":
            filtered_plot = filtered_plot[filtered_plot["cluster_label"] == selected_cluster]
        if selected_diff != "All Difficulties":
            filtered_plot = filtered_plot[filtered_plot["difficulty"].astype(str).str.lower().str.strip() == selected_diff.lower()]
            
        color_col = "cluster_label" if color_choice == "Cluster" else "difficulty"
        
        # Difficulty color mapping consistent with badges
        DIFF_COLOR_MAP = {
            "beginner": "#10B981", # Green
            "easy": "#0EA5E9",     # Cyan
            "medium": "#F59E0B",   # Amber
            "hard": "#EF4444",     # Red
            "challenge": "#8B5CF6" # Purple
        }
        
        # Vivid categorical palette for clusters
        CLUSTER_VIVID_PALETTE = [
            "#6366F1", "#14B8A6", "#F59E0B", "#EC4899", 
            "#3B82F6", "#10B981", "#8B5CF6", "#F97316", 
            "#06B6D4", "#E11D48", "#84CC16", "#A855F7"
        ]
        
        fig = go.Figure()
        group_idx = 0
        for cat_name, sub in filtered_plot.groupby(color_col):
            if color_choice == "Difficulty":
                marker_color = DIFF_COLOR_MAP.get(str(cat_name).lower(), "#64748B")
            else:
                marker_color = CLUSTER_VIVID_PALETTE[group_idx % len(CLUSTER_VIVID_PALETTE)]
                group_idx += 1
                
            hover_texts = [
                f"<b>{row['title']}</b> ({row['qcode']})<br>"
                f"Difficulty: {str(row['difficulty']).upper()}<br>"
                f"Cluster: {row.get('cluster_label', 'N/A')}<br>"
                f"Tags: {', '.join(row['tags'][:4]) if isinstance(row['tags'], list) and row['tags'] else 'None'}"
                for _, row in sub.iterrows()
            ]
            fig.add_trace(go.Scatter(
                x=sub["x"],
                y=sub["y"],
                mode="markers",
                name=str(cat_name),
                text=hover_texts,
                hoverinfo="text",
                marker=dict(size=7, opacity=0.85, color=marker_color)
            ))
            
        fig.update_layout(
            title=dict(
                text=f"2D Problem Semantic Space ({len(filtered_plot):,} problems shown)",
                font=dict(family="Space Grotesk, sans-serif", size=15)
            ),
            font=dict(family="Inter, sans-serif"),
            height=580,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            template="plotly_white",
            legend=dict(orientation="h", yanchor="bottom", y=-0.28, xanchor="center", x=0.5),
            margin=dict(l=10, r=10, t=35, b=60),
            xaxis=dict(title="UMAP Component 1", showgrid=True, gridcolor="#E2E8F0", zeroline=False),
            yaxis=dict(title="UMAP Component 2", showgrid=True, gridcolor="#E2E8F0", zeroline=False)
        )
        st.plotly_chart(fig, use_container_width=True)
        
        summary_path = "out/cluster_summary.csv"
        if os.path.exists(summary_path):
            st.markdown('<div class="section-heading" style="margin: 18px 0 10px 0 !important; font-size: 1.05rem !important;">Cluster Profiles & Exemplar Medoids</div>', unsafe_allow_html=True)
            sum_df = pd.read_csv(summary_path)
            st.dataframe(
                sum_df[["cluster_id", "top_topic_tags", "top_tfidf_terms", "size", "exemplar_1", "exemplar_2"]],
                column_config={
                    "cluster_id": "Cluster #",
                    "top_topic_tags": "Dominant Tags",
                    "top_tfidf_terms": "Characteristic Terms",
                    "size": "Problem Count",
                    "exemplar_1": "Medoid Problem 1",
                    "exemplar_2": "Medoid Problem 2"
                },
                use_container_width=True,
                hide_index=True
            )
            
        hidden_path = "out/hidden_matches.csv"
        if os.path.exists(hidden_path):
            with st.expander("Discovered Hidden Matches (High Semantic Similarity, Low Lexical Overlap)"):
                st.caption("Pairs of problems with completely different story words that share the exact same underlying algorithm.")
                h_df = pd.read_csv(hidden_path)
                st.dataframe(
                    h_df[["prob1_code", "prob2_code", "prob1_title", "prob2_title", "semantic_sim", "keyword_sim", "explanation"]].head(15),
                    column_config={
                        "prob1_code": "Query Code",
                        "prob2_code": "Match Code",
                        "prob1_title": "Query Title",
                        "prob2_title": "Match Title",
                        "semantic_sim": "Dense Sim (MiniLM)",
                        "keyword_sim": "Sparse Sim (TF-IDF)",
                        "explanation": "Algorithmic Rationale"
                    },
                    use_container_width=True,
                    hide_index=True
                )
    else:
        st.info("2D coordinates not available. Run python dejacode.py --data data/questions.csv to generate projection.")

# ----------------- TAB 4: Recommendations -----------------
with tabs[3]:
    st.caption("Select problems you have solved to get concept-aligned recommendations calibrated to your target difficulty level.")

    @st.cache_data(show_spinner=False)
    def get_accepted_solution_map(solutions_path="data/solutions.csv"):
        if not os.path.exists(solutions_path): return {}
        try:
            sol = pd.read_csv(solutions_path, usecols=["QCode", "SolutionID", "Status", "Language"])
            acc = sol[sol["Status"].astype(str).str.lower() == "accepted"].drop_duplicates(subset=["QCode"])
            return dict(zip(acc["QCode"], zip(acc["SolutionID"], acc["Language"])))
        except Exception:
            return {}

    sol_map = get_accepted_solution_map()

    # Controls Container Card
    st.markdown('<div class="control-card">', unsafe_allow_html=True)
    st.markdown('<div class="control-card-header">Select Solved Problems & Progression Goal</div>', unsafe_allow_html=True)
    
    preset = st.radio("Quick History Presets", ["None", "Beginner: FLOW017, CFRTEST", "Easy: SUMPAIR, CHEFLCM", "Medium: ANDTUPLE, DIVSUBS"], horizontal=True)
    
    col_rec_in, col_rec_opt = st.columns([2, 1])

    with col_rec_in:
        problem_options = [f"{row['qcode']} | {row['title']} ({str(row.get('difficulty', 'medium')).capitalize()})" for _, row in df.iterrows()]
        default_selected = [problem_options[0], problem_options[1]] if len(problem_options) >= 2 else []
        
        if "FLOW017" in preset:
            pre_selected = [opt for opt in problem_options if "FLOW017" in opt or "CFRTEST" in opt]
        elif "SUMPAIR" in preset:
            pre_selected = [opt for opt in problem_options if "SUMPAIR" in opt or "CHEFLCM" in opt]
        elif "ANDTUPLE" in preset:
            pre_selected = [opt for opt in problem_options if "ANDTUPLE" in opt or "DIVSUBS" in opt]
        else:
            pre_selected = default_selected

        selected_problems = st.multiselect(
            "Select problems you have solved:",
            options=problem_options,
            default=pre_selected
        )

    with col_rec_opt:
        step_choice = st.radio(
            "Progression Step",
            [
                "+1 Tier (Level Up - Recommended)",
                "0 (Reinforce Same Tier)",
                "-1 Tier (Review Foundations)"
            ],
            index=0
        )
        step_val = 1 if "+1" in step_choice else (0 if "0" in step_choice else -1)
        k_recs = st.slider("Recommendations Count", 3, 10, 5)

    st.markdown('</div>', unsafe_allow_html=True)

    # Clearly separated recommendations section
    st.markdown('<div style="height: 24px;"></div>', unsafe_allow_html=True)

    if selected_problems:
        solved_codes = [p.split(" | ")[0].strip() for p in selected_problems]
        recs_df = recommend_next(solved_codes, df, emb, k=k_recs, step=step_val)

        if not recs_df.empty:
            target_tier_name = recs_df.iloc[0]['difficulty'].upper()
            st.markdown(f'<div class="section-heading" style="margin-bottom: 14px !important;">Recommended Next Problems ({target_tier_name} tier)</div>', unsafe_allow_html=True)

            for idx, (_, rec) in enumerate(recs_df.iterrows(), 1):
                diff = str(rec.get("difficulty", "medium")).lower()
                badge_class = f"badge-{diff}" if diff in DIFF_ORDER else "badge-medium"
                link = f"https://www.codechef.com{rec['link']}" if str(rec.get("link", "")).startswith("/") else f"https://www.codechef.com/problems/{rec.get('qcode', '')}"
                tags_str = ", ".join(rec["tags"]) if isinstance(rec["tags"], list) and rec["tags"] else "None"
                acc_rate = f"{rec.get('acceptance_rate', 0.0) * 100:.1f}%" if "acceptance_rate" in rec else "N/A"
                accepted_n = f"{int(rec.get('accepted_count', 0)):,}" if "accepted_count" in rec else "N/A"
                subs_n = f"{int(rec.get('total_submissions', 0)):,}" if "total_submissions" in rec else "N/A"

                with st.container():
                    st.markdown(f"""
                    <div class="white-card">
                        <div style="display: flex; justify-content: space-between; align-items: flex-start;">
                            <div>
                                <h4 style="margin: 0;">
                                    #{idx}. <a href="{link}" target="_blank" class="problem-title" style="text-decoration: none; color: #4338CA;">{rec.get('title', 'Unknown')}</a>
                                    <span class="problem-code">({rec.get('qcode', 'N/A')})</span>
                                </h4>
                            </div>
                            <div style="display: flex; align-items: center; gap: 8px;">
                                <span class="badge-pill {badge_class}">{diff.upper()}</span>
                                <span class="score-pill">Concept Sim: {rec.get('similarity', 0.0):.3f}</span>
                            </div>
                        </div>
                        <div style="margin-top: 8px; font-size: 0.84rem; color: #64748B;">
                            <strong>Tags:</strong> {tags_str} &nbsp;|&nbsp;
                            <strong>Acceptance Rate:</strong> {acc_rate} ({accepted_n} / {subs_n} accepted)
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    with st.expander(f"Accepted Solution Code for {rec['qcode']}"):
                        sol_info = sol_map.get(rec["qcode"])
                        sol_id, lang = sol_info if sol_info else (None, None)
                        if sol_id:
                            code_snippet = get_code_snippet(sol_id)
                            if code_snippet:
                                st.caption(f"Accepted Submission: `{sol_id}` | Language: `{lang}`")
                                code_lang = "python" if "py" in str(lang).lower() else ("java" if "java" in str(lang).lower() else "cpp")
                                st.code(code_snippet, language=code_lang)
                            else:
                                st.info(f"Solution ID {sol_id} found in solutions.csv, but raw code file is not in program_codes/.")
                        else:
                            st.info(f"No accepted submission records found in solutions.csv for {rec['qcode']}.")
        else:
            st.warning("No candidate problems found matching the selected criteria.")
    else:
        st.info("Please select at least one solved problem above to generate recommendations.")
