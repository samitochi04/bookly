"""
app.py
=============================================================
Bookly, the book rating predictor web application.

Run locally with:
    streamlit run app.py

The app offers two ways to get a prediction:
  1. Single book: fill in a short form, get one predicted rating.
  2. Batch upload: drop in a CSV of books, get a table back
     with a predicted_rating column and a download button.

All prediction logic lives in predictor.py so the UI stays
focused on presentation.
=============================================================
"""

import streamlit as st
import pandas as pd

# ---------------------------------------------------------------
# Guard: this file must be launched with "streamlit run", not with
# plain "python". Running it with python produces hundreds of
# "missing ScriptRunContext" warnings and no app. If we detect that
# case, print a clear instruction and exit instead.
# ---------------------------------------------------------------
from streamlit.runtime.scriptrunner import get_script_run_ctx

if get_script_run_ctx() is None:
    import sys
    import subprocess

    script = __file__
    print("\nBookly must be started with Streamlit, not with plain python.\n")
    print("Launching it for you now. If this does not open, run:\n")
    print(f"    streamlit run {script}\n")
    # Relaunch the file the correct way, then stop this bare process.
    subprocess.run(["streamlit", "run", script] + sys.argv[1:])
    sys.exit(0)

from predictor import (
    predict_rating,
    predict_batch,
    COMMON_PUBLISHERS,
    LANGUAGE_OPTIONS,
)

# ---------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------
st.set_page_config(
    page_title="Bookly",
    page_icon="📖",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------
# Styling. A warm library card-catalog identity: ink text on a
# paper background, with a deep book-spine green as the one accent.
# ---------------------------------------------------------------
st.markdown(
    """
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,600&family=Inter:wght@400;500;600&display=swap');

      :root {
        --paper:   #f5f1e8;
        --ink:     #1f1b16;
        --spine:   #2f5d50;
        --spine-2: #3d7565;
        --edge:    #d9cfbb;
        --muted:   #6b6152;
      }

      .stApp { background: var(--paper); }

      /* Force all body text, widget labels, and markdown to ink color.
         Without this, labels inherit Streamlit's theme text color, which
         is near-white in dark mode and becomes invisible on our paper
         background. These rules make the app readable in any theme. */
      .stApp, .stApp p, .stApp li, .stApp label,
      .stApp span, .stApp div,
      [data-testid="stWidgetLabel"],
      [data-testid="stWidgetLabel"] p,
      .stMarkdown, .stMarkdown p {
        color: var(--ink);
      }

      /* Slider min/max range numbers and tick labels */
      [data-testid="stTickBarMin"],
      [data-testid="stTickBarMax"],
      .stSlider label { color: var(--muted) !important; }

      /* Headings use a characterful serif; body stays clean sans */
      h1, h2, h3 { font-family: 'Fraunces', Georgia, serif !important; color: var(--ink) !important; }
      html, body, [class*="css"] { font-family: 'Inter', system-ui, sans-serif; color: var(--ink); }

      /* Masthead */
      .masthead {
        border-bottom: 2px solid var(--ink);
        padding-bottom: 14px;
        margin-bottom: 6px;
      }
      .masthead .title {
        font-family: 'Fraunces', Georgia, serif;
        font-size: 3.2rem;
        font-weight: 600;
        letter-spacing: -0.02em;
        line-height: 1;
        margin: 0;
      }
      .masthead .sub {
        font-size: 0.95rem;
        color: var(--muted);
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-top: 8px;
      }

      /* Result card, styled like a library index card */
      .result-card {
        background: #fffdf7;
        border: 1px solid var(--edge);
        border-left: 6px solid var(--spine);
        border-radius: 4px;
        padding: 26px 28px;
        margin-top: 8px;
        box-shadow: 0 1px 0 rgba(0,0,0,0.04);
      }
      .result-card .score {
        font-family: 'Fraunces', Georgia, serif;
        font-size: 3.4rem;
        font-weight: 600;
        color: var(--spine);
        line-height: 1;
      }
      .result-card .outof { font-size: 1.3rem; color: var(--muted); }
      .result-card .band  { font-size: 0.95rem; color: var(--muted); margin-top: 10px; }

      /* Primary button in spine green */
      .stButton > button {
        background: var(--spine);
        color: #fffdf7;
        border: none;
        border-radius: 4px;
        padding: 0.55rem 1.3rem;
        font-weight: 600;
        letter-spacing: 0.02em;
      }
      .stButton > button:hover { background: var(--spine-2); color: #fffdf7; }

      /* Tabs */
      .stTabs [data-baseweb="tab-list"] { gap: 6px; }
      .stTabs [data-baseweb="tab"] {
        font-weight: 600;
        color: var(--muted);
      }
      .stTabs [aria-selected="true"] { color: var(--spine) !important; }

      /* ── Widget backgrounds ─────────────────────────────────────
         Streamlit inherits widget backgrounds from its internal theme
         variables. In dark mode those variables go dark and override
         whatever config.toml says. We kill that by explicitly setting
         the background on every widget container we use.          */

      /* Text inputs and number inputs */
      [data-testid="stTextInput"]       > div,
      [data-testid="stTextInput"]       > div > div,
      [data-testid="stNumberInput"]     > div,
      [data-testid="stNumberInputContainer"],
      [data-baseweb="input"],
      [data-baseweb="input"] > div,
      input[type="number"],
      input[type="text"] {
        background-color: #fffdf7 !important;
        color: var(--ink) !important;
      }

      /* Select / dropdown */
      [data-baseweb="select"],
      [data-baseweb="select"] > div,
      [data-baseweb="popover"],
      [data-baseweb="menu"],
      [data-baseweb="list"],
      [role="listbox"],
      [role="option"] {
        background-color: #fffdf7 !important;
        color: var(--ink) !important;
      }

      /* File uploader drop zone */
      [data-testid="stFileUploader"] section,
      [data-testid="stFileUploaderDropzone"],
      [data-testid="stFileUploader"] > div,
      .stFileUploader section {
        background-color: #fffdf7 !important;
        border-color: var(--edge) !important;
        color: var(--ink) !important;
      }

      /* File uploader instruction text */
      [data-testid="stFileUploaderDropzone"] *,
      [data-testid="stFileUploader"] small,
      [data-testid="stFileUploader"] span {
        color: var(--muted) !important;
      }

      /* Checkbox */
      [data-testid="stCheckbox"] span {
        color: var(--ink) !important;
      }

      /* Number input +/- buttons */
      [data-testid="stNumberInput"] button {
        background-color: var(--edge) !important;
        color: var(--ink) !important;
      }

      /* Star row */
      .stars { font-size: 1.5rem; letter-spacing: 2px; color: var(--spine); }

      /* Recolor Streamlit's default red accent to the spine green */
      [data-baseweb="slider"] [role="slider"] { background: var(--spine) !important; }
      [data-baseweb="slider"] div[style*="rgb(255"] { background: var(--spine) !important; }
      .stSlider [data-testid="stTickBar"] ~ div div { background: var(--spine) !important; }
      .stSlider div[data-baseweb="slider"] > div > div { background: var(--spine) !important; }
      [data-testid="stSliderThumbValue"] { color: var(--spine) !important; }
      input:focus { border-color: var(--spine) !important; }
      [data-baseweb="input"]:focus-within,
      [data-baseweb="select"]:focus-within { border-color: var(--spine) !important; }
      [data-testid="stCheckbox"] [data-baseweb="checkbox"] [data-checked="true"] {
        background: var(--spine) !important; border-color: var(--spine) !important;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------
# Masthead
# ---------------------------------------------------------------
st.markdown(
    """
    <div class="masthead">
      <div class="title">Bookly</div>
      <div class="sub">Book rating predictor &middot; DSTI School of Engineering</div>
    </div>
    """,
    unsafe_allow_html=True,
)
st.write("")
st.markdown(
    "Estimate the average rating a book will earn from its catalog details. "
    "The model was trained on roughly 11,000 books and reads the same signals a "
    "librarian might: who wrote it, who published it, how long it is, and how "
    "widely it has been read."
)


def star_row(score):
    """Return a unicode star string for a 0 to 5 score, in half steps."""
    full = int(score)
    half = 1 if (score - full) >= 0.5 else 0
    empty = 5 - full - half
    return "★" * full + ("½" if half else "") + "☆" * empty


def render_result(score):
    """Render the prediction result card."""
    low = max(1.0, score - 0.26)
    high = min(5.0, score + 0.26)
    st.markdown(
        f"""
        <div class="result-card">
          <span class="score">{score:.2f}</span>
          <span class="outof"> / 5.00</span>
          <div class="stars">{star_row(score)}</div>
          <div class="band">
            Typical error is around 0.18 stars, so the real rating most likely
            falls between {low:.2f} and {high:.2f}.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------
# Two modes: single book and batch upload
# ---------------------------------------------------------------
tab_single, tab_batch = st.tabs(["Single book", "Upload a CSV"])

# ---- Single book ----------------------------------------------
with tab_single:
    st.subheader("Book details")

    col_a, col_b = st.columns(2)

    with col_a:
        title = st.text_input(
            "Title",
            value="The Night Circus",
            help="Used only to detect a series number and title length.",
        )
        authors = st.text_input(
            "Author(s)",
            value="Erin Morgenstern",
            help="Separate multiple authors with a forward slash, for example: Neil Gaiman/Terry Pratchett",
        )
        publisher = st.selectbox(
            "Publisher",
            options=COMMON_PUBLISHERS + ["Other (type below)"],
            index=0,
        )
        if publisher == "Other (type below)":
            publisher = st.text_input("Publisher name", value="")

    with col_b:
        num_pages = st.number_input("Number of pages", min_value=1, max_value=6000, value=387)
        ratings_count = st.number_input("Number of ratings", min_value=0, value=250000, step=1000)
        text_reviews_count = st.number_input("Number of text reviews", min_value=0, value=15000, step=500)
        pub_year = st.slider("Publication year", 1900, 2026, 2011)
        language = st.selectbox("Language", options=LANGUAGE_OPTIONS, index=0)

    is_series = st.checkbox("This book is part of a numbered series")
    series_num = 0
    if is_series:
        series_num = st.number_input("Series number", min_value=1, max_value=50, value=1)

    st.write("")
    if st.button("Predict rating", type="primary"):
        # Derive the two title-based features the same way training did
        title_words = len(title.split()) if title.strip() else 3
        num_authors = len([a for a in authors.split("/") if a.strip()]) or 1

        score = predict_rating(
            num_pages=num_pages,
            ratings_count=ratings_count,
            text_reviews_count=text_reviews_count,
            pub_year=pub_year,
            series_num=series_num,
            num_authors=num_authors,
            title_words=title_words,
            publisher=publisher,
            authors=authors,
        )
        render_result(score)

# ---- Batch upload ---------------------------------------------
with tab_batch:
    st.subheader("Predict a whole list at once")
    st.markdown(
        "Upload a CSV with the same columns as the original dataset "
        "(`title`, `authors`, `num_pages`, `ratings_count`, "
        "`text_reviews_count`, `publisher`, `publication_date`). "
        "Missing columns are filled with sensible defaults."
    )

    uploaded = st.file_uploader("Choose a CSV file", type=["csv"])

    if uploaded is not None:
        try:
            books = pd.read_csv(uploaded, engine="python", on_bad_lines="skip")
            books.columns = books.columns.str.strip()
            st.write(f"Loaded {len(books)} books.")

            with st.spinner("Scoring books..."):
                scored = predict_batch(books)

            st.dataframe(
                scored[[c for c in ["title", "authors", "predicted_rating"] if c in scored.columns]],
                use_container_width=True,
                hide_index=True,
            )

            csv_bytes = scored.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download results as CSV",
                data=csv_bytes,
                file_name="bookly_predictions.csv",
                mime="text/csv",
            )
        except Exception as error:
            st.error(f"Could not read that file. Please check it is a valid CSV. Details: {error}")

# ---------------------------------------------------------------
# Footer
# ---------------------------------------------------------------
st.write("")
st.markdown(
    """
    <div style="border-top:1px solid var(--edge); margin-top:28px; padding-top:12px;
                color:var(--muted); font-size:0.85rem;">
      Bookly predicts an average rating from catalog metadata alone. It cannot read
      the book, so taste, writing quality, and word of mouth are outside its view.
      Treat the number as an informed estimate, not a verdict.
    </div>
    """,
    unsafe_allow_html=True,
)