"""Interactive Streamlit demo for Gujarati-English code-mixed text."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from src.classification import MultiTaskClassifier
from src.language_id import identify_languages
from src.preprocessing import preprocess_pipeline
from src.transliteration import GujaratiTransliterator

st.set_page_config(page_title="Gujarati-English NLP", page_icon="ગુજરાતી", layout="wide")
st.title("Gujarati-English Code-Mixed Text Understanding")
st.caption("Explore normalization, word-level language identification, transliteration, and baseline predictions.")

with st.sidebar:
    st.header("Configuration")
    show_symbols = st.checkbox("Include punctuation and numbers in LID", value=True)
    fuzzy_cutoff = st.slider("Transliteration fuzzy match threshold", 0.65, 0.98, 0.82, 0.01)
    st.info("The included classifiers are lightweight baselines. Train on representative labeled data for deployment.")

text = st.text_area("Enter a Gujarati-English message", "Kem cho bhai? Aa service saras che!", height=130, placeholder="Type a code-mixed message…")
if text.strip():
    try:
        cleaned = preprocess_pipeline(text)
        transliterator = GujaratiTransliterator(cutoff=fuzzy_cutoff)
        translated = transliterator.transliterate(cleaned)
        tagged = identify_languages(cleaned)
        if not show_symbols:
            tagged = [(token, tag) for token, tag in tagged if tag != "SYMBOL_NUMBER"]
        prediction = MultiTaskClassifier().predict(cleaned)

        left, right = st.columns(2)
        with left:
            st.subheader("Preprocessed text")
            st.code(cleaned, language=None)
        with right:
            st.subheader("Gujarati transliteration")
            st.code(translated, language=None)

        st.subheader("Word-level language identification")
        frame = pd.DataFrame(tagged, columns=["Token", "Language tag"])
        st.dataframe(frame, use_container_width=True, hide_index=True)

        st.subheader("Predictions")
        cols = st.columns(2)
        for col, name, result in [(cols[0], "Sentiment", prediction.sentiment), (cols[1], "Intent", prediction.intent)]:
            with col:
                st.metric(name, result.label, f"{result.confidence:.0%} confidence")
                st.progress(float(result.confidence), text=f"{result.label}: {result.confidence:.0%}")
                if result.probabilities:
                    st.caption(" | ".join(f"{label}: {score:.0%}" for label, score in result.probabilities.items()))
    except (OSError, ValueError, TypeError, RuntimeError) as exc:
        st.error(f"Could not process this text: {exc}")
else:
    st.info("Enter a message above to see the analysis.")
