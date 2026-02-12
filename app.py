import streamlit as st
from discourse import load_nlp_model, resolve_coreference
import nltk
from nltk.tokenize import sent_tokenize

nltk.download("punkt")

st.set_page_config(page_title="Sentence Segmentation with Discourse Processing")

st.title("Sentence Segmentation Tool using Discourse Processing")

st.write(
    "This tool segments text and resolves discourse references like 'she refers to Rita'."
)

# 🔥 Load model only once
@st.cache_resource
def get_model():
    return load_nlp_model()

nlp = get_model()

text = st.text_area("Enter text:", height=200)

if st.button("Process Text"):
    if text.strip():
        resolved_text = resolve_coreference(nlp, text)
        sentences = sent_tokenize(resolved_text)

        st.subheader("Processed Sentences:")
        for i, sentence in enumerate(sentences, 1):
            st.write(f"{i}. {sentence}")
    else:
        st.warning("Please enter some text.")