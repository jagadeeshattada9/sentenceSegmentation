import nltk
from nltk.tokenize import sent_tokenize
from discourse import resolve_coreference

nltk.download("punkt")

def discourse_aware_segmentation(text):
    resolved_text = resolve_coreference(text)
    sentences = sent_tokenize(resolved_text)
    return sentences