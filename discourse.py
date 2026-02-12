import spacy
from fastcoref import spacy_component

def load_nlp_model():
    nlp = spacy.load("en_core_web_sm")
    nlp.add_pipe(
        "fastcoref",
        config={
            "model_architecture": "LingMessCoref",
            "model_path": "biu-nlp/lingmess-coref"
        }
    )
    return nlp

def resolve_coreference(nlp, text):
    doc = nlp(text)
    return doc._.resolved_text