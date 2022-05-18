import logging

import spacy

from cltl.nlp.api import NLP, Doc, NamedEntity, POS, Token

logger = logging.getLogger(__name__)


class SpacyNLP(NLP):
    def __init__(self, spacy_model: str = "en_core_web_sm"):
        self._nlp = spacy.load(spacy_model)

    def analyze(self, text: str) -> Doc:
        doc = self._nlp(text)

        tokens = [Token(token.text, POS[token.pos_], (token.idx, token.idx + len(token.text))) for token in doc]
        entities = [NamedEntity(entity.text, entity.label_, (entity.start_char, entity.end_char)) for entity in doc.ents]

        return Doc(tokens, entities)