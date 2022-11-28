import logging
from enum import Enum
from typing import List

import spacy

from cltl.nlp.api import NLP, Doc, NamedEntity, POS, Token, Entity, EntityType, ObjectType

logger = logging.getLogger(__name__)


_ACCEPTED_OBJECTS = {object_type.name.lower() for object_type in ObjectType}
_RELATIONS = ('nsubj', 'nsubjpass', 'dobj', 'prep', 'pcomp', 'acomp')


class SpacyNLP(NLP):
    def __init__(self, spacy_model: str = "en_core_web_sm", relations: List[str] = _RELATIONS):
        self._nlp = spacy.load(spacy_model)
        self._relations = set(relations)

    def analyze(self, text: str) -> Doc:
        doc = self._nlp(text)

        tokens = [Token(token.text, POS[token.pos_], (token.idx, token.idx + len(token.text))) for token in doc]
        named_entities = [NamedEntity(entity.text, entity.label_, (entity.start_char, entity.end_char)) for entity in doc.ents]
        entities = self._analyze_entities(doc)

        return Doc(tokens, named_entities, entities)

    def _analyze_entities(self, doc):
        predicates = {}

        entities = []
        for token in doc:
            if token.dep_ in self._relations:
                head_id = token.head.i
                if head_id not in predicates:
                    predicates[head_id] = dict()

                type = None
                if token.pos_ == "PRON":
                    if token.text.lower() == 'i':
                        type = EntityType.SPEAKER
                    elif token.text.lower() == 'you':
                        type = EntityType.HEARER
                elif token.pos_ == "NOUN":
                    if token.lemma_.lower() in _ACCEPTED_OBJECTS:
                        type = EntityType.OBJECT

                if type:
                    entities.append(Entity(token.text, type, (token.idx, token.idx + len(token.text))))

                predicates[head_id][token.dep_] = token.lemma_

        return entities
