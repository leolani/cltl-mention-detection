import unittest

from cltl.nlp.api import POS, EntityType
from cltl.nlp.spacy_nlp import SpacyNLP


class TestSpacyNLP(unittest.TestCase):
    def setUp(self) -> None:
        self.nlp = SpacyNLP()

    def test_analyze_tokens(self):
        doc = self.nlp.analyze("This is a text sentence.")
        self.assertEqual(6, len(doc.tokens))
        self.assertEqual("This", doc.tokens[0].text)
        self.assertEqual((0, 4), doc.tokens[0].segment)
        self.assertEqual(POS.PRON, doc.tokens[0].pos)

        self.assertEqual(".", doc.tokens[5].text)
        self.assertEqual((23, 24), doc.tokens[5].segment)
        self.assertEqual(POS.PUNCT, doc.tokens[5].pos)

    def test_analyze_tokens_with_entity(self):
        doc = self.nlp.analyze("Piek travels to New York.")

        self.assertEqual(6, len(doc.tokens))

        self.assertEqual("Piek", doc.tokens[0].text)
        self.assertEqual((0, 4), doc.tokens[0].segment)
        self.assertEqual(POS.PROPN, doc.tokens[0].pos)

        self.assertEqual("New", doc.tokens[3].text)
        self.assertEqual((16, 19), doc.tokens[3].segment)
        self.assertEqual(POS.PROPN, doc.tokens[3].pos)

        self.assertEqual("York", doc.tokens[4].text)
        self.assertEqual((20, 24), doc.tokens[4].segment)
        self.assertEqual(POS.PROPN, doc.tokens[4].pos)

        self.assertEqual(".", doc.tokens[5].text)
        self.assertEqual((24, 25), doc.tokens[5].segment)
        self.assertEqual(POS.PUNCT, doc.tokens[5].pos)

    def test_analyze_named_entities(self):
        doc = self.nlp.analyze("Piek travels to New York.")

        self.assertEqual(2, len(doc.named_entities))

        self.assertEqual("Piek", doc.named_entities[0].text)
        self.assertEqual("PERSON", doc.named_entities[0].label)
        self.assertEqual((0, 4), doc.named_entities[0].segment)

        self.assertEqual("New York", doc.named_entities[1].text)
        self.assertEqual("GPE", doc.named_entities[1].label)
        self.assertEqual((16, 24), doc.named_entities[1].segment)

    def test_analyze_empty(self):
        doc = self.nlp.analyze("")
        self.assertEqual(0, len(doc.tokens))
        self.assertEqual(0, len(doc.named_entities))

    def test_analyze_objet_entities(self):
        doc = self.nlp.analyze("I see the book in the waste bin.")

        self.assertEqual(2, len(doc.entities))
        self.assertEqual(EntityType.SPEAKER, doc.entities[0].type)
        self.assertEqual(EntityType.OBJECT, doc.entities[1].type)
        self.assertEqual("book", doc.entities[1].text)

    def test_analyze_entities(self):
        doc = self.nlp.analyze("I know you from school")

        self.assertEqual(2, len(doc.entities))
        self.assertEqual(EntityType.SPEAKER, doc.entities[0].type)
        self.assertEqual(EntityType.HEARER, doc.entities[1].type)

