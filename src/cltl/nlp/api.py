import abc
import dataclasses
from enum import Enum, auto
from typing import List, Tuple


class POS(Enum):
    ADJ = auto()    # adjective
    ADP = auto()    # adposition
    ADV = auto()    # adverb
    AUX = auto()    # auxiliary
    CCONJ = auto()  # coordinating conjunction
    DET = auto()    # determiner
    INTJ = auto()   # interjection
    NOUN = auto()   # noun
    NUM = auto()    # numeral
    PART = auto()   # particle
    PRON = auto()   # pronoun
    PROPN = auto()  # proper noun
    PUNCT = auto()  # punctuation
    SCONJ = auto()  # subordinating conjunction
    SPACE = auto()  # space
    SYM = auto()    # symbol
    VERB = auto()   # verb
    X = auto()      # other


class EntityType(Enum):
    SPEAKER = auto()
    HEARER = auto()
    OBJECT = auto()


class ObjectType(Enum):
    CAT = "cat"
    DOG = "dog"
    PHONE = "phone"
    TV = "tv"
    PERSON = "person"
    LAPTOP = "laptop"
    BOTTLE = "bottle"
    CUP = "cup"
    BOOK = "book"


@dataclasses.dataclass
class Token:
    text: str
    pos: POS
    segment: Tuple[int, int]


@dataclasses.dataclass
class Entity:
    text: str
    type: EntityType
    segment: Tuple[int, int]

    @property
    def label(self):
        return self.type.name.lower()


@dataclasses.dataclass
class NamedEntity:
    text: str
    label: str
    segment: Tuple[int, int]


@dataclasses.dataclass
class Doc:
    tokens: List[Token]
    named_entities: List[NamedEntity]
    entities: List[Entity]


class NLP(abc.ABC):
    def analyze(self, text: str) -> Doc:
        raise NotImplementedError()