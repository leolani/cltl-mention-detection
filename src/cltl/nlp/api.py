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
    AIRPLANE = "airplane"
    APPLE = "apple"
    BACKPACK = "backpack"
    BANANA = "banana"
    BASEBALL_BAT = "baseball bat"
    BASEBALL_GLOVE = "baseball glove"
    BEAR = "bear"
    BED = "bed"
    BENCH = "bench"
    BICYCLE = "bicycle"
    BIRD = "bird"
    BOAT = "boat"
    BOOK = "book"
    BOTTLE = "bottle"
    BOWL = "bowl"
    BROCCOLI = "broccoli"
    BUS = "bus"
    CAKE = "cake"
    CAR = "car"
    CARROT = "carrot"
    CAT = "cat"
    CELL_PHONE = "cell phone"
    CHAIR = "chair"
    CLOCK = "clock"
    COUCH = "couch"
    COW = "cow"
    CUP = "cup"
    DOG = "dog"
    DONUT = "donut"
    ELEPHANT =  "elephant"
    FIRE_HYDRANT = "fire hydrant"
    FORK = "fork"
    FRISBEE = "frisbee"
    GIRAFFE = "giraffe"
    HAIR_DRIER = "hair drier"
    HANDBAG = "handbag"
    HORSE = "horse"
    HOT_DOG = "hot dog"
    KEYBOARD = "keyboard"
    KITE = "kite"
    KNIFE = "knife"
    LAPTOP = "laptop"
    MICROWAVE = "microwave"
    MOTORCYCLE = "motorcycle"
    MOUSE = "mouse"
    ORANGE = "orange"
    OVEN = "oven"
    PARKING_METER = "parking meter"
    PERSON = "person"
    PHONE = "phone"
    PIZZA = "pizza"
    POTTED_PLANT = "potted plant"
    REFRIGERATOR = "refrigerator"
    REMOTE = "remote"
    SANDWICH = "sandwich"
    SCISSORS = "scissors"
    SHEEP = "sheep"
    SINK = "sink"
    SKATEBOARD = "skateboard"
    SKIS = "skis"
    SNOWBOARD = "snowboard"
    SPOON = "spoon"
    SPORTS_BALL = "sports ball"
    STOP_SIGN = "stop sign"
    SUITCASE = "suitcase"
    SURFBOARD = "surfboard"
    TABLE = "table"
    TEDDY_BEAR = "teddy bear"
    TENNIS_RACKET = "tennis racket"
    TIE = "tie"
    TOASTER = "toaster"
    TOILET = "toilet"
    TOOTHBRUSH = "toothbrush"
    TRAFFIC_LIGHT = "traffic light"
    TRAIN = "train"
    TRUCK= "truck"
    TV = "tv"
    UMBRELLA = "umbrella"
    VASE = "vase"
    WINE_GLASS = "wine glass"
    ZEBRA = "zebra"


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