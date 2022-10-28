import abc
import logging
from dataclasses import dataclass, field
from typing import List, Tuple

from cltl.commons.discrete import UtteranceType
from emissor.representation.scenario import Mention

logger = logging.getLogger(__name__)


@dataclass
class Source:
    label: str
    type: List[str]
    uri: str


@dataclass
class Entity:
    label: str
    type: List[str]
    id: str
    uri: str

    @classmethod
    def create_person(cls, label: str, id_: str, uri: str):
        return cls(label, ["person"], id_, uri)


@dataclass
class ImageMention:
    visual: str
    detection: str
    source: Source
    image: str
    region: Tuple[int, int, int, int]
    item: Entity
    # TODO type Perspective
    perspective: dict
    confidence: float
    context_id: str
    timestamp: int
    utterance_type: UtteranceType = UtteranceType.IMAGE_MENTION


@dataclass
class TextMention:
    chat: str
    turn: str
    author: Entity
    utterance: str
    position: str
    item: Entity
    # TODO type Perspective
    perspective: dict
    confidence: float
    context_id: str
    timestamp: int
    utterance_type: UtteranceType = UtteranceType.TEXT_MENTION


@dataclass
class Perspective:
    emotion: str
    confidence: float

    @classmethod
    def create_emotion(cls, emotion: str, confidence: str):
        return cls(emotion,confidence)


@dataclass
class TextPerspective:
    chat: str
    turn: str
    author: Entity
    utterance: str
    position: str
    item: Entity
    perspective: Perspective
    context_id: str
    timestamp: int
    utterance_type: UtteranceType = UtteranceType.TEXT_ATTRIBUTION


@dataclass
class ImagePerspective:
    visual: str
    detection: str
    source: Source
    image: str
    region: Tuple[int, int, int, int]
    item: Entity
    perspective: Perspective
    context_id: str
    timestamp: int
    utterance_type: UtteranceType = UtteranceType.IMAGE_ATTRIBUTION


class MentionExtractor(abc.ABC):
    def extract_text_mentions(self, mentions: List[Mention], scenario_id: str) -> List[TextMention]:
        raise NotImplementedError()

    def extract_text_perspective(self, mentions: List[Mention], scenario_id: str) -> List[TextPerspective]:
        raise NotImplementedError()

    def extract_object_mentions(self, mentions: List[Mention], scenario_id: str) -> List[ImageMention]:
        raise NotImplementedError()

    def extract_face_mentions(self, mentions: List[Mention], scenario_id: str) -> List[ImageMention]:
        raise NotImplementedError()

    def extract_face_perspective(self, mentions: List[Mention], scenario_id: str) -> List[ImagePerspective]:
        raise NotImplementedError()