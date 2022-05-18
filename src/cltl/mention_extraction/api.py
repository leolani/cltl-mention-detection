import abc
import logging
from dataclasses import dataclass
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
    def create_person(cls, label: str, id: str, uri: str):
        return cls(label, ["person"], id, uri)


@dataclass
class ImageMention:
    visual: str
    detection: str
    source:Source
    image: str
    region: Tuple[int, int, int, int]
    item: Entity
    confidence: float
    context_id: str


@dataclass
class TextMention:
    chat: str
    turn: str
    author: Entity
    utterance: str
    position: str
    item: Entity
    confidence: float
    context_id: str
    utterance_type: UtteranceType = UtteranceType.IMAGE_MENTION


_IMAGE_SOURCE = Source("front-camera", ["sensor"], "http://cltl.nl/leolani/inputs/front-camera")


class MentionExtractor(abc.ABC):
    def extract_text_mentions(self, mentions: List[Mention], scenario_id: str) -> List[TextMention]:
        raise NotImplementedError()

    def extract_object_mentions(self, mentions: List[Mention], scenario_id: str) -> List[ImageMention]:
        raise NotImplementedError()

    def extract_face_mentions(self, mentions: List[Mention], scenario_id: str) -> List[ImageMention]:
        raise NotImplementedError()