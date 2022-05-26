import abc
import logging
from dataclasses import dataclass
from typing import List, Tuple

from cltl.combot.infra.time_util import timestamp_now
from cltl.commons.discrete import UtteranceType
from emissor.representation.scenario import Mention

from cltl.mention_extraction.api import MentionExtractor
import cltl.nlp.api as nlp

logger = logging.getLogger(__name__)


_ACCEPTED_OBJECTS = {object_type.name for object_type in nlp.ObjectType}


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
    confidence: float
    context_id: str
    timestamp: int
    utterance_type: UtteranceType = UtteranceType.TEXT_MENTION


_IMAGE_SOURCE = Source("front-camera", ["sensor"], "http://cltl.nl/leolani/inputs/front-camera")


class MentionDetector(abc.ABC):
    def filter_mentions(self, mentions: List[Mention], scenario_id: str) -> List[Mention]:
        return mentions


class TextMentionDetector(MentionDetector):
    def filter_mentions(self, mentions: List[Mention], scenario_id: str) -> List[Mention]:
        filtered = []
        for mention in mentions:
            annotations = [annotation for annotation in mention.annotations
                           if (annotation.type == nlp.NamedEntity.__name__ or annotation.type == nlp.Entity.__name__)]
            if annotations:
                filtered.append(Mention(mention.id, mention.segment, annotations))

        return filtered


class NewFaceMentionDetector(MentionDetector):
    def __init__(self):
        self._scenario_id = None
        self._faces = set()

    def filter_mentions(self, mentions: List[Mention], scenario_id: str) -> List[Mention]:
        if scenario_id != self._scenario_id:
            self._scenario_id = scenario_id
            self._faces = set()

        new_face_mentions = [mention for mention in mentions
                             if mention.annotations and mention.annotations[0].value.id not in self._faces]

        self._faces = self._faces | {mention.annotations[0].value.id for mention in new_face_mentions}

        return new_face_mentions


class ObjectMentionDetector(MentionDetector):
    def filter_mentions(self, mentions: List[Mention], scenario_id: str) -> List[Mention]:
        return [mention for mention in mentions
                if mention.annotations and mention.annotations[0].value.type in _ACCEPTED_OBJECTS]


class DefaultMentionExtractor(MentionExtractor):
    def __init__(self, text_detector: MentionDetector, face_detector: MentionDetector, object_detector: MentionDetector):
        self._text_detector = text_detector
        self._face_detector = face_detector
        self._object_detector = object_detector

    def extract_text_mentions(self, mentions: List[Mention], scenario_id: str) -> List[TextMention]:
        return [self.create_text_mention(mention, scenario_id)
                for mention in self._text_detector.filter_mentions(mentions, scenario_id)]

    def extract_object_mentions(self, mentions: List[Mention], scenario_id: str) -> List[ImageMention]:
        return [self.create_object_mention(mention, scenario_id)
                for mention in self._face_detector.filter_mentions(mentions, scenario_id)]

    def extract_face_mentions(self, mentions: List[Mention], scenario_id: str) -> List[ImageMention]:
        return [self.create_face_mention(mention, scenario_id)
                for mention in self._object_detector.filter_mentions(mentions, scenario_id)]

    def create_face_mention(self, mention: Mention, scenario_id: str):
        image_id = mention.id
        image_path = mention.id

        mention_id = mention.id
        bounds = mention.segment[0].to_tuple()
        face_id = mention.annotations[0].value
        confidence = 1.0

        return ImageMention(image_id, mention_id, _IMAGE_SOURCE, image_path, bounds,
                            Entity.create_person(None, face_id, None),
                            confidence, scenario_id, timestamp_now())

    def create_object_mention(self, mention: Mention, scenario_id: str):
        image_id = mention.id
        image_path = mention.id

        mention_id = mention.id
        bounds = mention.segment[0].to_tuple()
        # TODO multiple?
        object_label = mention.annotations[0].value
        confidence = 1.0

        return ImageMention(image_id, mention_id, _IMAGE_SOURCE, image_path, bounds,
                            Entity(None, [object_label], None, None),
                            confidence, scenario_id, timestamp_now())

    def create_text_mention(self, mention: Mention, scenario_id: str):
        author = Entity.create_person("SPEAKER", None, None)

        utterance = ""

        segment = mention.segment[0]
        signal_id = segment.container_id
        entity_text = mention.annotations[0].value.text
        entity_type = mention.annotations[0].value.label
        confidence = 1.0

        return TextMention(scenario_id, signal_id, author, utterance, f"{segment.start} - {segment.stop}",
                           Entity(entity_text, [entity_type], None, None),
                           confidence, scenario_id, timestamp_now())
