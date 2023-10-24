import abc
import logging
from enum import Enum
from typing import List

from cltl.combot.infra.time_util import timestamp_now
from cltl.combot.event.emissor import ConversationalAgent
from emissor.representation.scenario import Mention, class_type, Annotation

import cltl.nlp.api as nlp

from cltl.mention_extraction.api import MentionExtractor, ImagePerspective, TextPerspective, Perspective, Source, \
    TextMention, ImageMention, Entity

logger = logging.getLogger(__name__)


_ACCEPTED_OBJECTS = {object_type.value.lower() for object_type in nlp.ObjectType}


_IMAGE_SOURCE = Source("front-camera", ["sensor"], "http://cltl.nl/leolani/inputs/front-camera")


class MentionDetector(abc.ABC):
    """Detect mentions that contribute to knowledge.

    Select a subset of Mentions for further extraction, e.g. to prevent duplication or reduce
    the amount of information.
    """
    def filter_mentions(self, mentions: List[Mention], scenario_id: str) -> List[Mention]:
        return mentions


class TextMentionDetector(MentionDetector):
    def filter_mentions(self, mentions: List[Mention], scenario_id: str) -> List[Mention]:
        filtered = []
        for mention in mentions:
            annotations = [annotation for annotation in mention.annotations if self._is_entity(annotation)]
            if annotations:
                filtered.append(Mention(mention.id, mention.segment, annotations))

        return filtered

    def _is_entity(self, annotation):
        if annotation.type == nlp.NamedEntity.__name__:
            return True

        if annotation.type == nlp.Entity.__name__:
            if (isinstance(annotation.value.type, nlp.EntityType)
                    and annotation.value.type not in [nlp.EntityType.SPEAKER, nlp.EntityType.HEARER]):
                return True
            if (isinstance(annotation.value.type, str)
                    and annotation.value.type not in [nlp.EntityType.SPEAKER.name.lower(), nlp.EntityType.HEARER.name.lower()]):
                return True

        return False


class TextPerspectiveDetector(MentionDetector):
    # Nothing to filter
    pass


class ImagePerspectiveDetector(MentionDetector):
    def __init__(self, threshold: float):
        self._threshold = threshold

    def is_above_threshold(self, annotation: Annotation) -> bool:
        if annotation.value.confidence < self._threshold or annotation.value.type == 'NEUTRAL':
            return False

        if isinstance(annotation.value.type, Enum) and annotation.value.type.name == 'NEUTRAL':
            return False

        return True

    def filter_mentions(self, mentions: List[Mention], scenario_id: str) -> List[Mention]:
        return [mention for mention in mentions
                if any(self.is_above_threshold(annotation) for annotation in mention.annotations)]


class NewFaceMentionDetector(MentionDetector):
    def __init__(self):
        self._scenario_id = None
        self._faces = set()

    def filter_mentions(self, mentions: List[Mention], scenario_id: str) -> List[Mention]:
        if scenario_id != self._scenario_id:
            self._scenario_id = scenario_id
            self._faces = set()

        new_face_mentions = [mention for mention in mentions
                             if (mention.annotations
                                 and mention.annotations[0].value is not None
                                 and mention.annotations[0].value not in self._faces)]

        self._faces = self._faces | {mention.annotations[0].value for mention in new_face_mentions}

        return new_face_mentions


class ObjectMentionDetector(MentionDetector):
    def __init__(self):
        self._previous = set()

    def filter_mentions(self, mentions: List[Mention], scenario_id: str) -> List[Mention]:
        observed = [mention for mention in mentions
                if (mention.annotations
                    and mention.annotations[0].value is not None
                    and mention.annotations[0].value.label.lower() in _ACCEPTED_OBJECTS
                    and mention.annotations[0].value.label.lower() not in self._previous)]

        self._previous = set(mention.annotations[0].value.label.lower() for mention in mentions)

        return observed


class DefaultMentionExtractor(MentionExtractor):
    def __init__(self, text_detector: MentionDetector,
                 text_perspective_detector: TextPerspectiveDetector,
                 image_perspective_detector: ImagePerspectiveDetector,
                 face_detector: MentionDetector,
                 object_detector: MentionDetector):
        self._text_detector = text_detector
        self._text_perspective_detector = text_perspective_detector
        self._image_perspective_detector = image_perspective_detector
        self._face_detector = face_detector
        self._object_detector = object_detector

    def extract_text_mentions(self, mentions: List[Mention], scenario_id: str) -> List[TextMention]:
        return [self.create_text_mention(mention, scenario_id)
                for mention in self._text_detector.filter_mentions(mentions, scenario_id)]

    def extract_text_perspective(self, mentions: List[Mention], scenario_id: str) -> List[TextPerspective]:
        return [self.create_text_perspective(mention, scenario_id)
                for mention in self._text_perspective_detector.filter_mentions(mentions, scenario_id)]

    def extract_object_mentions(self, mentions: List[Mention], scenario_id: str) -> List[ImageMention]:
        return [self.create_object_mention(mention, scenario_id)
                for mention in self._object_detector.filter_mentions(mentions, scenario_id)]

    def extract_face_mentions(self, mentions: List[Mention], scenario_id: str) -> List[ImageMention]:
        return [self.create_face_mention(mention, scenario_id)
                for mention in self._face_detector.filter_mentions(mentions, scenario_id)]

    def extract_face_perspective(self, mentions: List[Mention], scenario_id: str) -> List[ImagePerspective]:
        return [self.create_image_perspective(mention, scenario_id)
                for mention in self._image_perspective_detector.filter_mentions(mentions, scenario_id)]

    def create_face_mention(self, mention: Mention, scenario_id: str):
        image_id = mention.id
        image_path = mention.id

        mention_id = mention.id
        bounds = mention.segment[0].bounds
        face_id = mention.annotations[0].value
        confidence = 1.0

        return ImageMention(image_id, mention_id, _IMAGE_SOURCE, image_path, bounds,
                            Entity(face_id, ["face"], face_id, None), {},
                            confidence, scenario_id, timestamp_now())

    def create_object_mention(self, mention: Mention, scenario_id: str):
        image_id = mention.id
        image_path = mention.id

        mention_id = mention.id
        bounds = mention.segment[0].bounds
        # TODO multiple?
        object_label = mention.annotations[0].value.label
        confidence = mention.annotations[0].value.confidence if hasattr(mention.annotations[0].value, 'confidence') else 1.0

        return ImageMention(image_id, mention_id, _IMAGE_SOURCE, image_path, bounds,
                            Entity(object_label, [object_label], None, None), {},
                            confidence, scenario_id, timestamp_now())

    def create_text_mention(self, mention: Mention, scenario_id: str):
        author = self._get_speaker()

        utterance = ""

        segment = mention.segment[0]
        signal_id = segment.container_id
        entity_text = mention.annotations[0].value.text
        entity_type = mention.annotations[0].value.label
        confidence = 1.0

        return TextMention(scenario_id, signal_id, author, utterance, f"{segment.start} - {segment.stop}",
                           Entity(entity_text, [entity_type], None, None), {},
                           confidence, scenario_id, timestamp_now())

    def create_text_perspective(self, mention, scenario_id):
        author = self._get_speaker()

        utterance = ""

        segment = mention.segment[0]
        signal_id = segment.container_id
        perspective = f"{mention.annotations[0].value.type}:{mention.annotations[0].value.value}"
        confidence = mention.annotations[0].value.confidence

        return TextPerspective(scenario_id, signal_id, author, utterance, f"{segment.start} - {segment.stop}",
                               author, Perspective(perspective, confidence), scenario_id, timestamp_now())

    def create_image_perspective(self, mention, scenario_id):
        # TODO
        image_id = mention.id
        image_path = mention.id

        speaker = self._get_speaker()

        mention_id = mention.id
        bounds = mention.segment[0].bounds

        annotations = filter(self._image_perspective_detector.is_above_threshold, mention.annotations)
        annotations = sorted(annotations, key = lambda ann: ann.value.confidence, reverse=True)
        # annotations should not empty, as filtered already by the _image_perspective_detector
        primary_emotion = next(iter(annotations)).value

        perspective = f"{primary_emotion.type}:{primary_emotion.value}"
        confidence = primary_emotion.confidence

        return ImagePerspective(image_id, mention_id, _IMAGE_SOURCE, image_path, bounds,
                                speaker, Perspective(perspective, confidence), scenario_id, timestamp_now())

    def _get_speaker(self):
        return Entity(ConversationalAgent.SPEAKER.name, [class_type(ConversationalAgent)], None, None)

