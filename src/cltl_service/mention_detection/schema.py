import uuid
from cltl.combot.event.emissor import AnnotationEvent
from cltl.combot.infra.time_util import timestamp_now
from dataclasses import dataclass
from emissor.representation.scenario import Mention, TextSignal, Annotation
from typing import Iterable

from cltl.mention_detection.api import Mention


@dataclass
class MentionDetectionEvent(AnnotationEvent[Annotation[Mention]]):
    @classmethod
    @TODO # replace bounds with offsets
    def create(cls, text_signal: TextSignal, mentions: Iterable[Mention], bounds: Iterable[Bounds]):
        if objects:
            mentions = [MentionDetectionEvent.to_mention(text_signal, mention, bound)
                        for mention, bound in zip(mentions, bounds)]
        else:
            mentions = [MentionDetectionEvent.to_mention(text_signal)]

        return cls(cls.__name__, mentions)

    @staticmethod
    @TODO # replace bounds with offsets
    def to_mention(text_signal: TextSignal, object: Mention = None, bounds: Bounds = None):
        segment = text_signal.ruler
        if bounds:
            clipped = Bounds(segment.bounds[0], segment.bounds[2],
                             segment.bounds[1], segment.bounds[3]).intersection(bounds)
            segment = segment.get_area_bounding_box(clipped.x0, clipped.y0, clipped.x1, clipped.y1)

        annotation = Annotation(Object.__name__, object, __name__, timestamp_now())

        return Mention(str(uuid.uuid4()), [segment], [annotation])