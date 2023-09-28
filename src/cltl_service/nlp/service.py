import logging
import uuid
from itertools import chain

from cltl.combot.event.emissor import TextSignalEvent, AnnotationEvent
from cltl.combot.infra.config import ConfigurationManager
from cltl.combot.infra.event import Event, EventBus
from cltl.combot.infra.resource import ResourceManager
from cltl.combot.infra.time_util import timestamp_now
from cltl.combot.infra.topic_worker import TopicWorker
from emissor.representation.scenario import Annotation, Mention

from cltl.nlp.api import NLP, Token, NamedEntity, Entity

logger = logging.getLogger(__name__)


class NLPService:
    """
    Service used to integrate the component into applications.
    """
    @classmethod
    def from_config(cls, nlp: NLP, event_bus: EventBus, resource_manager: ResourceManager,
                    config_manager: ConfigurationManager):
        config = config_manager.get_config("cltl.nlp.events")

        return cls(config.get("topic_in"), config.get("topic_out"), nlp, event_bus, resource_manager)

    def __init__(self, input_topic: str, output_topic: str, nlp: NLP,
                 event_bus: EventBus, resource_manager: ResourceManager):
        self._nlp = nlp

        self._event_bus = event_bus
        self._resource_manager = resource_manager

        self._input_topic = input_topic
        self._output_topic = output_topic

        self._topic_worker = None
        self._app = None

    def start(self, timeout=30):
        self._topic_worker = TopicWorker([self._input_topic], self._event_bus, provides=[self._output_topic],
                                         resource_manager=self._resource_manager, processor=self._process,
                                         name=self.__class__.__name__)
        self._topic_worker.start().wait()

    def stop(self):
        if not self._topic_worker:
            pass

        self._topic_worker.stop()
        self._topic_worker.await_stop()
        self._topic_worker = None

    def _process(self, event: Event[TextSignalEvent]):
        text_signal = event.payload.signal
        doc = self._nlp.analyze(text_signal.text)

        # TODO recap emissor Annotation classes -> NER, Token, etc.
        token_segments, token_annotations = self._convert_to_segment_annotation(text_signal, Token.__name__, doc.tokens)
        ner_segments, ner_annotations = self._convert_to_segment_annotation(text_signal, NamedEntity.__name__, doc.named_entities)
        entity_segments, entity_annotations = self._convert_to_segment_annotation(text_signal, Entity.__name__, doc.entities)

        mentions = [Mention(str(uuid.uuid4()), [segment], [annotation])
                    for segment, annotation
                    in chain(zip(token_segments, token_annotations),
                             zip(ner_segments, ner_annotations),
                             zip(entity_segments, entity_annotations))]

        if mentions:
            self._event_bus.publish(self._output_topic, Event.for_payload(AnnotationEvent.create(mentions)))

    def _convert_to_segment_annotation(self, text_signal, type, collection):
        annotations = [Annotation(type, element, NLP.__name__, timestamp_now()) for element in collection]
        segments = [text_signal.ruler.get_offset(*element.segment) for element in collection]

        return segments, annotations