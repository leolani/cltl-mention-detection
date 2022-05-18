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

from cltl.nlp.api import NLP, Token, NamedEntity

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
                                         resource_manager=self._resource_manager, processor=self._process)
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
        token_annotations = [Annotation(Token.__name__, token, NLP.__name__, timestamp_now())
                             for token in doc.tokens]
        token_segments = [text_signal.ruler.get_offset(*token.segment) for token in doc.tokens]

        entity_annotations = [Annotation(NamedEntity.__name__, entity, NLP.__name__, timestamp_now())
                             for entity in doc.entities]
        entity_segments = [text_signal.ruler.get_offset(*entity.segment) for entity in doc.entities]

        mentions = [Mention(str(uuid.uuid4()), [segment], [annotation])
                    for annotation, segment
                    in chain(zip(token_annotations, token_segments), zip(entity_annotations, entity_segments))]

        if mentions:
            self._event_bus.publish(self._output_topic, Event.for_payload(AnnotationEvent.create(mentions)))