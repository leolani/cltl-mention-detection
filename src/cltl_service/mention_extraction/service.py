import logging
from dataclasses import asdict
from typing import List

from cltl.combot.event.emissor import AnnotationEvent, ScenarioEvent, ScenarioStarted, ScenarioStopped
from cltl.combot.infra.config import ConfigurationManager
from cltl.combot.infra.event import Event, EventBus
from cltl.combot.infra.resource import ResourceManager
from cltl.combot.infra.topic_worker import TopicWorker
from cltl_service.object_recognition.schema import ObjectRecognitionEvent
from cltl_service.vector_id.schema import VectorIdentityEvent

from cltl.mention_extraction.api import MentionExtractor

logger = logging.getLogger(__name__)


class MentionExtractionService:
    """
    Service used to integrate the component into applications.
    """
    @classmethod
    def from_config(cls, mention_extractor: MentionExtractor,
                    event_bus: EventBus,
                    resource_manager: ResourceManager,
                    config_manager: ConfigurationManager):
        config = config_manager.get_config("cltl.mention_extraction.events")

        scenario_topic = config.get("scenario_topic")
        input_topics = config.get("topics_in", multi=True)
        output_topic = config.get("topic_out")

        return cls(mention_extractor, scenario_topic, input_topics, output_topic, event_bus, resource_manager)

    def __init__(self, mention_extractor: MentionExtractor,
                 scenario_topic: str, input_topics: List[str], output_topic: str, event_bus: EventBus,
                 resource_manager: ResourceManager):
        self._event_bus = event_bus
        self._resource_manager = resource_manager

        self._mention_extractor = mention_extractor

        self._input_topics = input_topics + [scenario_topic]
        self._output_topic = output_topic

        self._topic_worker = None
        self._app = None

        self._scenario_id = None

    def start(self):
        self._topic_worker = TopicWorker(self._input_topics, self._event_bus, provides=[self._output_topic],
                                         resource_manager=self._resource_manager, processor=self._process)
        self._topic_worker.start().wait()

    def stop(self):
        if not self._topic_worker:
            pass

        self._topic_worker.stop()
        self._topic_worker.await_stop()
        self._topic_worker = None

    def _process(self, event: Event):
        if event.payload.type == ScenarioStarted.__name__:
            self._scenario_id = event.payload.scenario.id
            return
        if event.payload.type == ScenarioStopped.__name__:
            self._scenario_id = None
            return

        if not self._scenario_id:
            logger.debug("No active scenario, skipping %s", event.payload.type)
            return

        if event.payload.type == AnnotationEvent.__name__:
            mention_factory = self._mention_extractor.extract_text_mentions
        elif event.payload.type == VectorIdentityEvent.__name__:
            mention_factory = self._mention_extractor.extract_face_mentions
        elif event.payload.type == ObjectRecognitionEvent.__name__:
            mention_factory = self._mention_extractor.extract_object_mentions
        else:
            raise ValueError("Unsupported event type %s", event.payload.type)

        mentions = mention_factory(event.payload.mentions, self._scenario_id)

        if mentions:
            self._event_bus.publish(self._output_topic, Event.for_payload([asdict(mention) for mention in mentions]))