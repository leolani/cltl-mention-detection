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

        input_topics = config.get("topics_in", multi=True)
        output_topic = config.get("topic_out")

        scenario_topic = config.get("topic_scenario")
        intentions = config.get("intentions", multi=True)
        intention_topic = config.get("topic_intention")

        return cls(mention_extractor, scenario_topic, input_topics, output_topic, intentions, intention_topic,
                   event_bus, resource_manager)

    def __init__(self, mention_extractor: MentionExtractor,
                 scenario_topic: str, input_topics: List[str], output_topic: str, intentions: List[str], intention_topic: str,
                 event_bus: EventBus, resource_manager: ResourceManager, object_rate: int = 5):
        self._event_bus = event_bus
        self._resource_manager = resource_manager

        self._mention_extractor = mention_extractor

        self._input_topics = input_topics + [scenario_topic, intention_topic]
        self._output_topic = output_topic

        self._intention_topic = intention_topic if intention_topic else None
        self._intentions = set(intentions) if intentions else {}
        self._active_intentions = {}

        self._topic_worker = None
        self._app = None

        self._scenario_id = None

        self._object_event_cnt = 0
        self._object_rate = object_rate

    def start(self):
        self._topic_worker = TopicWorker(self._input_topics, self._event_bus, provides=[self._output_topic],
                                         buffer_size=64,
                                         resource_manager=self._resource_manager, processor=self._process)
        self._topic_worker.start().wait()

    def stop(self):
        if not self._topic_worker:
            pass

        self._topic_worker.stop()
        self._topic_worker.await_stop()
        self._topic_worker = None

    def _process(self, event: Event):
        if event.metadata.topic == self._intention_topic:
            self._active_intentions = set(event.payload.intentions)
            logger.info("Set active intentions to %s", self._active_intentions)
            return

        if event.payload.type == ScenarioStarted.__name__:
            self._scenario_id = event.payload.scenario.id
            return
        if event.payload.type == ScenarioStopped.__name__:
            self._scenario_id = None
            return
        if event.payload.type == ScenarioEvent.__name__:
            return

        if not self._scenario_id:
            logger.debug("No active scenario, skipping %s", event.payload.type)
            return

        if self._intentions and not (self._active_intentions & self._intentions):
            logger.debug("Skipped event outside intention %s, active: %s (%s)",
                         self._intentions, self._active_intentions, event)
            return

        mention_factory = None
        if event.payload.type == AnnotationEvent.__name__:
            mention_factory = self._mention_extractor.extract_text_mentions
        elif event.payload.type == VectorIdentityEvent.__name__:
            mention_factory = self._mention_extractor.extract_face_mentions
        elif event.payload.type == ObjectRecognitionEvent.__name__:
            if self._object_event_cnt % self._object_rate == 0:
                mention_factory = self._mention_extractor.extract_object_mentions
            self._object_event_cnt += 1
        else:
            raise ValueError("Unsupported event type %s", event.payload.type)

        mentions = mention_factory(event.payload.mentions, self._scenario_id) if mention_factory else None

        if mentions:
            logger.debug("Detected %s mentions", len(mentions))
            self._event_bus.publish(self._output_topic, Event.for_payload([asdict(mention) for mention in mentions]))