import logging
from typing import Callable

from cltl.backend.source.client_source import ClientTextSource
from cltl.backend.spi.text import TextSource
from cltl.combot.infra.config import ConfigurationManager
from cltl.combot.infra.event import Event, EventBus
from cltl.combot.infra.resource import ResourceManager
from cltl.combot.infra.topic_worker import TopicWorker
from cltl.combot.event.emissor import ImageSignalEvent

from cltl.mention_detection.api import MentionDetector
from cltl_service.mention_detection.schema import MentionDetectionEvent

logger = logging.getLogger(__name__)


class MentionDetectionService:
    """
    Service used to integrate the component into applications.
    """
    @classmethod
    def from_config(cls, mention_detector: MentionDetector, event_bus: EventBus, resource_manager: ResourceManager,
                    config_manager: ConfigurationManager):
        config = config_manager.get_config("cltl.mention_detection.events")
        #TODO This is probably not needed for processing a text signal that is posted by the ASR component
        def text_loader(url) -> TextSource:
            return ClientTextSource.from_config(config_manager, url)


        return cls(config.get("text_topic"), config.get("mention_topic"), mention_detector, text_loader,
                   event_bus, resource_manager)

    def __init__(self, input_topic: str, output_topic: str, mention_detector: MentionDetector,
                 text_loader: Callable[[str], TextSource],
                 event_bus: EventBus, resource_manager: ResourceManager):
        self._mention_detector = mention_detector
        self._text_loader = text_loader

        self._event_bus = event_bus
        self._resource_manager = resource_manager

        self._input_topic = input_topic
        self._output_topic = output_topic

        self._topic_worker = None
        self._app = None

    def start(self, timeout=30):
        self._mention_detector.__enter__()
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
        self._mention_detector.__exit__(None, None, None)

    def _process(self, event: Event[ImageSignalEvent]):
        image_location = event.payload.signal.files[0]

        with self._image_loader(image_location) as source:
            text = source.capture()
        mentions, bounds = self._mention_detector.detect(text.text)

        mention_event = MentionDetectionEvent.create(event.payload.signal, mentions, bounds)
        self._event_bus.publish(self._output_topic, Event.for_payload(mention_event))