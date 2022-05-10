import logging

import numpy as np
from cltl.combot.infra.config import ConfigurationManager
from cltl.combot.infra.event import Event, EventBus
from cltl.combot.infra.resource import ResourceManager
from cltl.combot.infra.time_util import timestamp_now
from cltl.combot.infra.topic_worker import TopicWorker
from cltl.combot.event.emissor import TextSignalEvent
from emissor.representation.scenario import TextSignal
from flask import Flask, Response
from flask.json import JSONEncoder

from cltl.mention_detection.api import DemoProcessor

logger = logging.getLogger(__name__)


# TODO move to common util in combot
class NumpyJSONEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()

        return super().default(obj)


class TemplateService:
    """
    Service used to integrate the component into applications.
    """
    @classmethod
    def from_config(cls, processor: DemoProcessor, event_bus: EventBus, resource_manager: ResourceManager,
                    config_manager: ConfigurationManager):
        config = config_manager.get_config("cltl.mention-detection")

        return cls(config.get("topic_in"), config.get("topic_out"), processor, event_bus, resource_manager)

    def __init__(self, input_topic: str, output_topic: str, processor: DemoProcessor,
                 event_bus: EventBus, resource_manager: ResourceManager):
        self._processor = processor

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

    @property
    def app(self):
        """
        Flask endpoint for REST interface.
        """
        if self._app:
            return self._app

        self._app = Flask("audio_storage")
        self._app.json_encoder = NumpyJSONEncoder

        @self._app.route(f"/mention-detection/<paramter>", methods=['GET'])
        def store_audio(parameter: str):
            return Response(status=200)

        @self._app.after_request
        def set_cache_control(response):
            response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
            response.headers['Pragma'] = 'no-cache'
            response.headers['Expires'] = '0'

            return response

        return self._app

    def _process(self, event: Event[TextSignalEvent]):
        response = self._processor.respond(event.payload.signal.text)

        if response:
            eliza_event = self._create_payload(response)
            self._event_bus.publish(self._output_topic, Event.for_payload(eliza_event))

    def _create_payload(self, response):
        signal = TextSignal.for_scenario(None, timestamp_now(), timestamp_now(), None, response)

        return TextSignalEvent.create(signal)
