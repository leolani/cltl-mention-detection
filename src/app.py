import logging.config
import time

from cltl.mention_extraction.api import MentionExtractor
from cltl.mention_extraction.default_extractor import DefaultMentionExtractor
from cltl.nlp.api import NLP
from cltl.nlp.spacy_nlp import SpacyNLP
from cltl_service.mention_extraction.service import MentionExtractionService

logging.config.fileConfig('config/logging.config')

import json
from types import SimpleNamespace

from cltl.combot.infra.config.k8config import K8LocalConfigurationContainer
from cltl.combot.infra.di_container import singleton
from cltl.combot.infra.event.kombu import KombuEventBus
from cltl.combot.infra.event.memory import SynchronousEventBus
from cltl.combot.infra.di_container import
from cltl.combot.infra.resource.threaded import ThreadedResourceContainer
from cltl_service.nlp.service import NLPService
from kombu.serialization import register

logger = logging.getLogger(__name__)

K8LocalConfigurationContainer.load_configuration()

class InfraContainer(K8LocalConfigurationContainer, ThreadedResourceContainer):
    def start(self):
        pass

    def stop(self):
        pass

class NLPContainer(InfraContainer):
    @property
    @singleton
    def nlp(self) -> NLP:
        config = self.config_manager.get_config("cltl.nlp.spacy")

        return SpacyNLP(config.get('model'))

    @property
    @singleton
    def nlp_service(self) -> NLPService:
        return NLPService.from_config(self.nlp, self.event_bus, self.resource_manager, self.config_manager)

    def start(self):
        logger.info("Start NLP service")
        super().start()
        self.nlp_service.start()

    def stop(self):
        logger.info("Stop NLP service")
        self.nlp_service.stop()
        super().stop()


class MentionExtractionContainer(InfraContainer):
    @property
    @singleton
    def mention_extractor(self) -> MentionExtractor:
        return DefaultMentionExtractor()

    @property
    @singleton
    def mention_extraction_service(self) -> MentionExtractionService:
        return MentionExtractionService.from_config(self.mention_extractor,
                                                    self.event_bus, self.resource_manager, self.config_manager)

    def start(self):
        logger.info("Start Mention Extraction Service")
        super().start()
        self.mention_extraction_service.start()

    def stop(self):
        logger.info("Stop Mention Extraction Service")
        self.mention_extraction_service.stop()
        super().stop()

class ApplicationContainer(NLPContainer, MentionExtractionContainer):
    logger.info("Initialized ApplicationContainer")

    def start(self):
        super().start()

    def stop(self):
        super().start()

    @property
    @singleton
    def event_bus(self):
        config = self.config_manager.get_config("cltl.mention-detection.events")
        if config.get_boolean("local"):
            return SynchronousEventBus()
        else:
            register('cltl-json',
                     lambda x: json.dumps(x, default=vars),
                     lambda x: json.loads(x, object_hook=lambda d: SimpleNamespace(**d)),
                     content_type='application/json',
                     content_encoding='utf-8')
            return KombuEventBus('cltl-json', self.config_manager)


class Application(ApplicationContainer):
    def run(self):
        self.start()
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            self.stop()


if __name__ == '__main__':
    Application().run()
