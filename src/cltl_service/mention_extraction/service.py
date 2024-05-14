import logging
from dataclasses import asdict
from typing import List

import cltl_service.face_emotion_extraction.schema
from cltl.combot.event.emissor import AnnotationEvent, ScenarioEvent, ScenarioStarted, ScenarioStopped
from cltl.combot.infra.config import ConfigurationManager
from cltl.combot.infra.event import Event, EventBus
from cltl.combot.infra.resource import ResourceManager
from cltl.combot.infra.topic_worker import TopicWorker
from cltl_service.emotion_extraction.schema import EmotionRecognitionEvent
from cltl_service.object_recognition.schema import ObjectRecognitionEvent
from cltl_service.vector_id.schema import VectorIdentityEvent
from emissor.representation.scenario import class_type

from cltl.mention_extraction.api import MentionExtractor
from cltl.mention_extraction import object_label_translation

logger = logging.getLogger(__name__)


_OBJECT_RESPONSE = {        
        "tas": ["wat een mooie tas","waar heb je die tas gekocht"],
        "bril": ["mooie bril, van Hans?"],
        "stropdas": ["wat een bijzondere stropdas"],
        "jas": ["een leuk jas heb je aan", "Die jas staat u geweldig", "We hebben een mooie garderobe als u uw jas kwijt wilt"],
        "rugzak": ["wat een fantastische rugzak","Passen er veel spullen in die mooie rugzak?", "Waar heeft u die mooie rugzak gekocht?"],
        "paraplu": ["Jeetje, regent het buiten?", "Heeft u die paraplu nodig gehad vandaag?", "Wat een mooie paraplu heeft u"],
        "wijnglas": ["Helaas mag uw consumptie niet mee de zaal in", "Houd uw wijnglas alstublieft niet in mijn buurt", "Houd u over het algemeen meer van witte of rode wijn?"],
        "koffer": ["Bent u net op reis geweest met die mooie koffer?", "Wat een mooie koffer", "Wat een stijlvolle koffer"],
        "boek": ["Wat voor leuks leest u?", "U zult uw boek niet nodig hebben in deze leuke voorstelling", "Bent u een lezer? bekijk vooral dan ook ons programmaboekje"],
        "telefoon": ["Vergeet uw telefoon niet uit te zetten tijdens de voorstelling", "Mooie telefoon heeft u, die heeft vast veel gekost", "Mag ik ook eens bellen met uw telefoon?"],
        "knuffelbeer": ["Wat een lieve knuffel heb jij"],
        "bril": ["Wat voor sterkte heeft u?", "Zo een mooie bril heb ik nog nooit gezien"],
        "tas": ["Is die tas wel van u?", "U kunt uw tas ook achterlaten bij de garderobe"], 
        "tanden": ["Wat een mooie glimlach heeft u!", "Wat een mooie mond met tanden heeft u. U zou niet misstaan in een Colgate reclame"],
        "cup": ["Helaas mag uw consumptie niet mee de zaal in", "Houd uw kop alstublieft niet in mijn buurt", "Houd u over het algemeen meer van koffie of thee?"],
        "handbag": ["wat een mooie tas","waar heb je die tas gekocht"],
        "glasses": ["mooie bril, van Hans?"],
        "tie": ["wat een bijzondere stropdas"],
        "coat": ["een leuk jas heb je aan", "Die jas staat u geweldig", "We hebben een mooie garderobe als u uw jas kwijt wilt"],
        "backpack": ["wat een fantastische rugzak","Passen er veel spullen in die mooie rugzak?", "Waar heeft u die mooie rugzak gekocht?"],
        "umbrella": ["Jeetje, regent het buiten?", "Heeft u die paraplu nodig gehad vandaag?", "Wat een mooie paraplu heeft u."],
        "wine glass": ["Helaas mag uw consumptie niet mee de zaal in", "Houd uw wijnglas alstublieft niet in mijn buurt", "Houd u over het algemeen meer van witte of rode wijn?"],
        "suitcase": ["Bent u net op reis geweest met die mooie koffer?", "Wat een mooie koffer", "Wat een stijlvolle koffer"],
        "book": ["Wat voor leuks leest u?", "U zult uw boek niet nodig hebben in deze leuke voorstelling", "Bent u een lezer? bekijk vooral dan ook ons programmaboekje"],
        "cell phone": ["Vergeet uw telefoon niet uit te zetten tijdens de voorstelling", "Mooie telefoon heeft u, die heeft vast veel gekost", "Mag ik ook eens bellen met uw telefoon?", "Je mag wel een selfie nemen met je telefoon. Smile.", "Wil je misschien een selfie nemen? Vind ik geen probleem hoor."],
        "teddy bear": ["Wat een lieve knuffel heb jij"],
        "bird": ["Zie ik daar nu een vogel?", "Wat doet een vogel nu hier? Zie jij die ook?", "Kijk daar, een vogel."],
        "laptop": ["Waarom heb jij een laptop bij je? Stop nu eens met werken", "Wat een mooie laptop. Programmeer jij zelf ook?", "Is die laptop van jou?"],
        "glasses": ["Wat voor sterkte heeft u?", "Zo een mooie bril heb ik nog nooit gezien"],
        "tas": ["Is die tas wel van u?", "U kunt uw tas ook achterlaten bij de garderobe"], 
        "tanden": ["Wat een mooie glimlach heeft u!", "Wat een mooie mond met tanden heeft u. U zou niet misstaan in een Colgate reclame"]
    }
    

class MentionExtractionService:
    """
    Service used to integrate the component into applications.
    """
    @classmethod
    def from_config(cls, mention_extractor: MentionExtractor,
                    event_bus: EventBus,
                    resource_manager: ResourceManager,
                    config_manager: ConfigurationManager):
        langconfig = config_manager.get_config("cltl.language")
        config = config_manager.get_config("cltl.mention_extraction.events")
        object_rate = int(config.get("object_rate"))
        input_topics = config.get("topics_in", multi=True)
        output_topic = config.get("topic_out")

        scenario_topic = config.get("topic_scenario")
        intentions = config.get("intentions", multi=True)
        intention_topic = config.get("topic_intention")
        language = langconfig.get("language")

        return cls(mention_extractor, scenario_topic, input_topics, output_topic, intentions, intention_topic,
                   event_bus, resource_manager, language, object_rate)

    def __init__(self, mention_extractor: MentionExtractor,
                 scenario_topic: str, input_topics: List[str], output_topic: str, intentions: List[str], intention_topic: str,
                 event_bus: EventBus, resource_manager: ResourceManager, language: str = "en", object_rate: int = 5):
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

        self._language = language

    def start(self):
        self._topic_worker = TopicWorker(self._input_topics, self._event_bus, provides=[self._output_topic],
                                         buffer_size=64,
                                         resource_manager=self._resource_manager, processor=self._process,
                                         name=self.__class__.__name__)
        self._topic_worker.start().wait()

    def stop(self):
        if not self._topic_worker:
            pass

        self._topic_worker.stop()
        self._topic_worker.await_stop()
        self._topic_worker = None

    def _process(self, event: Event):
        if event.metadata.topic == self._intention_topic:
            self._active_intentions = {intention.label for intention in event.payload.intentions}
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
        elif event.payload.type == class_type(EmotionRecognitionEvent):
            mention_factory = self._mention_extractor.extract_text_perspective
        elif event.payload.type == class_type(cltl_service.face_emotion_extraction.schema.EmotionRecognitionEvent):
            mention_factory = self._mention_extractor.extract_face_perspective
        else:
            raise ValueError("Unsupported event type %s", event.payload.type)

        mentions = mention_factory(event.payload.mentions, self._scenario_id) if mention_factory else None

        if mentions:
            logger.debug("Detected %s mentions from %s", len(mentions), mention_factory.__name__)
            self._event_bus.publish(self._output_topic, Event.for_payload([asdict(mention) for mention in mentions]))

        # TODO Temporary code to create a better conversation
        if mentions and event.payload.type == ObjectRecognitionEvent.__name__:
            from collections import Counter
            from random import choice
            from cltl.combot.infra.time_util import timestamp_now
            from cltl.combot.event.emissor import TextSignalEvent
            from emissor.representation.scenario import TextSignal

            logger.debug("Detected %s mentions from %s", len(mentions), mention_factory.__name__)
            object_counts = Counter(mention.item.label for mention in mentions)
            GREET = ""
            FOLLOW_UP =""
            if self._language=="nl":
                I_SEE = ["Ik zie", "Zie ik dat goed", "Kijk daar heb je","Wat zie ik nu!"]
                I_GREET_ONE = ["Kijk een mens. Hoi", "Hallo jij daar", "Hallo hallo. Goed je te zien","Leuk je te zien mens", "Welkom en fijn dat je er bent", "Wat goed dat je gekomen bent", "Nou dat vind ik pas leuk je te zien", "En wie hebben we hier dan", "Kom binnen, kom binnen", "Wat fijn dat je er bent"]
                I_GREET_TWO = ["He mensen. Ik groet jullie", "Hallo, hoi, goed jullie te zien","Aah dat wordt gezellig met jullie", "Leuk jullie te zien", "Mensen, kom binnen", "Fijn, jullie zijn er ook", "Welkom en fijn dat jullie er zijn", "Wat goed dat jullie gekomen zijn", "Nou dat vind ik pas leuk jullie te zien", "En wie hebben we hier dan", "Kom binnen, kom binnen", "Wat fijn dat jullie er zijn"]
                dutch_counts = []
                for object, cnt in object_counts.items():
                    forms = object_label_translation.to_dutch(object)
                    dutch_counts.append({'singular': forms[0], 'plural':forms[1], 'cnt': cnt})
                    if object == "person":
                        if cnt==1:
                            GREET = ". "+ choice(I_GREET_ONE)
                        else:
                            GREET = ". "+ choice(I_GREET_TWO)
                    elif object in _OBJECT_RESPONSE:
                        FOLLOW_UP = ". " + choice(_OBJECT_RESPONSE[object])
                object_counts = dutch_counts
                counts = ', '.join([f"{result['cnt'] if result['cnt'] > 1 else 'een'} {result['plural'] if result['cnt']> 1 else result['singular']}"
                                for result in object_counts])
                counts = (counts[::-1].replace(' ,', ' ne ', 1))[::-1]
            else:
                I_SEE = ["I see", "I can see", "I think I see", "I observe",]
                counts = ', '.join([f"{count if count > 1 else 'a'} {label}{'s' if count> 1 else ''}"
                                    for label, count in object_counts.items()])
                counts = (counts[::-1].replace(' ,', ' dna ', 1))[::-1]
                if "person" in object_counts:
                    cnt = object_counts["person"]
                    if cnt==1:
                        GREET = ". Nice to see you human!"
                    else:
                        GREET = ". Nice to see you folks!"

            utterance =  f"{choice(I_SEE)} {counts}{GREET}{FOLLOW_UP}"

            signal = TextSignal.for_scenario(self._scenario_id, timestamp_now(), timestamp_now(), None, utterance)
            self._event_bus.publish("cltl.topic.text_out", Event.for_payload(TextSignalEvent.for_agent(signal)))
