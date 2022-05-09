import spacy
import time
from datetime import datetime
import os

import emissor as em
from emissor.persistence import ScenarioStorage
from emissor.representation.annotation import AnnotationType, Token, NER
from emissor.representation.container import Index
from emissor.representation.scenario import Modality, ImageSignal, TextSignal, Mention, Annotation, Scenario
import cltl.mention_detection.get_mentions as mentions

def relative_path(modality: Modality, file_name: str) -> str:
    rel_path = os.path.join(modality.name.lower())

    return os.path.join(rel_path, file_name) if file_name else rel_path


def absolute_path(scenario_storage: ScenarioStorage, scenario_id: str, modality: Modality, file_name: str = None) -> str:
    abs_path = os.path.abspath(os.path.join(scenario_storage.base_path, scenario_id, modality.name.lower()))

    return os.path.join(abs_path, file_name) if file_name else abs_path


def create_text_signal(scenario: Scenario, utterance: str, timestamp: int = None):
    timestamp = int(time.time() * 1e3) if timestamp is None else timestamp
    return TextSignal.for_scenario(scenario.id, timestamp, timestamp, [], utterance, [])

def create_scenario(scenarioPath: str, scenarioid: str):
    storage = ScenarioStorage(scenarioPath)

    os.makedirs(absolute_path(storage, scenarioid, Modality.IMAGE))
    # Not yet needed
    # os.makedirs(absolut_path(storage, scenarioid, Modality.TEXT))

    print(f"Directories for {scenarioid} created in {storage.base_path}")



if __name__ == "__main__":
    '''
    test annotation of a TextSignal
    '''

    ### Create the scenario folder, the json files and a scenarioStorage and scenario in memory
    nlp = spacy.load("en_core_web_sm")
    text = "I am from Amsterdam and I like cats."
    scenario_path = "data"
    scenario_id = datetime.today().strftime("%Y-%m-%d-%H:%M:%S")

    scenarioStorage = create_scenario(scenario_path, scenario_id)
    scenario_ctrl = scenarioStorage.create_scenario(scenario_id, int(time.time() * 1e3), None, "Leolani")
    textSignal = create_text_signal(scenario_ctrl, text, None )
    mentions.add_np_annotation_with_spacy (textSignal, nlp,  "Piek", "Leolani")
    mentions.add_ner_annotation_with_spacy(textSignal, nlp)
    scenario_ctrl.append_signal(textSignal)

    scenario_ctrl.scenario.ruler.end = int(time.time() * 1e3)
    scenarioStorage.save_scenario(scenario_ctrl)

