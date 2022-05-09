#### Example of an annotation function that adds annotations to a Signal
#### It adds NERC annotations to the TextSignal and returns a list of entities detected
from typing import Text
import requests
import uuid
import jsonpickle

import time
from emissor.representation.annotation import AnnotationType, Token, NER
from emissor.representation.container import Index
from emissor.representation.scenario import TextSignal, Mention, Annotation
from emissor.representation.entity import Emotion



def annotate_tokens(signal: TextSignal, token_text_list, segments, annotationType:str, processor_name:str):

        """
         given a TextSignal and a list of spaCy tokens and a corresponding list of segments, annotate the segments in the signal with a label (annotationType)
        """

        current_time = int(time.time() * 1e3)        

        annotations = [Annotation(annotationType.lower(), token_text, processor_name, current_time)
                       for token_text in token_text_list]

        signal.mentions.extend([Mention(str(uuid.uuid4()), [segment], [annotation])
                                for segment, annotation in zip(segments, annotations)])


def add_ner_annotation_with_spacy(signal: TextSignal, nlp):
    processor_name = "spaCy"
    utterance = ''.join(signal.seq)

    doc = nlp(utterance)

    segments, tokens = zip(*[(Index(signal.id, token.idx, token.idx + len(token)), Token.for_string(token.text))
                            for token in doc])
    
    annotate_tokens(signal, tokens, segments,AnnotationType.TOKEN.name, processor_name)


    entity_labels = [NER.for_string(ent.label_) for ent in doc.ents]

    entity_token_list = [ent.text for ent in doc.ents]
    
    segments = [token.ruler for token in tokens if token.value in entity_token_list]
    annotate_tokens(signal, entity_token_list, segments, AnnotationType.NER.name, processor_name)

    if entity_token_list:
        segments, tokens = zip(*[(Index(signal.id, token.idx, token.idx + len(token)), Token.for_string(token.text))
                                 for token in entity_token_list])
        annotate_tokens(signal, tokens, segments, AnnotationType.TOKEN.name, processor_name)


    # print(entity_list)
    return entity_token_list, entity_labels



def add_np_annotation_with_spacy(signal: TextSignal, nlp,  SPEAKER: str, HEARER: str):

    rels={'nsubj', 'nsubjpass', 'dobj', 'prep', 'pcomp', 'acomp'}
    """
    extract predicates with:
    -subject
    -object
    
    :param spacy.tokens.doc.Doc doc: spaCy object after processing text
    
    :rtype: list 
    :return: list of tuples (predicate, subject, object)
    """
    processor_name = "spaCy"
    utterance = ''.join(signal.seq)

    doc = nlp(utterance)
    
    predicates = {}
    subjects_and_objects_labels = []
    subject_and_object_tokens = []
    
    speaker_mentions =[]
    hearer_mentions =[]
    speaker_tokens = []
    hearer_tokens = []
  
    for token in doc:
        if token.dep_ in rels:
            
            head = token.head
            head_id = head.i
            
            if head_id not in predicates:
                predicates[head_id] = dict()
            #print(token.pos_)
            if token.pos_=="PRON" :
                if (token.text.lower()=='i'):
                    speaker_mentions.append(SPEAKER)  
                    speaker_tokens.append(token)
                elif (token.text.lower()=='you'):
                    hearer_mentions.append(HEARER)
                    hearer_tokens.append(token)
            elif token.pos_=="NOUN" or token.pos_=="VERB" or token.pos_=="PROPN":
                #TODO this should be filtered for labels from object recognition
                subjects_and_objects_labels.append(token.lemma_)
                subject_and_object_tokens.append(token)
            
            predicates[head_id][token.dep_] = token.lemma_

    #TODO make sure the correct annotations are made as well 
    if subject_and_object_tokens:
        segments, tokens = zip(*[(Index(signal.id, token.idx, token.idx + len(token)), Token.for_string(token.text))
                                for token in subject_and_object_tokens])
        annotate_tokens(signal, tokens, segments,AnnotationType.TOKEN.name, processor_name)

    if speaker_tokens:
        segments, tokens = zip(*[(Index(signal.id, token.idx, token.idx + len(token)), Token.for_string(token.text))
                            for token in speaker_tokens])
        annotate_tokens(signal, tokens, segments,AnnotationType.LINK.name, processor_name)

    if hearer_tokens:
        segments, tokens = zip(*[(Index(signal.id, token.idx, token.idx + len(token)), Token.for_string(token.text))
                            for token in hearer_tokens])
        annotate_tokens(signal, tokens, segments,AnnotationType.LINK.name, processor_name)

    return subjects_and_objects_labels


def recognize_emotion(utterance: str, url_erc: str = "http://127.0.0.1:10006"):
    """Recognize the speaker emotion of a given utterance.
    
    Args
    ----
    utterance:
    url_erc: the url of the emoberta api server.

    Returns
    -------
    ?
    """
    data = {"text": utterance}

    data = jsonpickle.encode(data)
    response = requests.post(url_erc, json=data)
    response = jsonpickle.decode(response.text)

    return response

# def create_emotion_mention(text_signal: TextSignal, source: str, current_time: int, 
#                            emotion: str):
#     emotion_annotation = Annotation(AnnotationType.EMOTION.emotion, 
#     )

# @emissor_dataclass(namespace="http://cltl.nl/leolani/n2mu")
# class EmotionPerson(Emotion):
#     emotion_prob: float