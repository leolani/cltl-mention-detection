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


    ents = [NER.for_string(ent.label_) for ent in doc.ents]    
    entity_list = [ent.text for ent in doc.ents]
    
    segments = [token.ruler for token in tokens if token.value in entity_list]
    annotate_tokens(signal, entity_list, segments, AnnotationType.NER.name, processor_name)
    
    

#    current_time = int(time.time() * 1e3)
#    annotations = [Annotation(AnnotationType.TOKEN.name.lower(), token, processor_name, current_time)
#                   for token in tokens]
#    ner_annotations = [Annotation(AnnotationType.NER.name.lower(), ent, processor_name, current_time)
#                       for ent in ents]
#
#    signal.mentions.extend([Mention(str(uuid.uuid4()), [offset], [annotation])
#                            for offset, annotation in zip(offsets, annotations)])
#    signal.mentions.extend([Mention(str(uuid.uuid4()), [segment], [annotation])
#                            for segment, annotation in zip(segments, ner_annotations)])
    # print(entity_list)
    return entity_list



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
    offsets, tokens = zip(*[(Index(signal.id, token.idx, token.idx + len(token)), Token.for_string(token.text))
                            for token in doc])

    
    predicates = {}
    subjects_and_objects = []
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
                subjects_and_objects.append(token.lemma_)
                subject_and_object_tokens.append(token)

            
            predicates[head_id][token.dep_] = token.lemma_
   #### Change this to create triples 

    #TODO make sure the correct annotations are made as well 
    if subject_and_object_tokens:
        segments, tokens = zip(*[(Index(signal.id, token.idx, token.idx + len(token)), Token.for_string(token.text))
                                for token in subject_and_object_tokens])
        annotate_tokens(signal, subjects_and_objects, segments,AnnotationType.TOKEN.name, processor_name)

    if speaker_tokens:
        segments, tokens = zip(*[(Index(signal.id, token.idx, token.idx + len(token)), Token.for_string(token.text))
                            for token in speaker_tokens])
        annotate_tokens(signal, speaker_mentions, segments,AnnotationType.LINK.name, processor_name)

    if hearer_tokens:
        segments, tokens = zip(*[(Index(signal.id, token.idx, token.idx + len(token)), Token.for_string(token.text))
                            for token in hearer_tokens])
        annotate_tokens(signal, hearer_mentions, segments,AnnotationType.LINK.name, processor_name)

    return subjects_and_objects


def get_subj_amod_triples_with_spacy(signal: TextSignal, nlp,  SPEAKER: str, HEARER: str):

    """
    extract predicates with:
    -subject
    -object
    
    :param spacy.tokens.doc.Doc doc: spaCy object after processing text
    
    :rtype: list 
    :return: list of tuples (predicate, subject, object)
    """
    rels={'nsubj', 'nsubjpass', 'auxpass', 'acomp'}
    processor_name = "spaCy"
    utterance = ''.join(signal.seq)

    doc = nlp(utterance)

    triples = []

    predicates={}
    acomp = []
    
    subject_tokens = []
    subject_mentions =[] 
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
            if token.dep_=='nsubj' or token.dep_=='nsubjpass':
                if (token.text.lower()=='i'):
                    predicates[head_id]['head'] = SPEAKER
                    speaker_tokens.append(token)
                    speaker_mentions.append(SPEAKER)
                elif (token.text.lower()=='you'):
                    predicates[head_id]['head'] = HEARER
                    hearer_tokens.append(token)
                    hearer_mentions.append(HEARER)
                elif token.pos_=="PROPN":
                    predicates[head_id]['head'] = token.lemma_
                    subject_tokens.append(token)
                    subject_mentions.append(SPEAKER)
            if token.dep_=='acomp' or token.dep=='auxpass':
                    predicates[head_id]['tail'] = token.lemma_
    for pred_token, pred_info in predicates.items():
        triple = (doc[pred_token].lemma_, 
                   pred_info.get('head', None),
                   pred_info.get('tail', None)
                  )
        if triple[1] and triple[2]:
            triples.append(triple)

    if triples:
 
        #TODO make sure the correct annotations are made as well 
        if subject_tokens:
            segments, tokens = zip(*[(Index(signal.id, token.idx, token.idx + len(token)), Token.for_string(token.text))
                                for token in subject_tokens])
            annotate_tokens(signal, subject_mentions, segments,AnnotationType.NER.name, processor_name)

        if speaker_tokens:
            segments, tokens = zip(*[(Index(signal.id, token.idx, token.idx + len(token)), Token.for_string(token.text))
                                for token in speaker_tokens])
            annotate_tokens(signal, speaker_mentions, segments,AnnotationType.LINK.name, processor_name)

        if hearer_tokens:
            segments, tokens = zip(*[(Index(signal.id, token.idx, token.idx + len(token)), Token.for_string(token.text))
                                for token in hearer_tokens])
            annotate_tokens(signal, hearer_mentions, segments,AnnotationType.LINK.name, processor_name)


   # print('Triples subj - aux - amod', triples)
    return triples


def get_subj_obj_triples_with_spacy(signal: TextSignal, nlp,  SPEAKER: str, HEARER: str):

    """
    extract predicates with:
    -subject
    -object
    
    :param spacy.tokens.doc.Doc doc: spaCy object after processing text
    
    :rtype: list 
    :return: list of tuples (predicate, subject, object)
    """
    rels={'nsubj', 'dobj', 'xcomp'}
    processor_name = "spaCy"
    utterance = ''.join(signal.seq)

    doc = nlp(utterance)

    triples = []
 
    predicates={}
    acomp = []
    
    subject_tokens = []
    subject_mentions =[] 
    object_tokens = []
    object_mentions =[] 

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
            if token.dep_=='nsubj':
                if (token.text.lower()=='i'):
                    predicates[head_id]['head'] = SPEAKER
                    speaker_tokens.append(token)
                    speaker_mentions.append(SPEAKER)
                elif (token.text.lower()=='you'):
                    predicates[head_id]['head'] = HEARER
                    hearer_tokens.append(token)
                    hearer_mentions.append(HEARER)
                elif token.pos_=="PROPN" or token.pos_=='NOUN':
                    predicates[head_id]['head'] = token.lemma_
                    subject_tokens.append(token)
                    subject_mentions.append(SPEAKER)
            if token.dep_=='dobj' or token.dep=='xcomp':
                if (token.text.lower()=='i'):
                    predicates[head_id]['tail'] = SPEAKER
                    speaker_tokens.append(token)
                    speaker_mentions.append(SPEAKER)
                elif (token.text.lower()=='you'):
                    predicates[head_id]['tail'] = HEARER
                    hearer_tokens.append(token)
                    hearer_mentions.append(HEARER)
                elif token.pos_=="PROPN" or token.pos_=='NOUN' or token.pos_=='ADJ':
                    predicates[head_id]['tail'] = token.lemma_
                    subject_tokens.append(token)
                    subject_mentions.append(SPEAKER)
    for pred_token, pred_info in predicates.items():
        triple = (doc[pred_token].lemma_, 
                   pred_info.get('head', None),
                   pred_info.get('tail', None)
                  )
        if triple[1] and triple[2]:
            if not triple in triples:
                triples.append(triple)

    if triples:
 
        #TODO make sure the correct annotations are made as well 
        if subject_tokens:
            segments, tokens = zip(*[(Index(signal.id, token.idx, token.idx + len(token)), Token.for_string(token.text))
                                for token in subject_tokens])
            annotate_tokens(signal, subject_mentions, segments,AnnotationType.NER.name, processor_name)

        if speaker_tokens:
            segments, tokens = zip(*[(Index(signal.id, token.idx, token.idx + len(token)), Token.for_string(token.text))
                                for token in speaker_tokens])
            annotate_tokens(signal, speaker_mentions, segments,AnnotationType.LINK.name, processor_name)

        if hearer_tokens:
            segments, tokens = zip(*[(Index(signal.id, token.idx, token.idx + len(token)), Token.for_string(token.text))
                                for token in hearer_tokens])
            annotate_tokens(signal, hearer_mentions, segments,AnnotationType.LINK.name, processor_name)


   # print('Triples subj - pred - obj', triples)
    return triples


def get_subj_attr_triples_with_spacy(signal: TextSignal, nlp,  SPEAKER: str, HEARER: str):

    """
    extract predicates with:
    -subject
    -object
    
    :param spacy.tokens.doc.Doc doc: spaCy object after processing text
    
    :rtype: list 
    :return: list of tuples (predicate, subject, object)
    """
    rels={'nsubj', 'intj','appos' 'attr'}
    processor_name = "spaCy"
    utterance = ''.join(signal.seq)

    doc = nlp(utterance)

    triples = []
 
    predicates={}
    acomp = []
    
    subject_tokens = []
    subject_mentions =[] 
    object_tokens = []
    object_mentions =[] 

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
            if token.dep_=='nsubj' or token.dep=='intj':
                if (token.text.lower()=='i'):
                    predicates[head_id]['head'] = SPEAKER
                    speaker_tokens.append(token)
                    speaker_mentions.append(SPEAKER)
                elif (token.text.lower()=='you'):
                    predicates[head_id]['head'] = HEARER
                    hearer_tokens.append(token)
                    hearer_mentions.append(HEARER)
                elif token.pos_=="PROPN" or token.pos_=='NOUN':
                    predicates[head_id]['head'] = token.lemma_
                    subject_tokens.append(token)
                    subject_mentions.append(SPEAKER)
            if token.dep_=='attr' or token.dep=='appos':
                if (token.text.lower()=='i'):
                    predicates[head_id]['tail'] = SPEAKER
                    speaker_tokens.append(token)
                    speaker_mentions.append(SPEAKER)
                elif (token.text.lower()=='you'):
                    predicates[head_id]['tail'] = HEARER
                    hearer_tokens.append(token)
                    hearer_mentions.append(HEARER)
                elif token.pos_=="PROPN" or token.pos_=='NOUN' or token.pos_=='ADJ':
                    predicates[head_id]['tail'] = token.lemma_
                    subject_tokens.append(token)
                    subject_mentions.append(token.lemma_)
    for pred_token, pred_info in predicates.items():
        triple = (doc[pred_token].lemma_, 
                   pred_info.get('head', None),
                   pred_info.get('tail', None)
                  )
        if triple[1] and triple[2]:
            if not triple in triples:
                triples.append(triple)

 
    if triples:
 
        #TODO make sure the correct annotations are made as well 
        if subject_tokens:
            segments, tokens = zip(*[(Index(signal.id, token.idx, token.idx + len(token)), Token.for_string(token.text))
                                for token in subject_tokens])
            annotate_tokens(signal, subject_mentions, segments,AnnotationType.NER.name, processor_name)

        if speaker_tokens:
            segments, tokens = zip(*[(Index(signal.id, token.idx, token.idx + len(token)), Token.for_string(token.text))
                                for token in speaker_tokens])
            annotate_tokens(signal, speaker_mentions, segments,AnnotationType.LINK.name, processor_name)

        if hearer_tokens:
            segments, tokens = zip(*[(Index(signal.id, token.idx, token.idx + len(token)), Token.for_string(token.text))
                                for token in hearer_tokens])
            annotate_tokens(signal, hearer_mentions, segments,AnnotationType.LINK.name, processor_name)



   # print('Triples subj - pred - attr', triples)
    return triples



#@TODO
def get_subj_prep_pobj_triples_with_spacy(signal: TextSignal, nlp,  SPEAKER: str, HEARER: str):

    """
    extract predicates with:
    -subject
    -object
    
    :param spacy.tokens.doc.Doc doc: spaCy object after processing text
    
    :rtype: list 
    :return: list of tuples (predicate, subject, object)
    """
    rels={'nsubj', 'nsubjpass', 'prep', 'pobj'}
    processor_name = "spaCy"
    utterance = ''.join(signal.seq)

    doc = nlp(utterance)

    triples = []

    predicates={}
    acomp = []
    
    subject_tokens = []
    subject_mentions =[] 
    object_tokens = []
    object_mentions =[] 

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
            if token.dep_=='nsubj':
                if (token.text.lower()=='i'):
                    predicates[head_id]['head'] = SPEAKER
                    speaker_tokens.append(token)
                    speaker_mentions.append(SPEAKER)
                elif (token.text.lower()=='you'):
                    predicates[head_id]['head'] = HEARER
                    hearer_tokens.append(token)
                    hearer_mentions.append(HEARER)
                elif token.pos_=="PROPN" or token.pos_=='NOUN':
                    predicates[head_id]['head'] = token.lemma_
                    subject_tokens.append(token)
                    subject_mentions.append(SPEAKER)
            elif token.dep_=='prep':
                predicates[head_id]['prep'] = token.lemma_
                for token_dep in doc:
                    if token_dep.dep_=='pobj' and token_dep.head.i == token.i:
                        #print(token_dep.head.i, token.i)
                        #### We now need to get the token that has a "pobj" dependency to this prep
                        if (token_dep.text.lower()=='i'):
                            predicates[head_id]['tail'] = SPEAKER
                            speaker_tokens.append(token_dep)
                            speaker_mentions.append(SPEAKER)
                        elif (token_dep.text.lower()=='you'):
                            predicates[head_id]['tail'] = HEARER
                            hearer_tokens.append(token_dep)
                            hearer_mentions.append(HEARER)
                        elif token_dep.pos_=="PROPN" or token_dep.pos_=='NOUN' or token_dep.pos_=='ADJ':
                            predicates[head_id]['tail'] = token_dep.lemma_
                            subject_tokens.append(token_dep)
                            subject_mentions.append(token_dep.lemma_)
        #print(predicates)
        for pred_token, pred_info in predicates.items():
            property_string = doc[pred_token].lemma_+"_"+pred_info.get('prep', str(None))
            triple = (property_string, 
                       pred_info.get('head', None),
                       pred_info.get('tail', None)
                      )
            if triple[1] and triple[2]:
                if not triple in triples:
                    triples.append(triple)
        if triples:
 
            #TODO make sure the correct annotations are made as well 
            if subject_tokens:
                segments, tokens = zip(*[(Index(signal.id, token.idx, token.idx + len(token)), Token.for_string(token.text))
                                    for token in subject_tokens])
                annotate_tokens(signal, subject_mentions, segments,AnnotationType.NER.name, processor_name)

            if speaker_tokens:
                segments, tokens = zip(*[(Index(signal.id, token.idx, token.idx + len(token)), Token.for_string(token.text))
                                    for token in speaker_tokens])
                annotate_tokens(signal, speaker_mentions, segments,AnnotationType.LINK.name, processor_name)

            if hearer_tokens:
                segments, tokens = zip(*[(Index(signal.id, token.idx, token.idx + len(token)), Token.for_string(token.text))
                                    for token in hearer_tokens])
                annotate_tokens(signal, hearer_mentions, segments,AnnotationType.LINK.name, processor_name)


   
  #  print('Triples subj - pred - prep-obj', triples)
    return triples


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