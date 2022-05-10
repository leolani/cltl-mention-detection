import logging

import numpy as np

from cltl.mention_detection.api import DemoProcessor

logger = logging.getLogger(__name__)


class DummyDemoProcessor(DemoProcessor):
    """
    Dummy implementation of the component.
    """
    def __init__(self, phrase: str):
        self.__phrase = phrase

    def respond(self, statement: str) -> str:
        logger.debug("Responding to statement: %s, numpy version", statement, np.__version__)

        return f"Mhm, {self.__phrase}." if statement else "Hi!"
