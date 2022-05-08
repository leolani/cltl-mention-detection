import abc
import dataclasses
from typing import List, Optional, Iterable, Tuple

import numpy as np


@dataclasses.dataclass
class Mention:
    """
    Information about a Face.

    Includes a vector representation of the face and optional meta information.
    """
    # TODO switch to np.typing.ArrayLike
    type: Optional[int]


class MentionDetector(abc.ABC):
    """
    Detect faces in an image.
    """
    @TODO #replace bouns by offsets
    def detect(self, image: np.ndarray) -> Tuple[Iterable[Mention], Iterable[Bounds]]:
        """
        Detect objects in an image.

        Parameters
        ----------
        image : np.ndarray
            The binary image.

        Returns
        -------
        Iterable[Object]
            The faces detected in the image.
        Iterable[Bounds]
            The positions of the detected objects in the image.
        """
        raise NotImplementedError()

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass
