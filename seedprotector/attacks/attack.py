#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   attack.py
@Time    :   2023/12/24
@Author  :   Bowen Xu
@License :   MIT License
@Desc    :   SEEDProtector
'''


"""
This module implements Backdoor Attacks to poison data used in ML code models.
"""

import logging
from typing import Callable, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)

class Attack(object):

    def __init__(self, perturbation: Union[Callable, List[Callable]]) -> None:
        """
        Initialize a backdoor poisoning attack.

        :param perturbation: A single perturbation function or list of perturbation functions that modify input.
        """
        super().__init__()
        self.perturbation = perturbation
        self._check_params()

    def poison(  # pylint: disable=W0221
        self, x: np.ndarray, y: Optional[np.ndarray] = None, broadcast=False, **kwargs
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calls perturbation function on input x and returns the perturbed input and poison labels for the data.

        :param x: An array with the points that initialize attack points.
        :param y: The target labels for the attack.
        :param broadcast: whether or not to broadcast single target label
        :return: An tuple holding the `(poisoning_examples, poisoning_labels)`.
        """
        if y is None:  # pragma: no cover
            raise ValueError("Target labels `y` need to be provided for a targeted attack.")

        if broadcast:
            y_attack = np.broadcast_to(y, (x.shape[0], y.shape[0]))
        else:
            y_attack = np.copy(y)

        num_poison = len(x)
        if num_poison == 0:  # pragma: no cover
            raise ValueError("Must input at least one poison point.")
        poisoned = np.copy(x)

        if callable(self.perturbation):
            return self.perturbation(poisoned), y_attack

        for perturb in self.perturbation:
            poisoned = perturb(poisoned)

        return poisoned, y_attack

    def _check_params(self) -> None:
        if not (callable(self.perturbation) or all((callable(perturb) for perturb in self.perturbation))):
            raise ValueError("Perturbation must be a function or a list of functions.")

