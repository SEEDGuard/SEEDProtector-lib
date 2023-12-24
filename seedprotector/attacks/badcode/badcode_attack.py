#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   attack.py
@Time    :   2023/12/24
@Author  :   Bowen Xu
@License :   MIT License
@Desc    :   SEEDProtector
'''

import seedprotector.attacks.attack as attack

class badcode_attack(attack.Attack):
    """
    This class implements the badcode attack.
    """

    def __init__(self, training_data, test_data, data, model, eval) -> None:
        pass