# -*- encoding: utf-8 -*-
'''
@File    :   config.py
@Time    :   2019/11/19 20:15:17
@Author  :   Cao Shuai
@Version :   1.0
@Contact :   caoshuai@stu.scu.edu.cn
@License :   (C)Copyright 2018-2019, MILAB_SCU
@Desc    :   None
'''

from collections import namedtuple
from typing import NamedTuple

class Config(NamedTuple):
    bert_model = './Rostlab/prot_t5_xl_uniref50'
