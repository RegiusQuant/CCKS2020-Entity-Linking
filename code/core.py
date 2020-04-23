# -*- coding: utf-8 -*-
# @Time    : 2020/4/23 下午12:35
# @Author  : RegiusQuant <315135833@qq.com>
# @Project : CCKS2020-Entity-Linking
# @File    : core.py
# @Desc    : 实体链指模块导入和全局变量

import os
import json
import random
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from tqdm import tqdm
from transformers import (
    DataProcessor,
    InputExample,
    BertConfig,
    BertTokenizer,
    BertForSequenceClassification,
    glue_convert_examples_to_features,
)

# 项目数据路径
DATA_PATH = '../data/'

# CCKS2020实体链指竞赛原始路径
RAW_PATH = DATA_PATH + 'ccks2020_el_data_v1/'

# 预处理后导出的pickle文件路径
PICKLE_PATH = DATA_PATH + 'pickle/'
if not os.path.exists(PICKLE_PATH):
    os.mkdir(PICKLE_PATH)

# 训练、验证、推断所需的tsv文件路径
TSV_PATH = DATA_PATH + 'tsv/'
if not os.path.exists(TSV_PATH):
    os.mkdir(TSV_PATH)

PICKLE_DATA = {
    # 实体名称对应的KBID列表
    'ENTITY_TO_KBIDS': None,
    # KBID对应的实体名称列表
    'KBID_TO_ENTITIES': None,
    # KBID对应的属性文本
    'KBID_TO_TEXT': None,
    # KBID对应的实体类型列表（注意：一个实体可能对应'|'分割的多个类型）
    'KBID_TO_TYPES': None,
    # KBID对应的属性列表
    'KBID_TO_PREDICATES': None,

    # 索引类型映射列表
    'IDX_TO_TYPE': None,
    # 类型索引映射字典
    'TYPE_TO_IDX': None,
}

for k in PICKLE_DATA:
    filename = k + '.pkl'
    if os.path.exists(PICKLE_PATH + filename):
        PICKLE_DATA[k] = pd.read_pickle(PICKLE_PATH + filename)
    else:
        print(f'File {filename} not Exist!')


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
