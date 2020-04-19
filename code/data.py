# -*- coding: utf-8 -*-
# @Time    : 2020/4/19 下午2:50
# @Author  : RegiusQuant <315135833@qq.com>
# @Project : CCKS2020-Entity-Linking
# @File    : data.py
# @Desc    : 数据预处理

import json
from collections import defaultdict
from pathlib import Path

import pandas as pd


class EntityLinkingDataProcessor:

    def __init__(self):
        pass

    @staticmethod
    def process_kb(raw_path, pickle_path):
        entity_to_kbids = defaultdict(set)
        kbid_to_entities = dict()
        kbid_to_text = dict()
        kbid_to_types = dict()

        idx_to_type = []
        type_to_idx = dict()

        with open(raw_path / 'kb.json', 'r') as f:
            for i, line in enumerate(f):
                temp = json.loads(line)

                kbid = temp['subject_id']
                entities = set(temp['alias'])
                entities.add(temp['subject'])
                for entity in entities:
                    entity_to_kbids[entity].add(kbid)
                kbid_to_entities[kbid] = entities

                data_list = []
                for x in temp['data']:
                    data_list.append(':'.join([x['predicate'], x['object']]))
                kbid_to_text[kbid] = ' '.join(data_list)

                type_list = temp['type'].split('|')
                kbid_to_types[kbid] = type_list
                for t in type_list:
                    if t not in type_to_idx:
                        type_to_idx[t] = len(idx_to_type)
                        idx_to_type.append(t)

        pd.to_pickle(entity_to_kbids, pickle_path / 'entity_to_kbids.pkl')
        pd.to_pickle(kbid_to_entities, pickle_path / 'kbid_to_entities.pkl')
        pd.to_pickle(kbid_to_text, pickle_path / 'kbid_to_text.pkl')
        pd.to_pickle(kbid_to_types, pickle_path / 'kbid_to_types.pkl')

        pd.to_pickle(idx_to_type, pickle_path / 'idx_to_type.pkl')
        pd.to_pickle(type_to_idx, pickle_path / 'type_to_idx.pkl')


if __name__ == '__main__':
    raw_data = Path('../data/ccks2020_el_data_v1')
    pickle_data = Path('../data/pickle')
    EntityLinkingDataProcessor.process_kb(raw_data, pickle_data)
