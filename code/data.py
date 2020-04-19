# -*- coding: utf-8 -*-
# @Time    : 2020/4/19 下午2:50
# @Author  : RegiusQuant <315135833@qq.com>
# @Project : CCKS2020-Entity-Linking
# @File    : data.py
# @Desc    : 数据预处理

import json
import random
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
                    data_list.append(':'.join([x['predicate'].strip(), x['object'].strip()]))
                kbid_to_text[kbid] = ' '.join(data_list)
                for c in ['\r', '\t', '\n']:
                    kbid_to_text[kbid] = kbid_to_text[kbid].replace(c, '')

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
        print('Process KB Finish!')

    @staticmethod
    def process_link(file_path, pickle_path, save_path, max_negs=2):
        entity_to_kbids = pd.read_pickle(pickle_path / 'entity_to_kbids.pkl')
        kbid_to_text = pd.read_pickle(pickle_path / 'kbid_to_text.pkl')

        link_dict = defaultdict(list)
        with open(file_path, 'r') as f:
            for i, line in enumerate(f):
                temp = json.loads(line)

                for data in temp['mention_data']:
                    if not data['kb_id'].isdigit():
                        continue

                    entity = data['mention']
                    kbids = list(entity_to_kbids[entity])
                    random.shuffle(kbids)

                    num_negs = 0
                    for kbid in kbids:
                        if kbid == data['kb_id']:
                            link_dict['entity'].append(entity)
                            link_dict['offset'].append(data['offset'])
                            link_dict['rawtext'].append(temp['text'])
                            link_dict['kbtext'].append(kbid_to_text[kbid])
                            link_dict['predict'].append(1)
                        else:
                            if num_negs >= max_negs:
                                continue
                            link_dict['entity'].append(entity)
                            link_dict['offset'].append(data['offset'])
                            link_dict['rawtext'].append(temp['text'])
                            link_dict['kbtext'].append(kbid_to_text[kbid])
                            link_dict['predict'].append(0)
                            num_negs += 1

        train_link = pd.DataFrame(link_dict)
        train_link.to_csv(save_path, index=False, sep='\t')
        print('Process Link Data Finish!')

    @staticmethod
    def process_type(file_path, pickle_path, save_path):
        kbid_to_types = pd.read_pickle(pickle_path / 'kbid_to_types.pkl')

        type_dict = defaultdict(list)
        with open(file_path, 'r') as f:
            for i, line in enumerate(f):
                temp = json.loads(line)

                for data in temp['mention_data']:
                    entity = data['mention']

                    if data['kb_id'].isdigit():
                        entity_type = kbid_to_types[data['kb_id']]
                    else:
                        entity_type = data['kb_id'].split('|')
                        for x in range(len(entity_type)):
                            entity_type[x] = entity_type[x][4:]
                    for e in entity_type:
                        type_dict['entity'].append(entity)
                        type_dict['offset'].append(data['offset'])
                        type_dict['rawtext'].append(temp['text'])
                        type_dict['type'].append(e)

        train_type = pd.DataFrame(type_dict)
        train_type.to_csv(save_path, index=False, sep='\t')
        print('Process Type Data Finish!')


def main():
    random.seed(2020)

    raw_path = Path('../data/ccks2020_el_data_v1')
    pickle_path = Path('../data/pickle')
    EntityLinkingDataProcessor.process_kb(raw_path, pickle_path)

    file_path = raw_path / 'train.json'
    save_path = Path('../data/csv/train_link.csv')
    EntityLinkingDataProcessor.process_link(file_path, pickle_path, save_path, max_negs=2)
    save_path = Path('../data/csv/train_type.csv')
    EntityLinkingDataProcessor.process_type(file_path, pickle_path, save_path)

    file_path = raw_path / 'dev.json'
    save_path = Path('../data/csv/valid_link.csv')
    EntityLinkingDataProcessor.process_link(file_path, pickle_path, save_path, max_negs=2)
    save_path = Path('../data/csv/valid_type.csv')
    EntityLinkingDataProcessor.process_type(file_path, pickle_path, save_path)


if __name__ == '__main__':
    main()
