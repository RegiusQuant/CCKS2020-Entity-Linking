# -*- coding: utf-8 -*-
# @Time    : 2020/4/19 下午2:50
# @Author  : RegiusQuant <315135833@qq.com>
# @Project : CCKS2020-Entity-Linking
# @File    : data.py
# @Desc    : 数据预处理

from core import *


class PicklePreprocessor:
    """生成全局变量Pickle文件的预处理器"""

    def __init__(self):
        # 实体名称对应的KBID列表
        self.entity_to_kbids = defaultdict(set)
        # KBID对应的实体名称列表
        self.kbid_to_entities = dict()
        # KBID对应的属性文本
        self.kbid_to_text = dict()
        # KBID对应的实体类型列表
        self.kbid_to_types = dict()
        # KBID对应的属性列表
        self.kbid_to_predicates = dict()

        # 索引类型映射列表
        self.idx_to_type = list()
        # 类型索引映射字典
        self.type_to_idx = dict()

    def run(self, shuffle_text=True):
        with open(RAW_PATH + 'kb.json', 'r') as f:
            for line in tqdm(f):
                line = json.loads(line)

                kbid = line['subject_id']
                # 将实体名与别名合并
                entities = set(line['alias'])
                entities.add(line['subject'])
                for entity in entities:
                    self.entity_to_kbids[entity].add(kbid)
                self.kbid_to_entities[kbid] = entities

                text_list, predicate_list = [], []
                for x in line['data']:
                    # 简单拼接predicate与object，这部分可以考虑别的方法尝试
                    text_list.append(':'.join(
                        [x['predicate'].strip(), x['object'].strip()]))
                    predicate_list.append(x['predicate'].strip())
                if shuffle_text:  # 对属性文本随机打乱顺序
                    random.shuffle(text_list)
                self.kbid_to_predicates[kbid] = predicate_list
                self.kbid_to_text[kbid] = ' '.join(text_list)
                # 删除文本中的特殊字符
                for c in ['\r', '\t', '\n']:
                    self.kbid_to_text[kbid] = self.kbid_to_text[kbid].replace(
                        c, '')

                type_list = line['type'].split('|')
                self.kbid_to_types[kbid] = type_list
                for t in type_list:
                    if t not in self.type_to_idx:
                        self.type_to_idx[t] = len(self.idx_to_type)
                        self.idx_to_type.append(t)

        # 保存pickle文件
        pd.to_pickle(self.entity_to_kbids, PICKLE_PATH + 'ENTITY_TO_KBIDS.pkl')
        pd.to_pickle(self.kbid_to_entities,
                     PICKLE_PATH + 'KBID_TO_ENTITIES.pkl')
        pd.to_pickle(self.kbid_to_text, PICKLE_PATH + 'KBID_TO_TEXT.pkl')
        pd.to_pickle(self.kbid_to_types, PICKLE_PATH + 'KBID_TO_TYPES.pkl')
        pd.to_pickle(self.kbid_to_predicates,
                     PICKLE_PATH + 'KBID_TO_PREDICATES.pkl')
        pd.to_pickle(self.idx_to_type, PICKLE_PATH + 'IDX_TO_TYPE.pkl')
        pd.to_pickle(self.type_to_idx, PICKLE_PATH + 'TYPE_TO_IDX.pkl')


class DataFramePreprocessor:
    """生成模型训练、验证、推断所需的tsv文件"""

    def __init__(self):
        pass

    def process_link_data(self, input_path, output_path, max_negs=-1):
        entity_to_kbids = PICKLE_DATA['ENTITY_TO_KBIDS']
        kbid_to_text = PICKLE_DATA['KBID_TO_TEXT']
        kbid_to_predicates = PICKLE_DATA['KBID_TO_PREDICATES']
        link_dict = defaultdict(list)

        with open(input_path, 'r') as f:
            for line in tqdm(f):
                line = json.loads(line)

                for data in line['mention_data']:
                    # 对测试集特殊处理
                    if 'kb_id' not in data:
                        data['kb_id'] = '0'

                    # KB中不存在的实体不进行链接
                    if not data['kb_id'].isdigit():
                        continue

                    entity = data['mention']
                    kbids = list(entity_to_kbids[entity])
                    random.shuffle(kbids)

                    num_negs = 0
                    for kbid in kbids:
                        if num_negs >= max_negs > 0 and kbid != data['kb_id']:
                            continue

                        link_dict['text_id'].append(line['text_id'])
                        link_dict['entity'].append(entity)
                        link_dict['offset'].append(data['offset'])
                        link_dict['short_text'].append(line['text'])
                        link_dict['kb_id'].append(kbid)
                        link_dict['kb_text'].append(kbid_to_text[kbid])
                        link_dict['kb_predicate_num'].append(
                            len(kbid_to_predicates[kbid]))
                        if kbid != data['kb_id']:
                            link_dict['predict'].append(0)
                            num_negs += 1
                        else:
                            link_dict['predict'].append(1)

        link_data = pd.DataFrame(link_dict)
        link_data.to_csv(output_path, index=False, sep='\t')

    def process_type_data(self, input_path, output_path):
        kbid_to_types = PICKLE_DATA['KBID_TO_TYPES']
        type_dict = defaultdict(list)

        with open(input_path, 'r') as f:
            for line in tqdm(f):
                line = json.loads(line)

                for data in line['mention_data']:
                    entity = data['mention']

                    # 测试集特殊处理
                    if 'kb_id' not in data:
                        entity_type = ['Other']
                    elif data['kb_id'].isdigit():
                        entity_type = kbid_to_types[data['kb_id']]
                    else:
                        entity_type = data['kb_id'].split('|')
                        for x in range(len(entity_type)):
                            entity_type[x] = entity_type[x][4:]
                    for e in entity_type:
                        type_dict['text_id'].append(line['text_id'])
                        type_dict['entity'].append(entity)
                        type_dict['offset'].append(data['offset'])
                        type_dict['short_text'].append(line['text'])
                        type_dict['type'].append(e)

        type_data = pd.DataFrame(type_dict)
        type_data.to_csv(output_path, index=False, sep='\t')

    def run(self):
        self.process_link_data(
            input_path=RAW_PATH + 'train.json',
            output_path=TSV_PATH + 'EL_TRAIN.tsv',
            max_negs=2,
        )
        print('Process EL_TRAIN Finish.')
        self.process_link_data(
            input_path=RAW_PATH + 'dev.json',
            output_path=TSV_PATH + 'EL_VALID.tsv',
            max_negs=-1,
        )
        print('Process EL_VALID Finish.')
        self.process_link_data(
            input_path=RAW_PATH + 'test.json',
            output_path=TSV_PATH + 'EL_TEST.tsv',
            max_negs=-1,
        )
        print('Process EL_TEST Finish.')

        self.process_type_data(
            input_path=RAW_PATH + 'train.json',
            output_path=TSV_PATH + 'ET_TRAIN.tsv',
        )
        print('Process ET_TRAIN Finish.')
        self.process_type_data(
            input_path=RAW_PATH + 'dev.json',
            output_path=TSV_PATH + 'ET_VALID.tsv',
        )
        print('Process ET_VALID Finish.')
        self.process_type_data(
            input_path=RAW_PATH + 'test.json',
            output_path=TSV_PATH + 'ET_TEST.tsv',
        )
        print('Process ET_TEST Finish.')
