# -*- coding: utf-8 -*-
# @Time    : 2020/4/19 下午4:05
# @Author  : RegiusQuant <315135833@qq.com>
# @Project : CCKS2020-Entity-Linking
# @File    : model_random_search.py
# @Desc    : 简单的随机生成结果


import json
import random
from pathlib import Path

import pandas as pd

import eval


class RandomSearchModel:

    def __init__(self):
        pass

    @staticmethod
    def generate_result(file_path, save_path, pickle_path):
        entity_to_kbids = pd.read_pickle(pickle_path / 'entity_to_kbids.pkl')
        idx_to_type = pd.read_pickle(pickle_path / 'idx_to_type.pkl')

        result = []
        with open(file_path, 'r') as f:
            for i, line in enumerate(f):
                temp = json.loads(line)
                for i, data in enumerate(temp['mention_data']):
                    if data['mention'] in entity_to_kbids:
                        #                 print(data['mention'], entity_to_kbids[data['mention']])
                        data['kb_id'] = random.choice(list(entity_to_kbids[data['mention']]))
                    else:
                        data['kb_id'] = 'NIL_' + random.choice(idx_to_type)
                result.append(temp)

        with open(save_path, 'w') as f:
            for r in result:
                json.dump(r, f, ensure_ascii=False)
                f.write('\n')


def main():
    data_path = Path('../../data')
    file_path = data_path / 'ccks2020_el_data_v1' / 'dev.json'
    pickle_path = data_path / 'pickle'
    save_path = data_path / 'result' / 'result.json'
    RandomSearchModel.generate_result(file_path, save_path, pickle_path)
    e = eval.Eval(file_path, save_path)
    prec, recall, f1 = e.micro_f1()
    print(prec, recall, f1)


if __name__ == '__main__':
    main()
