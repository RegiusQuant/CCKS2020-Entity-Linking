# -*- coding: utf-8 -*-
# @Time    : 2020/4/23 下午2:29
# @Author  : RegiusQuant <315135833@qq.com>
# @Project : CCKS2020-Entity-Linking
# @File    : main.py
# @Desc    : 程序运行主文件

from data import PicklePreprocessor, DataFramePreprocessor
from models.entity_linking_roberta import (
    EntityLinkingModel,
    EntityLinkingProcessor,
    EntityLinkingPredictor,
)
from models.entity_typing_roberta import (
    EntityTypingModel,
    EntityTypingProcessor,
    EntityTypingPredictor,
)
from core import *


def preprocess_pickle_file():
    processor = PicklePreprocessor()
    processor.run()


def preprocess_tsv_file():
    processor = DataFramePreprocessor()
    processor.run()


def generate_feature_pickle():
    processor = EntityLinkingProcessor()
    processor.generate_feature_pickle(max_length=384)

    processor = EntityTypingProcessor()
    processor.generate_feature_pickle(max_length=64)


def train_entity_linking_model(ckpt_name):
    model = EntityLinkingModel(max_length=384, batch_size=32)
    trainer = pl.Trainer(
        max_epochs=1,
        gpus=2,
        distributed_backend='dp',
        default_save_path=EL_SAVE_PATH,
        profiler=True,
    )
    trainer.fit(model)
    trainer.save_checkpoint(CKPT_PATH + ckpt_name)


def train_entity_typing_model(ckpt_name):
    model = EntityTypingModel(max_length=64, batch_size=64)
    trainer = pl.Trainer(
        max_epochs=2,
        gpus=2,
        distributed_backend='dp',
        default_save_path=ET_SAVE_PATH,
        profiler=True,
    )
    trainer.fit(model)
    trainer.save_checkpoint(CKPT_PATH + ckpt_name)


def generate_link_tsv_result(ckpt_name):
    predictor = EntityLinkingPredictor(ckpt_name, batch_size=24, use_pickle=True)
    predictor.generate_tsv_result('EL_VALID.tsv', tsv_type='Valid')
    predictor.generate_tsv_result('EL_TEST.tsv', tsv_type='Test')


def generate_type_tsv_result(ckpt_name):
    predictor = EntityTypingPredictor(ckpt_name, batch_size=64, use_pickle=True)
    predictor.generate_tsv_result('ET_VALID.tsv', tsv_type='Valid')
    predictor.generate_tsv_result('ET_TEST.tsv', tsv_type='Test')


def make_predication_result(input_name, output_name, el_ret_name, et_ret_name):
    entity_to_kbids = PICKLE_DATA['ENTITY_TO_KBIDS']

    el_ret = pd.read_csv(
        RESULT_PATH + el_ret_name, sep='\t', dtype={
            'text_id': np.str_,
            'offset': np.str_,
            'kb_id': np.str_
        })
    et_ret = pd.read_csv(RESULT_PATH + et_ret_name, sep='\t', dtype={'text_id': np.str_, 'offset': np.str_})

    result = []
    with open(RAW_PATH + input_name, 'r') as f:
        for line in tqdm(f):
            line = json.loads(line)
            for data in line['mention_data']:
                text_id = line['text_id']
                offset = data['offset']

                candidate_data = el_ret[(el_ret['text_id'] == text_id) & (el_ret['offset'] == offset)]
                # Entity Linking
                if len(candidate_data) > 0:
                    max_idx = candidate_data['logits'].idxmax()
                    data['kb_id'] = candidate_data.loc[max_idx]['kb_id']
                # Entity Typing
                else:
                    type_data = et_ret[(et_ret['text_id'] == text_id) & (et_ret['offset'] == offset)]
                    data['kb_id'] = 'NIL_' + type_data.iloc[0]['result']
            result.append(line)

    with open(RESULT_PATH + output_name, 'w') as f:
        for r in result:
            json.dump(r, f, ensure_ascii=False)
            f.write('\n')


if __name__ == '__main__':
    set_random_seed(2020)
    # preprocess_pickle_file()
    # preprocess_tsv_file()
    # generate_feature_pickle()

    # train_entity_linking_model('EL_BASE_EPOCH0.ckpt')
    # generate_link_tsv_result('EL_BASE_EPOCH0.ckpt')

    # train_entity_typing_model('ET_BASE_EPOCH1.ckpt')
    # generate_type_tsv_result('ET_BASE_EPOCH1.ckpt')

    # make_predication_result('dev.json', 'valid_result.json', 'EL_VALID_RESULT.tsv', 'ET_VALID_RESULT.tsv')
    make_predication_result('test.json', 'test_result.json', 'EL_TEST_RESULT.tsv', 'ET_TEST_RESULT.tsv')