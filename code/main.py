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


if __name__ == '__main__':
    set_random_seed(2020)
    # preprocess_pickle_file()
    # preprocess_tsv_file()
    # generate_feature_pickle()

    # train_entity_linking_model('EL_BASE_EPOCH0.ckpt')
    # generate_link_tsv_result('EL_BASE_EPOCH0.ckpt')

    # train_entity_typing_model('ET_BASE_EPOCH1.ckpt')
    generate_type_tsv_result('ET_BASE_EPOCH1.ckpt')
