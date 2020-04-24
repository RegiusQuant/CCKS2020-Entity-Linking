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
from core import *


def preprocess_pickle_file():
    processor = PicklePreprocessor()
    processor.run()


def preprocess_tsv_file():
    processor = DataFramePreprocessor()
    processor.run()


def generate_feature_pickle():
    processor = EntityLinkingProcessor()
    tokenizer = BertTokenizer.from_pretrained(PRETRAINED_PATH)

    train_examples = processor.get_train_examples(TSV_PATH + 'EL_TRAIN.tsv')
    valid_examples = processor.get_dev_examples(TSV_PATH + 'EL_VALID.tsv')
    test_examples = processor.get_test_examples(TSV_PATH + 'EL_TEST.tsv')

    processor.create_dataloader(
        examples=train_examples,
        tokenizer=tokenizer,
        max_length=384,
        shuffle=True,
        batch_size=32,
        use_pickle=False,
    )
    processor.create_dataloader(
        examples=valid_examples,
        tokenizer=tokenizer,
        max_length=384,
        shuffle=False,
        batch_size=32,
        use_pickle=False,
    )
    processor.create_dataloader(
        examples=test_examples,
        tokenizer=tokenizer,
        max_length=384,
        shuffle=False,
        batch_size=32,
        use_pickle=False,
    )


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


def generate_tsv_result(ckpt_name):
    predictor = EntityLinkingPredictor(ckpt_name, batch_size=24, use_pickle=True)
    # predictor.generate_tsv_result('EL_VALID.tsv', tsv_type='Valid')
    predictor.generate_tsv_result('EL_TEST.tsv', tsv_type='Test')


if __name__ == '__main__':
    set_random_seed(2020)
    # preprocess_pickle_file()
    # preprocess_tsv_file()
    # generate_feature_pickle()
    # train_entity_linking_model('EL_BASE_EPOCH0.ckpt')
    generate_tsv_result('EL_BASE_EPOCH0.ckpt')