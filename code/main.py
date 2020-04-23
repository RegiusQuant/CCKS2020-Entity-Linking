# -*- coding: utf-8 -*-
# @Time    : 2020/4/23 下午2:29
# @Author  : RegiusQuant <315135833@qq.com>
# @Project : CCKS2020-Entity-Linking
# @File    : main.py
# @Desc    : 程序运行主文件

from data import PicklePreprocessor, DataFramePreprocessor
from models.entity_linking_roberta import EntityLinkingModel
from core import *


def preprocess_pickle_file():
    processor = PicklePreprocessor()
    processor.run()


def preprocess_tsv_file():
    processor = DataFramePreprocessor()
    processor.run()


def train_entity_linking_model():
    model = EntityLinkingModel(max_length=384, batch_size=32)
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        filepath=EL_SAVE_PATH,
        save_top_k=True,
        verbose=True,
        monitor='val_loss',
        mode='min',
        prefix='EL_',
    )
    trainer = pl.Trainer(
        max_epochs=1,
        checkpoint_callback=checkpoint_callback,
        gpus=2,
        distributed_backend='dp',
        default_save_path=EL_SAVE_PATH,
        profiler=True,
    )
    trainer.fit(model)


if __name__ == '__main__':
    set_random_seed(2020)
    # preprocess_pickle_file()
    # preprocess_tsv_file()
    train_entity_linking_model()
