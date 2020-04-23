# -*- coding: utf-8 -*-
# @Time    : 2020/4/23 下午2:29
# @Author  : RegiusQuant <315135833@qq.com>
# @Project : CCKS2020-Entity-Linking
# @File    : main.py
# @Desc    : 程序运行主文件

from data import PicklePreprocessor, DataFramePreprocessor


def preprocess_pickle_file():
    processor = PicklePreprocessor()
    processor.run()
    print('Process Pickle File Finish.')


def preprocess_tsv_file():
    processor = DataFramePreprocessor()
    processor.run()
    print('Process TSV File Finish.')


if __name__ == '__main__':
    # preprocess_pickle_file()
    preprocess_tsv_file()
