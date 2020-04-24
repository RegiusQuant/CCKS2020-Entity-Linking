# -*- coding: utf-8 -*-
# @Time    : 2020/4/24 下午3:00
# @Author  : RegiusQuant <315135833@qq.com>
# @Project : CCKS2020-Entity-Linking
# @File    : test.py
# @Desc    : 功能测试

from core import *
from models.entity_linking_roberta import EntityLinkingProcessor
from models.entity_typing_roberta import EntityTypingProcessor


def test_create_dataloader():
    processor = EntityTypingProcessor()
    tokenizer = BertTokenizer.from_pretrained(PRETRAINED_PATH)
    use_pickle = False

    train_examples = processor.get_train_examples(TSV_PATH + 'ET_TRAIN.tsv')
    valid_examples = processor.get_dev_examples(TSV_PATH + 'ET_VALID.tsv')
    test_examples = processor.get_test_examples(TSV_PATH + 'ET_TEST.tsv')

    train_loader = processor.create_dataloader(
        examples=train_examples,
        tokenizer=tokenizer,
        max_length=64,
        shuffle=True,
        batch_size=64,
        use_pickle=use_pickle,
    )
    valid_loader = processor.create_dataloader(
        examples=valid_examples,
        tokenizer=tokenizer,
        max_length=64,
        shuffle=False,
        batch_size=64,
        use_pickle=use_pickle,
    )
    test_loader = processor.create_dataloader(
        examples=test_examples,
        tokenizer=tokenizer,
        max_length=64,
        shuffle=False,
        batch_size=64,
        use_pickle=use_pickle,
    )

    for loader in [train_loader, valid_loader, test_loader]:
        for batch in loader:
            print(batch)
            print('-' * 100)
            break


def test_evaluate():
    result_data = pd.read_csv(RESULT_PATH+'EL_VALID_RESULT.tsv', sep='\t')
    print(accuracy_score(result_data['predict'], result_data['result']))


if __name__ == '__main__':
    test_create_dataloader()
    # test_evaluate()