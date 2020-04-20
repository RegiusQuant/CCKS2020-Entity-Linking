# -*- coding: utf-8 -*-
# @Time    : 2020/4/20 下午1:12
# @Author  : RegiusQuant <315135833@qq.com>
# @Project : CCKS2020-Entity-Linking
# @File    : model_roberta.py
# @Desc    : 使用RoBERTa模型推断

import json
import logging
import random
from pathlib import Path

import numpy as np
import pandas as pd
import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from transformers import (
    DataProcessor,
    InputExample,
    BertConfig,
    BertTokenizer,
    BertForSequenceClassification,
    glue_convert_examples_to_features,
)

logging.getLogger('transformers.data.processors.glue').setLevel(logging.WARNING)
idx_to_type = pd.read_pickle('../../data/pickle/idx_to_type.pkl')


class ELProcessor(DataProcessor):

    def get_train_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(data_dir / 'train_link.csv'),
            set_type='train',
        )

    def get_dev_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(data_dir / 'valid_link.csv'),
            set_type='valid',
        )

    def get_infer_examples(self, entity, rawtext, kbtexts):
        examples = []
        for i, kbtext in enumerate(kbtexts):
            guid = f'test-{i}'
            text_a = entity + ' ' + rawtext
            text_b = kbtext
            examples.append(InputExample(
                guid=guid,
                text_a=text_a,
                text_b=text_b,
                label='0'
            ))
        return examples

    def get_labels(self):
        return ['0', '1']

    def _create_examples(self, lines, set_type):
        examples = []
        for i, line in enumerate(lines):
            if i == 0:
                continue
            guid = f'{set_type}-{i}'
            try:
                text_a = line[0] + ' ' + line[2]
                text_b = line[3]
                label = line[4]
                examples.append(InputExample(
                    guid=guid,
                    text_a=text_a,
                    text_b=text_b,
                    label=label,
                ))
            except:
                print(i)
                print(line)
        return examples


class ETProcessor(DataProcessor):

    def get_train_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(data_dir / 'train_type.csv'),
            set_type='train',
        )

    def get_dev_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(data_dir / 'valid_type.csv'),
            set_type='valid',
        )

    def get_infer_examples(self, entity, rawtext):
        examples = []
        examples.append(InputExample(
            guid=f'test-0',
            text_a=entity,
            text_b=rawtext,
            label='Other',
        ))
        return examples

    def get_labels(self):
        return idx_to_type

    def _create_examples(self, lines, set_type):
        examples = []
        for i, line in enumerate(lines):
            if i == 0:
                continue
            guid = f'{set_type}-{i}'
            try:
                text_a = line[0]
                text_b = line[2]
                label = line[3]
                examples.append(InputExample(
                    guid=guid,
                    text_a=text_a,
                    text_b=text_b,
                    label=label,
                ))
            except:
                print(i)
                print(line)
        return examples


class ELRoBERTaModel(pl.LightningModule):

    def __init__(self,
                 pretrained_path,
                 train_loader,
                 valid_loader):
        super(ELRoBERTaModel, self).__init__()
        self.train_loader = train_loader
        self.valid_loader = valid_loader

        # 预训练模型
        self.ptm = BertForSequenceClassification.from_pretrained(
            pretrained_path + '/pytorch_model.bin',
            config=pretrained_path + '/bert_config.json',
        )

        # 损失函数
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, input_ids, attention_mask, token_type_ids):
        return self.ptm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )[0]

    def training_step(self, batch, batch_idx):
        input_ids, attention_mask, token_type_ids, label = batch
        out = self(input_ids, attention_mask, token_type_ids)

        loss = self.criterion(out, label)

        _, pred = torch.max(out, dim=1)
        acc = (pred == label).float().mean()

        tensorboard_logs = {'train_loss': loss, 'train_acc': acc}
        return {'loss': loss, 'log': tensorboard_logs, 'progress_bar': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        input_ids, attention_mask, token_type_ids, label = batch
        out = self(input_ids, attention_mask, token_type_ids)

        loss = self.criterion(out, label)

        _, pred = torch.max(out, dim=1)
        acc = (pred == label).float().mean()

        return {'val_loss': loss, 'val_acc': acc}

    def validation_epoch_end(self, outputs):
        val_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        val_acc = torch.stack([x['val_acc'] for x in outputs]).mean()

        tensorboard_logs = {'val_loss': val_loss, 'val_acc': val_acc}
        return {'val_loss': val_loss, 'log': tensorboard_logs, 'progress_bar': tensorboard_logs}

    def configure_optimizers(self):
        return torch.optim.Adam([p for p in self.parameters() if p.requires_grad], lr=2e-5, eps=1e-8)

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.valid_loader


class ETRoBERTaModel(pl.LightningModule):

    def __init__(self,
                 pretrained_path,
                 train_loader,
                 valid_loader):
        super(ETRoBERTaModel, self).__init__()
        self.train_loader = train_loader
        self.valid_loader = valid_loader

        config = BertConfig.from_json_file(pretrained_path + '/bert_config.json')
        config.num_labels = len(idx_to_type)
        print(config)

        # 预训练模型
        self.ptm = BertForSequenceClassification.from_pretrained(
            pretrained_path + '/pytorch_model.bin',
            config=config,
        )

        # 损失函数
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, input_ids, attention_mask, token_type_ids):
        return self.ptm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )[0]

    def training_step(self, batch, batch_idx):
        input_ids, attention_mask, token_type_ids, label = batch
        out = self(input_ids, attention_mask, token_type_ids)

        loss = self.criterion(out, label)

        _, pred = torch.max(out, dim=1)
        acc = (pred == label).float().mean()

        tensorboard_logs = {'train_loss': loss, 'train_acc': acc}
        return {'loss': loss, 'log': tensorboard_logs, 'progress_bar': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        input_ids, attention_mask, token_type_ids, label = batch
        out = self(input_ids, attention_mask, token_type_ids)

        loss = self.criterion(out, label)

        _, pred = torch.max(out, dim=1)
        acc = (pred == label).float().mean()

        return {'val_loss': loss, 'val_acc': acc}

    def validation_epoch_end(self, outputs):
        val_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        val_acc = torch.stack([x['val_acc'] for x in outputs]).mean()

        tensorboard_logs = {'val_loss': val_loss, 'val_acc': val_acc}
        return {'val_loss': val_loss, 'log': tensorboard_logs, 'progress_bar': tensorboard_logs}

    def configure_optimizers(self):
        return torch.optim.Adam([p for p in self.parameters() if p.requires_grad], lr=2e-5, eps=1e-8)

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.valid_loader


class RoBERTaPredictor:

    def __init__(self,
                 pretrained_path,
                 el_ckpt_path,
                 et_ckpt_path):
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_path)
        self.el_model = ELRoBERTaModel.load_from_checkpoint(
            checkpoint_path=el_ckpt_path,
            pretrained_path=pretrained_path,
            train_loader=None,
            valid_loader=None,
        )
        self.el_model.eval()
        self.et_model = ETRoBERTaModel.load_from_checkpoint(
            checkpoint_path=et_ckpt_path,
            pretrained_path=pretrained_path,
            train_loader=None,
            valid_loader=None,
        )
        self.et_model.eval()
        self.el_processor = ELProcessor()
        self.et_processor = ETProcessor()

    def predict_el_result(self, entity, rawtext, kbtexts):
        examples = self.el_processor.get_infer_examples(entity, rawtext, kbtexts)
        features = glue_convert_examples_to_features(
            examples,
            self.tokenizer,
            label_list=['0', '1'],
            max_length=128,
            output_mode='classification',
            pad_on_left=False,
            pad_token=self.tokenizer.pad_token_id,
            pad_token_segment_id=0)

        dataset = torch.utils.data.TensorDataset(
            torch.LongTensor([f.input_ids for f in features]),
            torch.LongTensor([f.attention_mask for f in features]),
            torch.LongTensor([f.token_type_ids for f in features]),
            torch.LongTensor([f.label for f in features])
        )
        dataloader = torch.utils.data.DataLoader(
            dataset, shuffle=False, batch_size=32
        )

        scores = []
        for batch in dataloader:
            input_ids, attention_mask, token_type_ids, label = batch
            outputs = self.el_model(input_ids, attention_mask, token_type_ids)
            outputs = F.softmax(outputs, dim=-1)[:, 1]
            scores.extend(outputs.tolist())
        scores = np.array(scores)
        return np.argmax(scores)

    def predict_et_result(self, entity, rawtext):
        examples = self.et_processor.get_infer_examples(entity, rawtext)
        features = glue_convert_examples_to_features(
            examples,
            self.tokenizer,
            label_list=idx_to_type,
            max_length=64,
            output_mode='classification',
            pad_on_left=False,
            pad_token=self.tokenizer.pad_token_id,
            pad_token_segment_id=0)

        dataset = torch.utils.data.TensorDataset(
            torch.LongTensor([f.input_ids for f in features]),
            torch.LongTensor([f.attention_mask for f in features]),
            torch.LongTensor([f.token_type_ids for f in features]),
            torch.LongTensor([f.label for f in features])
        )
        dataloader = torch.utils.data.DataLoader(
            dataset, shuffle=False, batch_size=32
        )

        for batch in dataloader:
            input_ids, attention_mask, token_type_ids, label = batch
            outputs = self.et_model(input_ids, attention_mask, token_type_ids)
            return outputs[0].argmax().item()

    def generate_result(self, file_path, save_path, pickle_path):
        entity_to_kbids = pd.read_pickle(pickle_path / 'entity_to_kbids.pkl')
        kbid_to_text = pd.read_pickle(pickle_path / 'kbid_to_text.pkl')
        idx_to_type = pd.read_pickle(pickle_path / 'idx_to_type.pkl')

        result = []
        with open(file_path, 'r') as f:
            for i, line in enumerate(tqdm.tqdm(f)):
                temp = json.loads(line)
                for _, data in enumerate(temp['mention_data']):
                    # Entity Linking
                    if data['mention'] in entity_to_kbids:
                        kbids = list(entity_to_kbids[data['mention']])
                        if len(kbids) == 1:
                            data['kb_id'] = kbids[0]
                        else:
                            entity = data['mention']
                            rawtext = temp['text']
                            kbtexts = [kbid_to_text[x] for x in kbids]
                            idx = self.predict_el_result(entity, rawtext, kbtexts)
                            data['kb_id'] = kbids[idx]
                    # Entity Typing
                    else:
                        idx = self.predict_et_result(
                            entity=data['mention'],
                            rawtext=temp['text']
                        )
                        data['kb_id'] = 'NIL_' + idx_to_type[idx]
                result.append(temp)

        with open(save_path, 'w') as f:
            for r in result:
                json.dump(r, f, ensure_ascii=False)
                f.write('\n')


def main():
    el_ckpt_path = '../../ckpt/EL-RoBERTa-128-0419.ckpt'
    et_ckpt_path = '../../ckpt/ET-RoBERTa-64-0420.ckpt'
    pretrained_path = '/media/bnu/data/transformers-pretrained-model/chinese_roberta_wwm_ext_pytorch'
    predictor = RoBERTaPredictor(pretrained_path, el_ckpt_path, et_ckpt_path)

    data_path = Path('../../data')
    file_path = data_path / 'ccks2020_el_data_v1' / 'test.json'
    pickle_path = data_path / 'pickle'
    save_path = data_path / 'result' / 'test_result.json'
    predictor.generate_result(file_path, save_path, pickle_path)


if __name__ == '__main__':
    main()
