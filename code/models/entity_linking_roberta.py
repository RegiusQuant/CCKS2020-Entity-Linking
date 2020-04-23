# -*- coding: utf-8 -*-
# @Time    : 2020/4/23 下午9:32
# @Author  : RegiusQuant <315135833@qq.com>
# @Project : CCKS2020-Entity-Linking
# @File    : entity_linking_roberta.py
# @Desc    : 实体链接RoBERTa模型

from core import *


class EntityLinkingProcessor(DataProcessor):
    """实体链接数据处理"""

    def get_train_examples(self, file_path):
        return self._create_examples(
            self._read_tsv(file_path),
            set_type='train',
        )

    def get_dev_examples(self, file_path):
        return self._create_examples(
            self._read_tsv(file_path),
            set_type='valid',
        )

    def get_test_examples(self, file_path):
        return self._create_examples(
            self._read_tsv(file_path),
            set_type='test',
        )

    def get_labels(self):
        return ['0', '1']

    def _create_examples(self, lines, set_type):
        examples = []
        for i, line in enumerate(lines):
            if i == 0:
                continue
            guid = f'{set_type}-{i}'
            text_a = line[1] + ' ' + line[3]
            text_b = line[5]
            label = line[-1]
            examples.append(InputExample(
                guid=guid,
                text_a=text_a,
                text_b=text_b,
                label=label,
            ))
        return examples

    def create_dataloader(self, examples, tokenizer, max_length=384, shuffle=False, batch_size=32):

        features = glue_convert_examples_to_features(
            examples,
            tokenizer,
            label_list=self.get_labels(),
            max_length=max_length,
            output_mode='classification',
            pad_on_left=False,
            pad_token=tokenizer.pad_token_id,
            pad_token_segment_id=0,
        )

        dataset = torch.utils.data.TensorDataset(
            torch.LongTensor([f.input_ids for f in features]), torch.LongTensor([f.attention_mask for f in features]),
            torch.LongTensor([f.token_type_ids for f in features]), torch.LongTensor([f.label for f in features]))

        dataloader = torch.utils.data.DataLoader(
            dataset,
            shuffle=shuffle,
            batch_size=batch_size,
        )
        return dataloader


class EntityLinkingModel(pl.LightningModule):
    """实体链接模型"""

    def __init__(self, max_length=384, batch_size=32):
        super(EntityLinkingModel, self).__init__()
        # 输入最大长度
        self.max_length = max_length
        self.batch_size = batch_size

        self.tokenizer = BertTokenizer.from_pretrained(PRETRAINED_PATH)

        # 预训练模型配置信息
        self.config = BertConfig.from_json_file(PRETRAINED_PATH + 'bert_config.json')
        self.config.num_labels = 1

        # 预训练模型
        self.bert = BertForSequenceClassification.from_pretrained(
            PRETRAINED_PATH + 'pytorch_model.bin',
            config=self.config,
        )

        # 二分类损失函数
        self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, input_ids, attention_mask, token_type_ids):
        logits = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )[0]
        return logits.squeeze()

    def prepare_data(self):
        self.processor = EntityLinkingProcessor()
        self.train_examples = self.processor.get_train_examples(TSV_PATH + 'EL_TRAIN.tsv')
        self.valid_examples = self.processor.get_dev_examples(TSV_PATH + 'EL_VALID.tsv')
        self.test_examples = self.processor.get_test_examples(TSV_PATH + 'EL_TEST.tsv')

        self.train_loader = self.processor.create_dataloader(
            examples=self.train_examples,
            tokenizer=self.tokenizer,
            max_length=self.max_length,
            shuffle=True,
            batch_size=32,
        )
        self.valid_loader = self.processor.create_dataloader(
            examples=self.valid_examples,
            tokenizer=self.tokenizer,
            max_length=self.max_length,
            shuffle=False,
            batch_size=32,
        )
        self.test_loader = self.processor.create_dataloader(
            examples=self.test_examples,
            tokenizer=self.tokenizer,
            max_length=self.max_length,
            shuffle=False,
            batch_size=32,
        )

    def training_step(self, batch, batch_idx):
        input_ids, attention_mask, token_type_ids, labels = batch
        logits = self(input_ids, attention_mask, token_type_ids)
        loss = self.criterion(logits, labels.float())

        preds = (logits > 0).int()
        acc = (preds == labels).float().mean()

        tensorboard_logs = {'train_loss': loss, 'train_acc': acc}
        return {'loss': loss, 'log': tensorboard_logs, 'progress_bar': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        input_ids, attention_mask, token_type_ids, labels = batch
        logits = self(input_ids, attention_mask, token_type_ids)
        loss = self.criterion(logits, labels.float())

        preds = (logits > 0).int()
        acc = (preds == labels).float().mean()

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
