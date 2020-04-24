# -*- coding: utf-8 -*-
# @Time    : 2020/4/24 下午6:51
# @Author  : RegiusQuant <315135833@qq.com>
# @Project : CCKS2020-Entity-Linking
# @File    : entity_typing_roberta.py
# @Desc    : 实体类别推断RoBERTa模型

from core import *


class EntityTypingProcessor(DataProcessor):
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
        return PICKLE_DATA['IDX_TO_TYPE']

    def _create_examples(self, lines, set_type):
        examples = []
        for i, line in enumerate(lines):
            if i == 0:
                continue
            guid = f'{set_type}-{i}'
            text_a = line[1]
            text_b = line[3]
            label = line[-1]
            examples.append(InputExample(
                guid=guid,
                text_a=text_a,
                text_b=text_b,
                label=label,
            ))
        return examples

    def create_dataloader(self, examples, tokenizer, max_length=64,
                          shuffle=False, batch_size=64, use_pickle=False):
        pickle_name = 'ET_FEATURE_' + examples[0].guid.split('-')[0].upper() + '.pkl'
        if use_pickle:
            features = pd.read_pickle(PICKLE_PATH + pickle_name)
        else:
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
            pd.to_pickle(features, PICKLE_PATH + pickle_name)

        dataset = torch.utils.data.TensorDataset(
            torch.LongTensor([f.input_ids for f in features]),
            torch.LongTensor([f.attention_mask for f in features]),
            torch.LongTensor([f.token_type_ids for f in features]),
            torch.LongTensor([f.label for f in features]),
        )

        dataloader = torch.utils.data.DataLoader(
            dataset,
            shuffle=shuffle,
            batch_size=batch_size,
            num_workers=2,
        )
        return dataloader

    def generate_feature_pickle(self, max_length):
        tokenizer = BertTokenizer.from_pretrained(PRETRAINED_PATH)

        train_examples = self.get_train_examples(TSV_PATH + 'ET_TRAIN.tsv')
        valid_examples = self.get_dev_examples(TSV_PATH + 'ET_VALID.tsv')
        test_examples = self.get_test_examples(TSV_PATH + 'ET_TEST.tsv')

        self.create_dataloader(
            examples=train_examples,
            tokenizer=tokenizer,
            max_length=max_length,
            shuffle=True,
            batch_size=32,
            use_pickle=False,
        )
        self.create_dataloader(
            examples=valid_examples,
            tokenizer=tokenizer,
            max_length=max_length,
            shuffle=False,
            batch_size=32,
            use_pickle=False,
        )
        self.create_dataloader(
            examples=test_examples,
            tokenizer=tokenizer,
            max_length=max_length,
            shuffle=False,
            batch_size=32,
            use_pickle=False,
        )

class EntityTypingModel(pl.LightningModule):
    """实体类型推断模型"""

    def __init__(self, max_length=64, batch_size=64, use_pickle=True):
        super(EntityTypingModel, self).__init__()
        # 输入最大长度
        self.max_length = max_length
        self.batch_size = batch_size
        self.use_pickle = use_pickle

        self.tokenizer = BertTokenizer.from_pretrained(PRETRAINED_PATH)

        # 预训练模型配置信息
        self.config = BertConfig.from_json_file(PRETRAINED_PATH + 'bert_config.json')
        self.config.num_labels = len(PICKLE_DATA['IDX_TO_TYPE'])

        # 预训练模型
        self.bert = BertForSequenceClassification.from_pretrained(
            PRETRAINED_PATH + 'pytorch_model.bin',
            config=self.config,
        )

        # 二分类损失函数
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, input_ids, attention_mask, token_type_ids):
        return self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )[0]

    def prepare_data(self):
        self.processor = EntityTypingProcessor()
        self.train_examples = self.processor.get_train_examples(TSV_PATH + 'ET_TRAIN.tsv')
        self.valid_examples = self.processor.get_dev_examples(TSV_PATH + 'ET_VALID.tsv')
        self.test_examples = self.processor.get_test_examples(TSV_PATH + 'ET_TEST.tsv')

        self.train_loader = self.processor.create_dataloader(
            examples=self.train_examples,
            tokenizer=self.tokenizer,
            max_length=self.max_length,
            shuffle=True,
            batch_size=self.batch_size,
            use_pickle=self.use_pickle,
        )
        self.valid_loader = self.processor.create_dataloader(
            examples=self.valid_examples,
            tokenizer=self.tokenizer,
            max_length=self.max_length,
            shuffle=False,
            batch_size=self.batch_size,
            use_pickle=self.use_pickle,
        )
        self.test_loader = self.processor.create_dataloader(
            examples=self.test_examples,
            tokenizer=self.tokenizer,
            max_length=self.max_length,
            shuffle=False,
            batch_size=self.batch_size,
            use_pickle=self.use_pickle,
        )

    def training_step(self, batch, batch_idx):
        input_ids, attention_mask, token_type_ids, labels = batch
        outputs = self(input_ids, attention_mask, token_type_ids)
        loss = self.criterion(outputs, labels)

        _, preds = torch.max(outputs, dim=1)
        acc = (preds == labels).float().mean()

        tensorboard_logs = {'train_loss': loss, 'train_acc': acc}
        return {'loss': loss, 'log': tensorboard_logs, 'progress_bar': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        input_ids, attention_mask, token_type_ids, labels = batch
        outputs = self(input_ids, attention_mask, token_type_ids)
        loss = self.criterion(outputs, labels)

        _, preds = torch.max(outputs, dim=1)
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


class EntityTypingPredictor:

    def __init__(self, ckpt_name, batch_size=8, use_pickle=True):
        self.ckpt_name = ckpt_name
        self.batch_size = batch_size
        self.use_pickle = use_pickle

    def generate_tsv_result(self, tsv_name, tsv_type='Valid'):
        processor = EntityTypingProcessor()
        tokenizer = BertTokenizer.from_pretrained(PRETRAINED_PATH)

        if tsv_type == 'Valid':
            examples = processor.get_dev_examples(TSV_PATH + tsv_name)
        elif tsv_type == 'Test':
            examples = processor.get_test_examples(TSV_PATH + tsv_name)
        else:
            raise ValueError('tsv_type error')
        dataloader = processor.create_dataloader(
            examples=examples,
            tokenizer=tokenizer,
            max_length=64,
            shuffle=False,
            batch_size=self.batch_size,
            use_pickle=self.use_pickle,
        )

        model = EntityTypingModel.load_from_checkpoint(
            checkpoint_path=CKPT_PATH + self.ckpt_name,
        )
        model.to(DEVICE)
        model = nn.DataParallel(model)
        model.eval()

        result_list = []
        for batch in tqdm(dataloader):
            for i in range(len(batch)):
                batch[i] = batch[i].to(DEVICE)

            input_ids, attention_mask, token_type_ids, labels = batch
            outputs = model(input_ids, attention_mask, token_type_ids)
            _, preds = torch.max(outputs, dim=1)
            result_list.extend(preds.tolist())

        idx_to_type = PICKLE_DATA['IDX_TO_TYPE']
        result_list = [idx_to_type[x] for x in result_list]
        tsv_data = pd.read_csv(TSV_PATH + tsv_name, sep='\t')
        tsv_data['result'] = result_list
        result_name = tsv_name.split('.')[0] + '_RESULT.tsv'
        tsv_data.to_csv(RESULT_PATH + result_name, index=False, sep='\t')
