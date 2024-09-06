# Copyright (c) 2019, Facebook, Inc. and its affiliates. All Rights Reserved
"""BERT finetuning runner."""

from __future__ import absolute_import, division, print_function

import argparse
import csv
import logging
import os
import random
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from torch.nn import CrossEntropyLoss, MSELoss
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import matthews_corrcoef, f1_score

from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE, WEIGHTS_NAME, CONFIG_NAME
from pytorch_pretrained_bert.modeling import BertForSequenceClassification
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.optimization import BertAdam, warmup_linear

"""文件名设置"""
PRED_FILE = "predictions.tsv"
EVAL_FILE = "eval_results.txt"
TEST_FILE = "test_results.txt"

"""日志配置"""
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


class InputExample(object):
    """
    A single training/test example for simple sequence classification.

    这个类用于表示单个训练/测试示例,特别是在简单的序列分类任务中。它封装了每个示例的基本信息,如唯一标识符(guid)、第一个序列的未分词文本(text_a)、可选的第二个序列的未分词文本(text_b),以及该示例的标签(label,可选)。

    第一个序列的未分词文本: 当我们提到"第一个序列的未分词文本"时,我们通常是在谈论一段原始文本数据,这段数据还没有被转换成模型能够直接处理的格式(比如词汇索引或词嵌入)。这里的"序列"指的是一系列有序的元素,在自然语言处理中,这些元素通常是单词或字符。
    """

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class InputFeatures(object):
    """
    A single set of features of data.
    
    它用于表示单个数据集的特征集。在自然语言处理(NLP)任务中,尤其是在使用像BERT这样的预训练模型进行微调时,数据需要被转换成特定的格式,以便模型能够理解和处理。InputFeatures类就封装了这种转换后的数据格式。
    """

    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        # 这是一个整数列表(或其他可迭代对象),表示输入文本的词汇索引。
        self.input_ids = input_ids
        # 一个整数列表,用于指示哪些元素是实际的输入,哪些元素是填充(padding)用于保持序列长度一致的
        self.input_mask = input_mask
        # 整数列表,用于区分序列中的不同部分(例如,在问答任务中,问题和答案可能被视为不同的部分)
        self.segment_ids = segment_ids
        # 一个整数,表示该数据集的标签或目标的索引
        self.label_id = label_id


class DataProcessor(object):
    """
    Base class for data converters for sequence classification data sets.
    
    旨在为序列分类数据集的数据转换器提供一个基础框架。这个类本身不直接用于数据处理,而是作为其他具体数据处理器(如用于特定数据集的子类)的模板。通过实现这个基类中的抽象方法,子类可以定制如何读取和处理特定数据集的训练集、开发集(也称为验证集)以及获取数据集的标签列表。
    """

    def get_train_examples(self, data_dir):
        """
        Gets a collection of `InputExample`s for the train set.
        从给定的data_dir(数据集目录)中读取训练集的数据,并返回一个包含InputExample实例的集合
        """
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """
        Gets a collection of `InputExample`s for the dev set.
        读取开发集(或验证集)的数据
        """
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """
        Reads a tab separated value file.
        读取制表符分隔值(TSV)文件: 
        接受一个输入文件路径input_file和一个可选的quotechar参数(用于处理被引号包围的字段),然后使用csv.reader以制表符为分隔符读取文件内容,并将每行作为一个列表返回
        """
        with open(input_file, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                if sys.version_info[0] == 2:
                    line = list(cell for cell in line)
                lines.append(line)
            return lines


class MrpcProcessor(DataProcessor):
    """
    Processor for the MRPC data set (GLUE version).
    
    MrpcProcessor类继承自DataProcessor类,是专门为处理MRPC(Microsoft Research Paraphrase Corpus)数据集(特别是GLUE版本的MRPC)而设计的。
    
    MRPC是一个用于评估句子对语义等价性(即判断两个句子是否具有相同的含义)的基准数据集。这个类通过重写DataProcessor基类中的方法,提供了从MRPC数据集的TSV文件中读取训练集、开发集(验证集)和测试集数据,并创建InputExample实例的功能。
    """

    def get_train_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {}".format(os.path.join(data_dir, "train.tsv")))
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """这是一个辅助方法,用于从给定的行列表(lines)中创建InputExample实例"""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a = line[3]
            text_b = line[4]
            label = None if set_type == "test" else line[0]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class MnliProcessor(DataProcessor):
    """
    Processor for the MultiNLI data set (GLUE version).

    MnliProcessor类继承自DataProcessor类,专门用于处理MultiNLI(Multi-Genre Natural Language Inference)数据集(特别是GLUE版本的MultiNLI)。MultiNLI是一个用于评估自然语言推理(NLI)系统性能的大型数据集,它包含了来自不同来源和领域的句子对,每个句子对都被标注为三种关系之一：蕴含(entailment)、矛盾(contradiction)或中立(neutral)。
    """

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir, eval_set="MNLI-m"):
        """See base class."""
        if eval_set is None or eval_set == "MNLI-m":
            return self._create_examples(
                self._read_tsv(os.path.join(data_dir, "dev_matched.tsv")), "dev")
        else:
            assert eval_set == "MNLI-mm"
            return self._create_examples(
                self._read_tsv(os.path.join(data_dir, "dev_mismatched.tsv")), "dev")

    def get_test_examples(self, data_dir, eval_set="MNLI-m"):
        """See base class."""
        if eval_set is None or eval_set == "MNLI-m":
            return self._create_examples(
                self._read_tsv(os.path.join(data_dir, "test_matched.tsv")), "test")
        elif eval_set == "MNLI-mm":
            return self._create_examples(
                self._read_tsv(os.path.join(data_dir, "test_mismatched.tsv")), "test")
        else:
            assert eval_set == "AX"
            return self._create_examples(
                self._read_tsv(os.path.join(data_dir, "ax.tsv")), "ax")

    def get_labels(self):
        """See base class."""
        return ["contradiction", "entailment", "neutral"]

    def _create_examples(self, lines, set_type):
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            if set_type == "ax":
                text_a = line[1]
                text_b = line[2]
                label = None
            else:
                text_a = line[8]
                text_b = line[9]
                label = None if set_type == "test" else line[-1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class ColaProcessor(DataProcessor):
    """
    Processor for the CoLA data set (GLUE version).
    
    ColaProcessor类继承自DataProcessor类,专门用于处理CoLA(Corpus of Linguistic Acceptability)数据集(特别是GLUE版本的CoLA)。CoLA数据集是一个用于评估语言模型在句子级可接受性判断任务上性能的数据集。它包含了一系列英语句子,每个句子都被标注为可接受(label为1)或不可接受(label为0)

    句子级可接受性判断任务: 
    这种任务要求系统或模型能够判断给定的句子在自然语言环境中是否被认为是合适的、可理解的或符合语言规范的。
    """

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        examples = []
        for (i, line) in enumerate(lines):
            if (i == 0) and (set_type == "test"):
                continue
            guid = "%s-%s" % (set_type, i)
            if set_type == 'test':
                text_a = line[1]
                label = None
            else:
                text_a = line[3]
                label = line[1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples


class Sst2Processor(DataProcessor):
    """Processor for the SST-2 data set (GLUE version).
    
    Sst2Processor类继承自DataProcessor类,专门用于处理SST-2(Stanford Sentiment Treebank version 2)数据集(特别是GLUE版本的SST-2)。SST-2是一个情感分类数据集,包含电影评论的句子及其对应的情感标签(正面或负面)。
    """

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            if set_type == "test":
                text_a = line[1]
                label = None
            else:
                text_a = line[0]
                label = line[1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples


class StsbProcessor(DataProcessor):
    """
    Processor for the STS-B data set (GLUE version).

    StsbProcessor类继承自DataProcessor类,专门用于处理STS-B(Semantic Textual Similarity Benchmark)数据集(特别是GLUE版本的STS-B)。
    
    STS-B是一个用于评估模型在理解两个句子之间语义相似度任务上的数据集。
    """

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_labels(self):
        """See base class."""
        return [None]

    def _create_examples(self, lines, set_type):
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[7]
            text_b = line[8]
            label = None if set_type == "test" else line[-1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class QqpProcessor(DataProcessor):
    """Processor for the QQP data set (GLUE version).
    
    QqpProcessor类继承自DataProcessor类,专门用于处理QQP(Quora Question Pairs)数据集(特别是GLUE版本的QQP)。QQP数据集包含来自Quora网站的问题对,以及一个标签,指示这些问题对是否是重复的。
    """

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            if set_type == "test":
                text_a = line[1]
                text_b = line[2]
                label = None
            else:
                try:
                    text_a = line[3]
                    text_b = line[4]
                    label = line[5]
                except IndexError:
                    continue
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class QnliProcessor(DataProcessor):
    """Processor for the STS-B data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_labels(self):
        """See base class."""
        return ["entailment", "not_entailment"]

    def _create_examples(self, lines, set_type):
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[1]
            text_b = line[2]
            label = None if set_type == "test" else line[-1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class RteProcessor(DataProcessor):
    """Processor for the RTE data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_labels(self):
        """See base class."""
        return ["entailment", "not_entailment"]

    def _create_examples(self, lines, set_type):
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[1]
            text_b = line[2]
            label = None if set_type == "test" else line[-1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class WnliProcessor(DataProcessor):
    """Processor for the WNLI data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[1]
            text_b = line[2]
            label = None if set_type == "test" else line[-1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


PROCESSORS = {
    "cola": ColaProcessor,
    "mnli": MnliProcessor,
    "mrpc": MrpcProcessor,
    "sst-2": Sst2Processor,
    "sts-b": StsbProcessor,
    "qqp": QqpProcessor,
    "qnli": QnliProcessor,
    "rte": RteProcessor,
    "wnli": WnliProcessor,
}

OUTPUT_MODES = {
    "cola": "classification",
    "mnli": "classification",
    "mrpc": "classification",
    "sst-2": "classification",
    "sts-b": "regression",
    "qqp": "classification",
    "qnli": "classification",
    "rte": "classification",
    "wnli": "classification",
}

EVAL_METRICS = {
    "cola": "mcc",
    "mnli": "acc",
    "mrpc": "acc_and_f1",
    "sst-2": "acc",
    "sts-b": "corr",
    "qqp": "acc_and_f1",
    "qnli": "acc",
    "rte": "acc",
    "wnli": "acc",
}


def convert_examples_to_features(examples, label_list, max_seq_length,
                                 tokenizer, output_mode):
    """Loads a data file into a list of `InputBatch`s."""

    if output_mode == 'classification':
        label_map = {label: i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))

        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[:(max_seq_length - 2)]

        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = [0] * len(tokens)

        if tokens_b:
            tokens += tokens_b + ["[SEP]"]
            segment_ids += [1] * (len(tokens_b) + 1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        if example.label is None:
            label_id = None
        else:
            if output_mode == "classification":
                label_id = label_map[example.label]
            elif output_mode == "regression":
                label_id = float(example.label)
            else:
                raise KeyError(output_mode)

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join(
                    [str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info(
                    "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            if example.label is None:
                logger.info("label: <UNK>")
            else:
                logger.info("label: %s (id = %d)" % (example.label, label_id))

        features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              label_id=label_id))
    return features


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def acc_and_f1(preds, labels):
    acc = simple_accuracy(preds, labels)
    f1 = f1_score(y_true=labels, y_pred=preds)
    return {
        "acc": acc,
        "f1": f1,
        "acc_and_f1": (acc + f1) / 2,
    }


def pearson_and_spearman(preds, labels):
    pearson_corr = pearsonr(preds, labels)[0]
    spearman_corr = spearmanr(preds, labels)[0]
    return {
        "pearson": pearson_corr,
        "spearmanr": spearman_corr,
        "corr": (pearson_corr + spearman_corr) / 2,
    }


def compute_metrics(task_name, preds, labels):
    assert len(preds) == len(labels)
    if task_name == "cola":
        return {"mcc": matthews_corrcoef(labels, preds)}
    elif task_name == "sst-2":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "mrpc":
        return acc_and_f1(preds, labels)
    elif task_name == "sts-b":
        return pearson_and_spearman(preds, labels)
    elif task_name == "qqp":
        return acc_and_f1(preds, labels)
    elif task_name == "mnli":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "qnli":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "rte":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "wnli":
        return {"acc": simple_accuracy(preds, labels)}
    else:
        raise KeyError(task_name)


def evaluate(task_name, model, device, eval_dataloader, eval_label_ids, num_labels):
    model.eval()
    eval_loss = 0
    nb_eval_steps = 0
    preds = []
    for eval_example in eval_dataloader:
        if len(eval_example) == 4:
            input_ids, input_mask, segment_ids, label_ids = eval_example
        else:
            input_ids, input_mask, segment_ids = eval_example
            label_ids = None

        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)

        with torch.no_grad():
            logits = model(input_ids, segment_ids, input_mask, labels=None)

        if label_ids is not None:
            label_ids = label_ids.to(device)
            if OUTPUT_MODES[task_name] == "classification":
                loss_fct = CrossEntropyLoss()
                tmp_eval_loss = loss_fct(logits.view(-1, num_labels), label_ids.view(-1))
            elif OUTPUT_MODES[task_name] == "regression":
                loss_fct = MSELoss()
                tmp_eval_loss = loss_fct(logits.view(-1), label_ids.view(-1))
            eval_loss += tmp_eval_loss.mean().item()

        nb_eval_steps += 1
        if len(preds) == 0:
            preds.append(logits.detach().cpu().numpy())
        else:
            preds[0] = np.append(
                preds[0], logits.detach().cpu().numpy(), axis=0)

    eval_loss = eval_loss / nb_eval_steps
    preds = preds[0]
    if OUTPUT_MODES[task_name] == "classification":
        preds = np.argmax(preds, axis=1)
    elif OUTPUT_MODES[task_name] == "regression":
        preds = np.squeeze(preds)

    if eval_label_ids is not None:
        result = compute_metrics(task_name, preds, eval_label_ids.numpy())
        result['eval_loss'] = eval_loss
    else:
        result = {}

    logger.info("***** Eval results *****")
    for key in sorted(result.keys()):
        logger.info("  %s = %s", key, str(result[key]))
    return preds, result


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    n_gpu = torch.cuda.device_count()

    logger.info("device: {} n_gpu: {}, 16-bits training: {}".format(
        device, n_gpu, args.fp16))

    # 梯度累积步骤参数验证 (gradient_accumulation_steps)
    """
    这段代码检查 gradient_accumulation_steps 是否小于 1。因为累积步骤的最小有效值应该是 1（表示不累积，直接进行每批次的反向传播和更新）。如果该值小于 1，程序会抛出一个 ValueError 异常，指出无效的参数
    """
    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                            args.gradient_accumulation_steps))

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    # 随机数种子设置: 目的是为了控制训练过程中的随机性，确保实验的可重复性
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    # 训练和评估参数验证: 目的是防止用户在调用脚本时忘记指定任务。如果既不训练也不评估，代码将没有任何实际操作
    if not args.do_train and not args.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    # 输出目录的准备
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir) # 检查指定的输出目录是否存在。如果不存在，它会使用 os.makedirs 创建该目录。
    if args.do_train:
        logger.addHandler(logging.FileHandler(os.path.join(args.output_dir, "train.log"), 'w'))
    else:
        logger.addHandler(logging.FileHandler(os.path.join(args.output_dir, "eval.log"), 'w'))
    logger.info(args)

    task_name = args.task_name.lower() # 从命令行参数 args.task_name 获取任务名称，并将其转换为小写

    # 任务存在性检查
    if task_name not in PROCESSORS:
        raise ValueError("Task not found: %s" % (task_name))

    # 初始化数据处理器 (processor)
    processor = PROCESSORS[task_name]()
    label_list = processor.get_labels()
    id2label = {i: label for i, label in enumerate(label_list)}
    num_labels = len(label_list)
    eval_metric = EVAL_METRICS[task_name]

    # 初始化分词器 (tokenizer)
    tokenizer = BertTokenizer.from_pretrained(args.model, do_lower_case=args.do_lower_case)

    # 准备开发集数据 (eval_examples 和 eval_features)
    if args.do_train or (not args.eval_test):
        if task_name == "mnli":
            # 从指定数据目录中加载开发集（验证集）数据
            # 如果任务是 "mnli"，开发集数据可能有多个版本（如 "matched" 和 "mismatched"），因此可能需要根据 args.eval_set 加载不同的子集
            eval_examples = processor.get_dev_examples(args.data_dir, eval_set=args.eval_set) 
        else:
            # 从指定数据目录中加载开发集（验证集）数据
            eval_examples = processor.get_dev_examples(args.data_dir)
        # 将开发集示例转换为模型输入所需的特征格式（例如，token IDs、注意力掩码等）
        eval_features = convert_examples_to_features(
            eval_examples, label_list, args.max_seq_length, tokenizer, OUTPUT_MODES[task_name])
        logger.info("***** Dev *****")
        logger.info("  Num examples = %d", len(eval_examples))
        logger.info("  Batch size = %d", args.eval_batch_size)
        # 将特征转换为张量 (Tensor)
        all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long) # 输入文本的 token IDs
        all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long) # 用于指示哪些 token 是实际输入（而不是填充）
        all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long) # 用于区分输入中的不同句子（例如，对于句子对任务）

        # 处理标签 (all_label_ids): 据任务的输出模式（分类或回归）处理标签数据
        if OUTPUT_MODES[task_name] == "classification": 
            # 将标签 ID 转换为 long 类型的 Tensor
            all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
        elif OUTPUT_MODES[task_name] == "regression":
            # 将标签 ID 转换为 float 类型的 Tensor
            all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.float)
            if args.fp16:
                # 如果启用了半精度训练（args.fp16），则将标签进一步转换为半精度浮点数
                all_label_ids = all_label_ids.half()

        # 创建数据集和数据加载器 (eval_data, eval_dataloader)
        eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
        eval_dataloader = DataLoader(eval_data, batch_size=args.eval_batch_size)
        eval_label_ids = all_label_ids

    # 加载训练数据 (train_examples)
    if args.do_train:
        train_examples = processor.get_train_examples(args.data_dir)
        # 转换训练特征 (train_features): 将训练样例转换为模型可以处理的特征格式
        train_features = convert_examples_to_features(
            train_examples, label_list, args.max_seq_length, tokenizer, OUTPUT_MODES[task_name])
        """
        args.max_seq_length: 指定输入序列的最大长度。
        tokenizer: 使用之前初始化的 BERT 分词器进行文本的分词和编码。
        OUTPUT_MODES[task_name]: 根据任务的输出模式（如分类或回归）决定特征转换的方式
        """
        # 训练数据排序或打乱 (train_features)
        if args.train_mode == 'sorted' or args.train_mode == 'random_sorted':
            # 根据 input_mask 的总和进行排序。input_mask 总和表示有效输入 token 的数量，排序的目的是为了更好地处理填充序列（padding sequence）。
            train_features = sorted(train_features, key=lambda f: np.sum(f.input_mask))
        else:
            # 随机打乱训练数据顺序，以确保训练过程中的样本顺序是随机的。
            random.shuffle(train_features)

        # 将训练特征转换为张量 (Tensor)
        all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)

        # 处理标签 (all_label_ids)
        if OUTPUT_MODES[task_name] == "classification":
            all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)
        elif OUTPUT_MODES[task_name] == "regression":
            all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.float)
            if args.fp16:
                all_label_ids = all_label_ids.half()

        # 创建数据集和数据加载器 (train_data, train_dataloader)
        train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
        train_dataloader = DataLoader(train_data, batch_size=args.train_batch_size, drop_last=True)
        train_batches = [batch for batch in train_dataloader]
        # 计算评估步数 (eval_step): 根据训练数据的批次数量和每个 epoch 进行评估的次数（args.eval_per_epoch），确定每多少个训练批次后进行一次模型评估
        eval_step = max(1, len(train_batches) // args.eval_per_epoch)

        # 计算训练优化步数 (num_train_optimization_steps): 决定训练过程中的优化次数。这个值与训练数据的批次数量、梯度累积步数（args.gradient_accumulation_steps），以及训练的 epoch 数（args.num_train_epochs）有关
        num_train_optimization_steps = \
            len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

        logger.info("***** Training *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_optimization_steps)

        # 初始化学习率列表 (lrs)
        best_result = None
        lrs = [args.learning_rate] if args.learning_rate else \
            [1e-6, 2e-6, 3e-6, 5e-6, 1e-5, 2e-5, 3e-5, 5e-5] # 一组学习率，用于在不同学习率下进行模型训练
        for lr in lrs:
            cache_dir = args.cache_dir if args.cache_dir else \
                PYTORCH_PRETRAINED_BERT_CACHE
            # 从预训练的 BERT 模型中加载 BertForSequenceClassification 模型，该模型已经针对分类任务进行了优化
            model = BertForSequenceClassification.from_pretrained(
                args.model, cache_dir=cache_dir, num_labels=num_labels) # 加载预训练模型 (BertForSequenceClassification)
            """
            args.model:指定要加载的预训练模型的名称或路径。
            cache_dir:指定缓存目录，存放下载的预训练模型文件。如果未指定，使用默认缓存目录 PYTORCH_PRETRAINED_BERT_CACHE。
            num_labels:设置模型的输出类别数，通常对应于任务的标签数。
            """

            # 处理半精度训练和多 GPU 设置
            if args.fp16:
                model.half()
            model.to(device)
            if n_gpu > 1:
                model = torch.nn.DataParallel(model)

            # Prepare optimizer
            """
            为训练过程准备优化器（optimizer），以便在训练过程中更新模型的参数。优化器在训练神经网络时至关重要，因为它决定了模型参数如何根据损失函数的梯度进行调整。
            """

            # model.named_parameters(): 获取模型中所有需要优化的参数，并将它们转换为一个包含 (name, parameter) 元组的列表。这些参数包括模型的权重、偏置项等
            param_optimizer = list(model.named_parameters())

            # 设置不进行权重衰减的参数 (no_decay)
            # weight decay：是一种正则化技术，用于防止模型过拟合。它通过在每次更新时对权重施加一个衰减因子，使权重逐渐减小。
            no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight'] # no_decay：定义了一组参数名称, 这些参数通常不需要进行权重衰减（weight decay）。

            # 分组优化器参数 (optimizer_grouped_parameters)
            optimizer_grouped_parameters = [
                # 包括模型中所有不在 no_decay 列表中的参数。对于这些参数，设置 weight_decay 为 0.01，即每次更新时都会施加权重衰减。
                {'params': [p for n, p in param_optimizer
                            if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
                # 包括模型中在 no_decay 列表中的参数（如 bias 和 LayerNorm 参数）。对于这些参数，不进行权重衰减（weight_decay = 0.0）
                {'params': [p for n, p in param_optimizer
                            if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
                ]
            
            # 处理半精度训练 (fp16)
            if args.fp16:
                try:
                    """
                    apex: 是由 NVIDIA 提供的一个库，用于混合精度训练和分布式训练。FP16_Optimizer 和 FusedAdam 是 apex 提供的优化器，专门用于半精度训练。
                    """
                    from apex.optimizers import FP16_Optimizer
                    from apex.optimizers import FusedAdam
                except ImportError:
                    raise ImportError("Please install apex from https://www.github.com/nvidia/apex"
                                      "to use distributed and fp16 training.")

                optimizer = FusedAdam(optimizer_grouped_parameters,
                                      lr=lr,
                                      bias_correction=False,
                                      max_grad_norm=1.0)
                if args.loss_scale == 0:
                    optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
                else:
                    optimizer = FP16_Optimizer(optimizer, static_loss_scale=args.loss_scale)

            else:
                optimizer = BertAdam(optimizer_grouped_parameters,
                                     lr=lr,
                                     warmup=args.warmup_proportion,
                                     t_total=num_train_optimization_steps)

            global_step = 0
            nb_tr_steps = 0
            nb_tr_examples = 0
            tr_loss = 0
            start_time = time.time()

            for epoch in range(int(args.num_train_epochs)):
                model.train()
                logger.info("Start epoch #{} (lr = {})...".format(epoch, lr))
                if args.train_mode == 'random' or args.train_mode == 'random_sorted':
                    random.shuffle(train_batches)
                for step, batch in enumerate(train_batches):
                    batch = tuple(t.to(device) for t in batch)
                    input_ids, input_mask, segment_ids, label_ids = batch
                    logits = model(input_ids, segment_ids, input_mask, labels=None)

                    if OUTPUT_MODES[task_name] == "classification":
                        loss_fct = CrossEntropyLoss()
                        loss = loss_fct(logits.view(-1, num_labels), label_ids.view(-1))
                    elif OUTPUT_MODES[task_name] == "regression":
                        loss_fct = MSELoss()
                        loss = loss_fct(logits.view(-1), label_ids.view(-1))

                    if n_gpu > 1:
                        loss = loss.mean()
                    if args.gradient_accumulation_steps > 1:
                        loss = loss / args.gradient_accumulation_steps

                    if args.fp16:
                        optimizer.backward(loss)
                    else:
                        loss.backward()

                    tr_loss += loss.item()
                    nb_tr_examples += input_ids.size(0)
                    nb_tr_steps += 1

                    if (step + 1) % args.gradient_accumulation_steps == 0:
                        if args.fp16:
                            lr_this_step = lr * \
                                warmup_linear(global_step/num_train_optimization_steps, args.warmup_proportion)
                            for param_group in optimizer.param_groups:
                                param_group['lr'] = lr_this_step
                        optimizer.step()
                        optimizer.zero_grad()
                        global_step += 1

                    if (step + 1) % eval_step == 0:
                        logger.info('Epoch: {}, Step: {} / {}, used_time = {:.2f}s, loss = {:.6f}'.format(
                                     epoch, step + 1, len(train_dataloader),
                                     time.time() - start_time, tr_loss / nb_tr_steps))
                        save_model = False
                        if args.do_eval:
                            preds, result = evaluate(task_name, model, device,
                                                     eval_dataloader, eval_label_ids, num_labels)
                            model.train()
                            result['global_step'] = global_step
                            result['epoch'] = epoch
                            result['learning_rate'] = lr
                            result['batch_size'] = args.train_batch_size
                            logger.info("First 20 predictions:")
                            for pred, label in zip(preds[:20], eval_label_ids.numpy()[:20]):
                                if OUTPUT_MODES[task_name] == 'classification':
                                    sign = u'\u2713' if pred == label else u'\u2718'
                                    logger.info("pred = %s, label = %s %s" % (id2label[pred], id2label[label], sign))
                                else:
                                    logger.info("pred = %.4f, label = %.4f" % (pred, label))
                            if (best_result is None) or (result[eval_metric] > best_result[eval_metric]):
                                best_result = result
                                save_model = True
                                logger.info("!!! Best dev %s (lr=%s, epoch=%d): %.2f" %
                                            (eval_metric, str(lr), epoch, result[eval_metric] * 100.0))
                        else:
                            save_model = True

                        if save_model:
                            model_to_save = model.module if hasattr(model, 'module') else model
                            output_model_file = os.path.join(args.output_dir, WEIGHTS_NAME)
                            output_config_file = os.path.join(args.output_dir, CONFIG_NAME)
                            torch.save(model_to_save.state_dict(), output_model_file)
                            model_to_save.config.to_json_file(output_config_file)
                            tokenizer.save_vocabulary(args.output_dir)
                            if best_result:
                                output_eval_file = os.path.join(args.output_dir, EVAL_FILE)
                                with open(output_eval_file, "w") as writer:
                                    for key in sorted(result.keys()):
                                        writer.write("%s = %s\n" % (key, str(result[key])))

    if args.do_eval:
        if args.eval_test:
            if task_name == "mnli":
                eval_examples = processor.get_test_examples(args.data_dir, eval_set=args.eval_set)
            else:
                eval_examples = processor.get_test_examples(args.data_dir)
            eval_features = convert_examples_to_features(
                eval_examples, label_list, args.max_seq_length, tokenizer, OUTPUT_MODES[task_name])
            logger.info("***** Test *****")
            logger.info("  Num examples = %d", len(eval_examples))
            logger.info("  Batch size = %d", args.eval_batch_size)
            all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
            all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
            all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
            eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids)
            eval_dataloader = DataLoader(eval_data, batch_size=args.eval_batch_size)
            eval_label_ids = None

        model = BertForSequenceClassification.from_pretrained(args.output_dir, num_labels=num_labels)
        if args.fp16:
            model.half()
        model.to(device)
        preds, result = evaluate(task_name, model, device, eval_dataloader, eval_label_ids, num_labels)
        pred_file = os.path.join(args.output_dir, PRED_FILE)
        with open(pred_file, "w") as f_out:
            f_out.write("index\tprediction\n")
            for i, pred in enumerate(preds):
                if OUTPUT_MODES[task_name] == 'classification':
                    f_out.write("%d\t%s\n" % (i, id2label[pred]))
                else:
                    f_out.write("%d\t%.6f\n" % (i, pred))
        output_eval_file = os.path.join(args.output_dir, TEST_FILE)
        with open(output_eval_file, "w") as writer:
            for key in sorted(result.keys()):
                writer.write("%s = %s\n" % (key, str(result[key])))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", default=None, type=str, required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--model", default=None, type=str, required=True)
    parser.add_argument("--task_name", default=None, type=str, required=True,
                        help="The name of the task to train.")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--eval_per_epoch", default=10, type=int,
                        help="How many times it evaluates on dev set per epoch")
    parser.add_argument("--max_seq_length", default=128, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_train", action='store_true', help="Whether to run training.")
    parser.add_argument("--train_mode", type=str, default='random_sorted', choices=['random', 'sorted', 'random_sorted'])
    parser.add_argument("--do_eval", action='store_true', help="Whether to run eval on the dev set.")
    parser.add_argument("--do_lower_case", action='store_true', help="Set this flag if you are using an uncased model.")
    parser.add_argument("--eval_test", action='store_true', help="Whether to eval on the test set.")
    parser.add_argument("--eval_set", type=str, default=None, help="Whether to evalu on the test set.")
    parser.add_argument("--train_batch_size", default=32, type=int, help="Total batch size for training.")
    parser.add_argument("--eval_batch_size", default=32, type=int, help="Total batch size for eval.")
    parser.add_argument("--learning_rate", default=None, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs", default=3.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion", default=0.1, type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--loss_scale', type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")
    args = parser.parse_args()

    main(args)
