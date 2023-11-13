# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import csv
import os
import logging
import argparse
import random
from tqdm import tqdm, trange
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from tokenization import BertTokenizer
from model import ARFN_Classification
from optimization import BertAdam
from torchvision import transforms
from PIL import Image
from sklearn.metrics import precision_recall_fscore_support
from torch.utils.tensorboard import SummaryWriter
import sys
import time


class Logger(object):
    def __init__(self, filename='default.log', stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'a',encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.terminal.flush()  # 不启动缓冲,实时输出
        self.log.flush()

    def flush(self):
        pass

def image_process(image_path, transform):
    image = Image.open(image_path).convert('RGB')
    image = transform(image)
    return image

def macro_f1(y_true, y_pred):
    preds = np.argmax(y_pred, axis=-1)
    true = y_true
    p_macro, r_macro, f_macro, support_macro \
      = precision_recall_fscore_support(true, preds, average='macro')
    return p_macro, r_macro, f_macro

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label

class MMInputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b, img_id, label=None):
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.img_id = img_id
        self.label = label

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id

class MMInputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, added_input_mask, segment_ids, s2_input_ids, s2_input_mask, \
                 s2_segment_ids, img_feat, label_id ):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.added_input_mask = added_input_mask
        self.segment_ids = segment_ids
        self.s2_input_ids = s2_input_ids
        self.s2_input_mask = s2_input_mask
        self.s2_segment_ids = s2_segment_ids
        self.img_feat = img_feat
        self.label_id = label_id

class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding='utf-8') as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines


class AbmsaProcessor(DataProcessor):
    """Processor for the MRPC data set (GLUE version)."""

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
        return ["0", "1", "2"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a = line[3].lower()
            text_b = line[4].lower()
            img_id = line[2]
            label = line[1]
            examples.append(
                MMInputExample(guid=guid, text_a=text_a, text_b=text_b, img_id=img_id, label=label))
        return examples

def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer):
    label_map = {label : i for i, label in enumerate(label_list)}
    features = []
    for (ex_index, example) in enumerate(examples):
        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[:(max_seq_length - 2)]
        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = [0] * len(tokens)

        if tokens_b:
            tokens += tokens_b + ["[SEP]"]
            segment_ids += [1] * (len(tokens_b) + 1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        label_id = label_map[example.label]
        if ex_index < 1:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join(
                    [str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info(
                    "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info("label: %s (id = %d)" % (example.label, label_id))

        features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              label_id=label_id))
    return features

def convert_mm_examples_to_features(examples, label_list, max_seq_length, max_entity_length, tokenizer, crop_size, path_img):
    label_map = {label : i for i, label in enumerate(label_list)}
    features = []
    count = 0
    a=0

    transform = transforms.Compose([
        transforms.RandomCrop(crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))])

    sent_length_a = 0
    entity_length_b = 0
    total_length = 0
    for (ex_index, example) in enumerate(examples):
        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = tokenizer.tokenize(example.text_b)
        if len(tokens_b) >= entity_length_b:
            entity_length_b = len(tokens_b)
        if len(tokens_a) >= sent_length_a:
            sent_length_a = len(tokens_a)

        if len(tokens_b) > max_entity_length - 2:
            s2_tokens = tokens_b[:(max_entity_length - 2)]
        else:
            s2_tokens = tokens_b
        s2_tokens = ["[CLS]"] + s2_tokens + ["[SEP]"]
        s2_segment_ids = [0] * len(s2_tokens)
        s2_input_ids = tokenizer.convert_tokens_to_ids(s2_tokens)
        s2_input_mask = [1] * len(s2_input_ids)

        # Zero-pad up to the sequence length.
        s2_padding = [0] * (max_entity_length - len(s2_input_ids))
        s2_input_ids += s2_padding
        s2_input_mask += s2_padding
        s2_segment_ids += s2_padding

        _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)

        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = [0] * len(tokens)

        if tokens_b:
            tokens += tokens_b + ["[SEP]"]
            segment_ids += [1] * (len(tokens_b) + 1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        if len(tokens) >= total_length:
            total_length = len(tokens)
        input_mask = [1] * len(input_ids)
        added_input_mask = [1] * (len(input_ids)+49) #1 or 49 is for encoding regional image representations

        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        added_input_mask += padding
        segment_ids += padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        label_id = label_map[example.label]
################################################################### "Image feature path"
        image_name = example.img_id
        res_path = r'./data_image'
        res_pt_name = image_name.replace('.jpg', '.pt')
        image_path = os.path.join(res_path, res_pt_name)
        image_path_orgfile = os.path.join(path_img, image_name)
##################################################################################################org
        if not os.path.exists(image_path_orgfile):
            print(image_path_orgfile)
        try:
            image = torch.load(image_path)
            image = image.squeeze(dim=1)
        except:
            count += 1
            print('image has problem!')
            image_path_fail = os.path.join(res_path, '17_06_4705.pt')
            image = torch.load(image_path_fail)
            image = image.squeeze(dim=1)

        if ex_index < 1:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join(
                    [str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info(
                    "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info("label: %s (id = %d)" % (example.label, label_id))

        features.append(
                MMInputFeatures(input_ids=input_ids, input_mask=input_mask, added_input_mask=added_input_mask,
                              segment_ids=segment_ids,
                              s2_input_ids=s2_input_ids, s2_input_mask=s2_input_mask, s2_segment_ids=s2_segment_ids,
                              img_feat = image,
                              label_id=label_id ))

    print('the number of problematic samples: ' + str(count))
    print('the max length of sentence a: '+str(sent_length_a+2) + ' entity b: '+str(entity_length_b+2) + \
          ' total length: '+str(total_length+3))
    return features

def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()

def accuracy(out, labels):
    outputs = np.argmax(out, axis=1)
    return np.sum(outputs == labels)

def warmup_linear(x, warmup=0.002):
    if x < warmup:
        return x/warmup
    return 1.0 - x

def main(epoch,best_accuracy):
    parser = argparse.ArgumentParser()
    # sys.stdout = Logger('./log.log', sys.stdout)
    # sys.stderr = Logger('./log.log', sys.stderr)
    ## parameters
    parser.add_argument("--data_dir",
                        default='./absa_data/twitter',
                        type=str,
                        required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--bert_model", default=None, type=str, required=True,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                        "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
                        "bert-base-multilingual-cased, bert-base-chinese.")
    parser.add_argument("--task_name",
                        default='twitter',
                        type=str,
                        required=True,
                        help="The name of the task to train.")
    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    ## Other parameters
    parser.add_argument("--max_seq_length",
                        default=64,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--max_entity_length",
                        default=16,
                        type=int,
                        help="The maximum entity input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_train",
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval",
                        action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--train_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=16,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=2,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.2,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--crop_size', type=int, default=224, help='crop size of image')
    parser.add_argument('--path_image', default='./data', help='path to images')
    parser.add_argument('--model_name', default='ARFN', help='model name')
    parser.add_argument('--pooling', default='first', help='pooling method') # first, cls, concat

    args = parser.parse_args()
    print("**************current model: "+args.model_name+"******************","pooling method: "+args.pooling+"******************")

    if args.task_name == "twitter":        # twitter-2017 dataset
        args.path_image = r"./data"
    elif args.task_name == "twitter2015":  # twitter-2015 dataset
        args.path_image = r"./data"
    else:
        print("The task name is not right!")

    processors = {
        "twitter2015": AbmsaProcessor,    # our twitter-2015 dataset
        "twitter": AbmsaProcessor         # our twitter-2017 dataset
    }

    num_labels_task = {
        "twitter2015": 3,                # our twitter-2015 dataset
        "twitter": 3                     # our twitter-2017 dataset
    }

    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    n_gpu = torch.cuda.device_count()
    logger.info("device: {} n_gpu: {}".format(device, n_gpu))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                            args.gradient_accumulation_steps))

    args.train_batch_size = int(args.train_batch_size / args.gradient_accumulation_steps)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if not args.do_train and not args.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train:
        raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))
    os.makedirs(args.output_dir, exist_ok=True)

    task_name = args.task_name.lower()

    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))

    processor = processors[task_name]()
    num_labels = num_labels_task[task_name]
    label_list = processor.get_labels()

    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)
    train_examples = None
    num_train_steps = None
    if args.do_train:
        train_examples = processor.get_train_examples(args.data_dir)
        num_train_steps = int(
            len(train_examples) / args.train_batch_size / args.gradient_accumulation_steps * args.num_train_epochs)

    # Prepare model
    model = ARFN_Classification.from_pretrained(args.bert_model, cache_dir='./cache',num_labels=num_labels,pooling=args.pooling)
    model.to(device)

    # Prepare optimizer
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    t_total = num_train_steps
    optimizer = BertAdam(optimizer_grouped_parameters,lr=args.learning_rate, warmup=args.warmup_proportion,t_total=t_total)

    global_step = 0
    nb_tr_steps = 0
    tr_loss = 0
    output_model_file = ''
    if args.do_eval and epoch>-1:
        output_model_file = os.path.join(args.output_dir, str(epoch)+".bin")
    else:
        output_model_file = os.path.join(args.output_dir, "pytorch_model.bin")
    if args.do_train:
        train_features = convert_mm_examples_to_features(
            train_examples, label_list, args.max_seq_length, args.max_entity_length, tokenizer, args.crop_size, args.path_image)

        all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
        all_added_input_mask = torch.tensor([f.added_input_mask for f in train_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
        all_s2_input_ids = torch.tensor([f.s2_input_ids for f in train_features], dtype=torch.long)
        all_s2_input_mask = torch.tensor([f.s2_input_mask for f in train_features], dtype=torch.long)
        all_s2_segment_ids = torch.tensor([f.s2_segment_ids for f in train_features], dtype=torch.long)
        all_img_feats = torch.stack([f.img_feat for f in train_features])
        all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)
        train_data = TensorDataset(all_input_ids, all_input_mask, all_added_input_mask, all_segment_ids,\
                                   all_s2_input_ids, all_s2_input_mask, all_s2_segment_ids,
                                   all_img_feats, all_label_ids)
        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)
        #'''
        eval_examples = processor.get_dev_examples(args.data_dir)
        eval_features = convert_mm_examples_to_features(
            eval_examples, label_list, args.max_seq_length, args.max_entity_length, tokenizer, args.crop_size, args.path_image)

        all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
        all_added_input_mask = torch.tensor([f.added_input_mask for f in eval_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
        all_s2_input_ids = torch.tensor([f.s2_input_ids for f in eval_features], dtype=torch.long)
        all_s2_input_mask = torch.tensor([f.s2_input_mask for f in eval_features], dtype=torch.long)
        all_s2_segment_ids = torch.tensor([f.s2_segment_ids for f in eval_features], dtype=torch.long)
        all_img_feats = torch.stack([f.img_feat for f in eval_features])
        all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
        eval_data = TensorDataset(all_input_ids, all_input_mask, all_added_input_mask, all_segment_ids, \
                                  all_s2_input_ids, all_s2_input_mask, all_s2_segment_ids,\
                                  all_img_feats, all_label_ids)
        # Run prediction for full data
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)
        max_acc = 0.0

        logger.info("*************** Running training ***************")
        writer = SummaryWriter(log_dir = args.output_dir+'/ARFN')

        for train_idx in trange(int(args.num_train_epochs), desc="Epoch"):

            logger.info("********** Epoch: "+ str(train_idx) + " **********")
            logger.info("  Num examples = %d", len(train_examples))
            logger.info("  Batch size = %d", args.train_batch_size)
            logger.info("  Num steps = %d", num_train_steps)
            model.train()

            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
                batch = tuple(t.to(device) for t in batch)
                input_ids, input_mask, added_input_mask, segment_ids, s2_input_ids, s2_input_mask, s2_segment_ids, \
                img_feats, label_ids = batch
                with torch.no_grad():
###############################resnet image feature
                    img_att = img_feats

                if train_idx == 0 and step == 0:
                    loss = model(input_ids, s2_input_ids, img_att, segment_ids, s2_segment_ids, input_mask, s2_input_mask, \
                             added_input_mask, label_ids, True)
                else:
                    loss = model(input_ids, s2_input_ids, img_att, segment_ids, s2_segment_ids, input_mask, s2_input_mask, \
                             added_input_mask, label_ids)
                if n_gpu > 1:
                    loss = loss.mean() # mean() to average on multi-gpu.
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                loss.backward()
                tr_loss += loss.item()
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    # modify learning rate with special warm up BERT uses
                    lr_this_step = args.learning_rate * warmup_linear(global_step/t_total, args.warmup_proportion)
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr_this_step
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1

            logger.info("***** Running evaluation on Dev Set*****")
            logger.info("  Num examples = %d", len(eval_examples))
            logger.info("  Batch size = %d", args.eval_batch_size)
            model.eval()

            eval_loss, eval_accuracy = 0, 0
            nb_eval_steps, nb_eval_examples = 0, 0

            true_label_list = []
            pred_label_list = []

            for input_ids, input_mask, added_input_mask, segment_ids,  s2_input_ids, s2_input_mask, s2_segment_ids, \
                img_feats, label_ids in tqdm(eval_dataloader, desc="Evaluating"):
                input_ids = input_ids.to(device)
                input_mask = input_mask.to(device)
                added_input_mask = added_input_mask.to(device)
                segment_ids = segment_ids.to(device)
                s2_input_ids = s2_input_ids.to(device)
                s2_input_mask = s2_input_mask.to(device)
                s2_segment_ids = s2_segment_ids.to(device)
                img_feats = img_feats.to(device)
                label_ids = label_ids.to(device)

                with torch.no_grad():

                    img_att = img_feats
                    tmp_eval_loss = model(input_ids, s2_input_ids, img_att, segment_ids, s2_segment_ids,
                                          input_mask, s2_input_mask, added_input_mask, label_ids)
                    logits = model(input_ids, s2_input_ids, img_att, segment_ids, s2_segment_ids, input_mask,
                                   s2_input_mask, added_input_mask)

                logits = logits.detach().cpu().numpy()
                label_ids = label_ids.to('cpu').numpy()
                true_label_list.append(label_ids)
                pred_label_list.append(logits)
                tmp_eval_accuracy = accuracy(logits, label_ids)

                eval_loss += tmp_eval_loss.mean().item()
                eval_accuracy += tmp_eval_accuracy

                nb_eval_examples += input_ids.size(0)
                nb_eval_steps += 1

            eval_loss = eval_loss / nb_eval_steps
            eval_accuracy = eval_accuracy / nb_eval_examples
            loss = tr_loss / nb_tr_steps if args.do_train else None
            true_label = np.concatenate(true_label_list)
            pred_outputs = np.concatenate(pred_label_list)
            precision, recall, F_score = macro_f1(true_label, pred_outputs)
            result = {'eval_loss': eval_loss,
                      'eval_accuracy': eval_accuracy,
                      'f_score': F_score,
                      'global_step': global_step,
                      'loss': loss}
            localtime = time.time()
            writer.add_scalar('loss', loss,train_idx,localtime)
            writer.add_scalar('f_score', F_score,train_idx,localtime)
            writer.add_scalar('eval_loss', eval_loss,train_idx,localtime)
            writer.add_scalar('eval_accuracy', eval_accuracy,train_idx,localtime)

            logger.info("***** Dev Eval results *****")
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(result[key]))

            model_to_save1 = model.module if hasattr(model, 'module') else model  # Only save the model it-self
            output_model_index = os.path.join(args.output_dir, str(train_idx) +".bin")
            torch.save(model_to_save1.state_dict(), output_model_index)

            if eval_accuracy >= max_acc:
                # Save a trained model
                model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self

                if args.do_train:
                    t = os.path.join(args.output_dir,  'pytorch_model_'+str(train_idx)+'.bin')
                    torch.save(model_to_save.state_dict(), t )
                    torch.save(model_to_save.state_dict(), output_model_file)

                max_acc = eval_accuracy
        writer.close()

    model_state_dict = torch.load(output_model_file)
    model = ARFN_Classification.from_pretrained(args.bert_model,state_dict=model_state_dict,num_labels=num_labels,pooling=args.pooling)
    model.to(device)

    if args.do_eval:
        eval_examples = processor.get_test_examples(args.data_dir)
        eval_features = convert_mm_examples_to_features(
            eval_examples, label_list, args.max_seq_length, args.max_entity_length, tokenizer, args.crop_size,
            args.path_image)
        logger.info("***** Running evaluation on Test Set*****")
        logger.info("  Num examples = %d", len(eval_examples))
        logger.info("  Batch size = %d", args.eval_batch_size)
        all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
        all_added_input_mask = torch.tensor([f.added_input_mask for f in eval_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
        all_s2_input_ids = torch.tensor([f.s2_input_ids for f in eval_features], dtype=torch.long)
        all_s2_input_mask = torch.tensor([f.s2_input_mask for f in eval_features], dtype=torch.long)
        all_s2_segment_ids = torch.tensor([f.s2_segment_ids for f in eval_features], dtype=torch.long)
        all_img_feats = torch.stack([f.img_feat for f in eval_features])
        all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
        eval_data = TensorDataset(all_input_ids, all_input_mask, all_added_input_mask, all_segment_ids, \
                                  all_s2_input_ids, all_s2_input_mask, all_s2_segment_ids,
                                  all_img_feats, all_label_ids)
        # Run prediction for full data
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

        model.eval()

        eval_loss, eval_accuracy = 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0

        true_label_list = []
        pred_label_list = []

        for input_ids, input_mask, added_input_mask, segment_ids, s2_input_ids, s2_input_mask, s2_segment_ids, \
            img_feats, label_ids in tqdm(eval_dataloader, desc="Evaluating"):
            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)
            added_input_mask = added_input_mask.to(device)
            segment_ids = segment_ids.to(device)
            s2_input_ids = s2_input_ids.to(device)
            s2_input_mask = s2_input_mask.to(device)
            s2_segment_ids = s2_segment_ids.to(device)
            img_feats = img_feats.to(device)
            label_ids = label_ids.to(device)

            with torch.no_grad():

                img_att = img_feats
                tmp_eval_loss = model(input_ids, s2_input_ids, img_att, segment_ids, s2_segment_ids,
                                      input_mask, s2_input_mask, added_input_mask, label_ids)
                logits = model(input_ids, s2_input_ids, img_att, segment_ids, s2_segment_ids, input_mask,
                               s2_input_mask, added_input_mask)

            logits = logits.detach().cpu().numpy()
            label_ids = label_ids.to('cpu').numpy()
            true_label_list.append(label_ids)
            pred_label_list.append(logits)
            tmp_eval_accuracy = accuracy(logits, label_ids)

            eval_loss += tmp_eval_loss.mean().item()
            eval_accuracy += tmp_eval_accuracy

            nb_eval_examples += input_ids.size(0)
            nb_eval_steps += 1

        eval_loss = eval_loss / nb_eval_steps
        eval_accuracy = eval_accuracy / nb_eval_examples
        loss = tr_loss / nb_tr_steps if args.do_train else None
        true_label = np.concatenate(true_label_list)
        pred_outputs = np.concatenate(pred_label_list)
        precision, recall, F_score = macro_f1(true_label, pred_outputs)
        result = {'eval_loss': eval_loss,
                  'eval_accuracy': eval_accuracy,
                  'precision': precision,
                  'recall': recall,
                  'f_score': F_score,
                  'global_step': global_step,
                  'loss': loss}

        pred_label = np.argmax(pred_outputs, axis=-1)
        fout_p = ''
        fout_t = ''
        if epoch>-1:
            fout_p = open(os.path.join(args.output_dir, str(epoch)+"pred.txt"), 'w')
            fout_t = open(os.path.join(args.output_dir, str(epoch)+"true.txt"), 'w')
        else:
            fout_p = open(os.path.join(args.output_dir, "pred.txt"), 'w')
            fout_t = open(os.path.join(args.output_dir, "true.txt"), 'w')

        for i in range(len(pred_label)):
            attstr = str(pred_label[i])
            fout_p.write(attstr + '\n')
        for i in range(len(true_label)):
            attstr = str(true_label[i])
            fout_t.write(attstr + '\n')

        fout_p.close()
        fout_t.close()

        output_eval_file = ''
        if epoch > -1:
            output_eval_file = os.path.join(args.output_dir, str(epoch)+"eval_results.txt")
        else:
            output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            logger.info("***** Test Eval results *****")
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))

        if best_accuracy<eval_accuracy:
            best_accuracy = eval_accuracy

            output_eval_file = os.path.join(args.output_dir, 'best_' + str(epoch) + "eval_results.txt")
            with open(output_eval_file, "w") as writer:
                logger.info("***** Test Eval results *****")
                for key in sorted(result.keys()):
                    logger.info("  %s = %s", key, str(result[key]))
                    writer.write("%s = %s\n" % (key, str(result[key])))

        return best_accuracy

if __name__ == "__main__":
    best_accuracy = 0
    ##train
    best_accuracy = main(-1, best_accuracy)

    ##test

    # best_accuracy = main(20,best_accuracy)
    # print(best_accuracy)

