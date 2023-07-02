from random import random

import numpy as np
import paddle
import paddle.nn.functional as F
from paddlenlp.data import Stack, Tuple, Pad
import itertools


def load_dict(dict_path):
    vocab = {}
    i = 0
    for line in open(dict_path, 'r', encoding='utf-8'):
        key = line.strip('\n')
        vocab[key] = i
        i += 1
    return vocab


def parse_probs_mark(probs, contexts, label_vocab):
    return_result = []
    class_lists = paddle.tolist(paddle.argmax(probs, axis=2))
    idx = -1
    for class_list in class_lists:
        idx += 1
        class_list =  class_list[1:-1]
        entityMentions = []
        articleId = int(random() * 10000)
        sentId = int(random() * 10000)
        # 去掉结束符和换行符
        sentText = ''.join(contexts[idx][1:-2])
        for key in range(0, len(sentText)):
            item = class_list[key]
            if (label_vocab[item][0] == 'B') and label_vocab[item] != 'O':
                start = key
                end = key + 1
                Mentions = {}
                while (end < len(sentText) and (
                        label_vocab[class_list[end]][-2:] == label_vocab[class_list[start]][-2:] and
                        label_vocab[class_list[end]][0] != 'B')):
                    end += 1
                Mentions['start'] = start
                Mentions['end'] = end
                Mentions['text'] = "".join(sentText[start:end])
                Mentions['label'] = label_vocab[item][-2:]
                if Mentions['text'] != "":
                    entityMentions.append(Mentions)
        return_result.append(
            {"articleId": articleId, "sentId": sentId, "entityMentions": entityMentions, "sentText": sentText})
    return return_result

def convert_example_for_relate(example, tokenizer, vocab):
    if isinstance(example, str):
        tokens = example
        tokenized_input = tokenizer(
            tokens, return_length=True, is_split_into_words=True)
        return tokenized_input['input_ids'], tokenized_input[
            'token_type_ids'], 1
    else:
        tokens, labels = example
        tokenized_input = tokenizer(
                tokens, return_length=True, is_split_into_words=True)
        tokenized_input['labels'] = labels
        tokenized_input['labels'] = vocab[labels]
        return tokenized_input['input_ids'], tokenized_input[
                'token_type_ids'], 1, tokenized_input['labels']

def convert_examples(examples, tokenizer, vocab):
    if isinstance(examples, list):
        # 加入了分隔符的一句话
        tokens = []
        for (key,example) in enumerate(examples):
            if key != 0:
                tokens += ['[CLS]'] + [i for i in example]
            else:
                tokens += [i for i in example]
        tokenized_input = tokenizer(
            tokens, return_length=True, is_split_into_words=True)
        return tokenized_input['input_ids'], tokenized_input[
            'token_type_ids'], tokenized_input['seq_len']

def convert_example(example, tokenizer, vocab):
    if isinstance(example, str):
        tokens = example
        tokens = [i for i in tokens]
        tokenized_input = tokenizer(
            tokens, return_length=True, is_split_into_words=True)
        return tokenized_input['input_ids'], tokenized_input[
            'token_type_ids'], tokenized_input['seq_len']
    else:
        tokens, labels = example
        tokenized_input = tokenizer(
            tokens, return_length=True, is_split_into_words=True)
        no_entity_id = 'O'
        # Token '[CLS]' and '[SEP]' will get label 'O'
        # labels =  ['UNK'] + labels + ['UNK']
        # 保证label与input_ids长度一致
        # -2 for [CLS] and [SEP]
        # if len(tokenized_input['input_ids']) - 2 < len(labels):
        labels = [no_entity_id] + labels + [no_entity_id]
        tokenized_input['labels'] = labels
        tokenized_input['labels'] = [vocab[x] for x in tokenized_input['labels']]
        return tokenized_input['input_ids'], tokenized_input[
            'token_type_ids'], tokenized_input['seq_len'], tokenized_input['labels']


@paddle.no_grad()
def evaluate(model, metric, data_loader):
    model.eval()
    metric.reset()
    for input_ids, seg_ids, lens, labels in data_loader:
        logits = model(input_ids, seg_ids)
        preds = paddle.argmax(logits, axis=-1)
        if preds.ndim == 1:
            preds = paddle.unsqueeze(preds,axis=-1)
        if labels.ndim == 1:
            labels = paddle.unsqueeze(labels, axis=-1)
        n_infer, n_label, n_correct = metric.compute(None, lens, preds, labels)
        metric.update(n_infer.numpy(), n_label.numpy(), n_correct.numpy())
        precision, recall, f1_score = metric.accumulate()
    print("eval precision: %f - recall: %f - f1: %f" %
          (precision, recall, f1_score))


def predict(model, data_loader, ds, label_vocab):
    pred_list = []
    len_list = []
    for input_ids, seg_ids, lens, labels in data_loader:
        logits = model(input_ids, seg_ids)
        pred = paddle.argmax(logits, axis=-1)
        pred_list.append(pred.numpy())
        len_list.append(lens.numpy())
    preds = parse_decodes2(ds, pred_list, len_list, label_vocab)
    return preds


def parse_entityMentions(entityMentions, context_len):
    result = [0] * context_len;
    for (key, item) in enumerate(entityMentions):
        result[item['start']] = 'B' + item['label']
        for index in range(item['start'] + 1, item['end']):
            result[index] = 'I' + item['label']
    for i in range(0, len(result)):
        if result[i] == 0:
            result[i] = 'O'
    return result


def parse_relationMentions(entityMentions, relationMentions, words):
    result = []
    # 所有成对组合 list(itertools.combinations(x, 2))
    start_to_end = dict()
    for (key, entityMention) in enumerate(entityMentions):
        start = entityMention["start"]
        end = entityMention["end"]
        start_to_end[start] = end
    for (key, relationMention) in enumerate(relationMentions):
        e1start = relationMention["e1start"]
        e21start = relationMention["e21start"]
        label = relationMention["label"]
        e1end = start_to_end[e1start]
        e21end = start_to_end[e21start]
        start = e21start if e1start > e21start else e1start
        end = e21end if e1end < e21end else e1end
        sub_text = words[start:end]
        result.append({
            "words": sub_text,
            "labels": label
        })
    return result


def parse_decodes1(ds, decodes, lens, label_vocab):
    decodes = [x for batch in decodes for x in batch]
    lens = [x for batch in lens for x in batch]
    id_label = dict(zip(label_vocab.values(), label_vocab.keys()))
    
    outputs = []
    for idx, end in enumerate(lens):
        sent = ds.data[idx][0][:end]
        tags = [id_label[x] for x in decodes[idx][:end]]
        sent_out = []
        tags_out = []
        words = ""
        for s, t in zip(sent, tags):
            if t.endswith('-B') or t == 'O':
                if len(words):
                    sent_out.append(words)
                tags_out.append(t.split('-')[0])
                words = s
            else:
                words += s
        if len(sent_out) < len(tags_out):
            sent_out.append(words)
        outputs.append(''.join(
            [str((s, t)) for s, t in zip(sent_out, tags_out)]))
    return outputs


def parse_decodes2(ds, decodes, lens, label_vocab):
    decodes = [x for batch in decodes for x in batch]
    lens = [x for batch in lens for x in batch]
    id_label = dict(zip(label_vocab.values(), label_vocab.keys()))

    outputs = []
    for idx, end in enumerate(lens):
        sent = ds.data[idx][0][:end]
        tags = [id_label[x] for x in decodes[idx][1:end]]
        sent_out = []
        tags_out = []
        words = ""
        for s, t in zip(sent, tags):
            if t.endswith('-B') or t == 'O':
                if len(words):
                    sent_out.append(words)
                tags_out.append(t.split('-')[0])
                words = s
            else:
                words += s
        if len(sent_out) < len(tags_out):
            sent_out.append(words)
        outputs.append(''.join(
            [str((s, t)) for s, t in zip(sent_out, tags_out)]))
    return outputs
