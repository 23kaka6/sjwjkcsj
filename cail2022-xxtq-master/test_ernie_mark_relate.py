import itertools
import json
import os

import paddle
from functools import partial
import paddle.nn.functional as F
from paddlenlp.datasets import MapDataset
from paddlenlp.data import Stack, Tuple, Pad
from utils import load_dict
from paddlenlp.transformers import ErnieTokenizer, ErnieForSequenceClassification
from utils import convert_example,parse_probs_mark
max_seq = 1000
originals = []
steps = []
def load_dataset(datafiles):
    def read(data_path):
        with open(data_path, 'r', encoding='utf-8') as fp:
            for line in fp.readlines():
                line = json.loads(line)
                step = line['step']
                words = line['text']
                # 去掉换行符
                words = words[:-1]
                steps.append(step)
                yield words

    if isinstance(datafiles, str):
        return MapDataset(list(read(datafiles)))
    elif isinstance(datafiles, list) or isinstance(datafiles, tuple):
        return [MapDataset(list(read(datafile))) for datafile in datafiles]

test_ds = load_dataset(datafiles=(
        './data/relation_test.json'))

label_vocab = load_dict('./data/dicts/relate_dict.txt')

MODEL_NAME = "ernie-1.0"
tokenizer_file = "./save/params/ernie/vocab.txt"
tokenizer = ErnieTokenizer(tokenizer_file)

trans_func = partial(convert_example, tokenizer=tokenizer, vocab=label_vocab)
test_ds.map(trans_func)

ignore_label = -1
batchify_fn = lambda samples, fn=Tuple(
    Pad(axis=0, pad_val=tokenizer.pad_token_id),  # input_ids
    Pad(axis=0, pad_val=tokenizer.pad_token_type_id),  # token_type_ids
    Stack(),  # seq_len
): fn(samples)

BATCH_SIZE = 10
test_loader = paddle.io.DataLoader(
    dataset=test_ds,
    batch_size=BATCH_SIZE,
    return_list=True,
    collate_fn=batchify_fn)

mark_model_dir = "./save/params/ernie/relate/"
model = ErnieForSequenceClassification.from_pretrained(mark_model_dir)

index_to_label = [item for (key,item) in enumerate(label_vocab)]
step = 0
lines = []
relate_lines = []
index_to_label_vocab = [item for (key,item) in enumerate(label_vocab)]
for idx, (input_ids, token_type_ids, length) in enumerate(test_loader):
    logits = model(input_ids, token_type_ids)
    classfiy = paddle.argmax(F.sigmoid(logits),axis=1)
    max_nums = paddle.amax(F.sigmoid(logits),axis=1)
    classfiy_list = paddle.tolist(classfiy)
    labels = []
    for (key,item) in enumerate(paddle.tolist(max_nums)):
        if item > 0.7:
            labels.append(index_to_label[classfiy_list[key]])
        else:
            # 低于的不打标签，舍弃
            labels.append(0)
    input_orders = []
    for i in range(0,BATCH_SIZE):
        # input_orders.append(steps[idx * BATCH_SIZE + BATCH_SIZE])
        if idx * BATCH_SIZE + i < len(steps):
            relate_lines.append({
                "index":steps[idx * BATCH_SIZE + i],
                "label":labels[i]
            })
    step += 1
    print("step : ", step)



save_dir = './data/'
save_train_path = save_dir + 'relate_false.json'
if  os.path.exists(save_dir):
    with open(save_train_path, 'w', encoding='utf-8') as f:
        for line in relate_lines:
            f.write(json.dumps(line, ensure_ascii=False) + '\n')
