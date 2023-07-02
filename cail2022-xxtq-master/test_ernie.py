import json
import os

import paddle
from functools import partial
import paddle.nn.functional as F
from paddlenlp.datasets import MapDataset
from paddlenlp.data import Stack, Tuple, Pad
from utils import load_dict
from paddlenlp.transformers import ErnieTokenizer, ErnieForTokenClassification
from utils import convert_example,parse_probs_mark
max_seq = 1000
originals = []
def load_dataset(datafiles):
    def read(data_path):
        with open(data_path, 'r', encoding='utf-8') as fp:
            for line in fp.readlines():
                words = line
                original = ['[CLS]'] + [i for i in words] + ['[SEP]']
                originals.append(original)
                yield words

    if isinstance(datafiles, str):
        return MapDataset(list(read(datafiles)))
    elif isinstance(datafiles, list) or isinstance(datafiles, tuple):
        return [MapDataset(list(read(datafile))) for datafile in datafiles]

test_ds = load_dataset(datafiles=(
        './data/step1_test.json'))



label_vocab = load_dict('./data/dicts/entity_dict.txt')
MODEL_NAME = "ernie-1.0"
tokenizer_file = "./save/params/ernie/vocab.txt"
# tokenizer = ErnieTokenizer.from_pretrained(MODEL_NAME)
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

model_dir = "./save/params/ernie/mark/"
model = ErnieForTokenClassification.from_pretrained(model_dir)

step = 0
lines = []
index_to_label_vocab = [item for (key,item) in enumerate(label_vocab)]
for idx, (input_ids, token_type_ids, length) in enumerate(test_loader):
    logits = model(input_ids, token_type_ids)
    probs = F.sigmoid(logits)
    context =  [tokenizer.vocab.to_tokens(i) for i in paddle.tolist(input_ids)]
    line = parse_probs_mark(probs,originals[idx*BATCH_SIZE:(idx+1)*BATCH_SIZE],index_to_label_vocab)
    lines.append(line)
    step += 1
    if step%100 == 0:
        print("step : ",step)

if not os.path.exists('./save/result/ernie'):
    os.mkdir('./save/result/ernie')

if  os.path.exists('./save/result/ernie'):
    with open('data/false.json', 'w', encoding='utf-8') as f:
        for batch_line in lines:
            for line in batch_line:
                f.write(json.dumps(line,ensure_ascii=False) + '\n')



