import json

import paddle
from functools import partial
from paddlenlp.datasets import MapDataset
from paddlenlp.data import Stack, Tuple, Pad
from paddlenlp.metrics import ChunkEvaluator
from utils import load_dict, evaluate
from paddle.io import DataLoader, BatchSampler
from paddlenlp.transformers import ErnieTokenizer, ErnieForSequenceClassification
from utils import convert_example_for_relate

max_seq = 1000


def load_dataset(datafiles):
    def read(data_path):
        with open(data_path, 'r', encoding='utf-8') as fp:
            for line in fp.readlines():
                line_json = json.loads(line)
                words = line_json['words']
                labels = line_json["labels"]
                words = [i for i in words]
                yield words, labels

    if isinstance(datafiles, str):
        return MapDataset(list(read(datafiles)))
    elif isinstance(datafiles, list) or isinstance(datafiles, tuple):
        return [MapDataset(list(read(datafile))) for datafile in datafiles]


train_ds, dev_ds = load_dataset(datafiles=(
    'cail2022-xxtq-master/data/step2_train.json', 'cail2022-xxtq-master/data/step2_train.json'))

relate_label_vocab = load_dict('cail2022-xxtq-master/data/relations.txt')
MODEL_NAME = "ernie-1.0"
tokenizer_file = "./save/params/ernie/vocab.txt"
tokenizer = ErnieTokenizer(tokenizer_file)
trans_func = partial(convert_example_for_relate, tokenizer=tokenizer, vocab=relate_label_vocab)
trans_func = partial(convert_example_for_relate, tokenizer=tokenizer, vocab=relate_label_vocab)

train_ds.map(trans_func)
dev_ds.map(trans_func)

ignore_label = -1
batchify_fn = lambda samples, fn=Tuple(
    Pad(axis=0, pad_val=tokenizer.pad_token_id),  # input_ids
    Pad(axis=0, pad_val=tokenizer.pad_token_type_id),  # token_type_ids
    Stack(),  # seq_len
    Pad(axis=0, pad_val=ignore_label)  # labels
): fn(samples)
train_batch_sampler = BatchSampler(train_ds, batch_size=1, shuffle=True)
dev_batch_sampler = BatchSampler(train_ds, batch_size=2, shuffle=False)
train_loader = DataLoader(
    dataset=train_ds,
    batch_sampler=train_batch_sampler,
    return_list=True,
    collate_fn=batchify_fn)
dev_loader = DataLoader(
    dataset=dev_ds,
    batch_sampler=dev_batch_sampler,
    return_list=True,
    collate_fn=batchify_fn)

mark_seq_model = ErnieForSequenceClassification.from_pretrained(MODEL_NAME, num_classes=len(relate_label_vocab))

mark_metric = ChunkEvaluator(label_list=relate_label_vocab.keys(), suffix=True)
mark_loss_fn = paddle.nn.loss.CrossEntropyLoss(ignore_index=ignore_label)
mark_optimizer = paddle.optimizer.AdamW(learning_rate=2e-4, parameters=mark_seq_model.parameters())

step = 0
for epoch in range(1):
    mark_seq_model.train()
    for idx, (input_ids, token_type_ids, length, labels) in enumerate(train_loader):
        logits = mark_seq_model(input_ids, token_type_ids)
        loss = paddle.mean(mark_loss_fn(logits, labels))
        loss.backward()
        mark_optimizer.step()
        mark_optimizer.clear_grad()
        evaluate(mark_seq_model, mark_metric, dev_loader)
        step += 1
        if step % 50 == 0:
            print("epoch:%d - step:%d - loss: %f" % (epoch, step, loss))
    # evaluate(mark_seq_model, mark_metric, dev_loader)

mark_seq_model.save_pretrained('./checkpoint/relate/')
print("执行完成,关系提取")
