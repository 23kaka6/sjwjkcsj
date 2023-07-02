import json
import os
max_len = 0
# for file_index in range(train_file_list_index, train_file_clip_num):
with open("../data/step2_train.json", 'r', encoding='utf-8') as fp:
    for line in fp.readlines():
        line_json = json.loads(line)
        if len(line_json['sentText']) > max_len:
            max_len = len(line_json['sentText'])

with open("../data/step2_test.json", 'r', encoding='utf-8') as fp:
    for line in fp.readlines():
        if len(line) > max_len:
            max_len = len(line)

print('max_len',max_len)