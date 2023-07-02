import json
import os

entity_dict_path = '../data/dicts/entity_dict.txt'
entity_vocab = ['O']
relate_dict_path = '../data/dicts/relate_dict.txt'
relate_vocab = []
mark_mode = ['B', 'I']
# for file_index in range(train_file_list_index, train_file_clip_num):
with open("../data/step1_train.json", 'r', encoding='utf-8') as fp:
    for line in fp.readlines():
        line_json = json.loads(line)
        entityMentions = line_json['entityMentions']
        relationMentions = line_json['relationMentions']
        for (key, item) in enumerate(entityMentions):
            if ('B' + item['label'] not in entity_vocab) or ('I' + item['label'] not in entity_vocab):
                for i in mark_mode:
                    entity_vocab.append(i + item['label'])
        for (key, item) in enumerate(relationMentions):
            if item['label'] not in relate_vocab:
                relate_vocab.append(item['label'])

if not os.path.exists('../data/dicts/'):
    os.mkdir('../data/dicts/')

if not os.path.exists(entity_dict_path):
    with open(entity_dict_path, 'w', encoding='utf-8') as f:
        for i in entity_vocab:
            f.write(i + '\n')

if not os.path.exists(relate_dict_path):
    with open(relate_dict_path, 'w', encoding='utf-8') as f:
        for i in relate_vocab:
            f.write(i + '\n')
