import itertools
import json
import os
#
# save_dir = './data/'
# save_train_path = save_dir + 'relation_test.json'
# sub_texts = []
false_step = -1

with open("../data/relate_false.json", 'r', encoding='utf-8') as fp1 , open("../data/false.json", 'r', encoding='utf-8') as fp2:
    for false_line in fp2.readlines():
        false_step += 1
        false_line = json.loads(false_line)
        for relate_false_line in fp1.readlines():
            relate_false_line = json.loads(relate_false_line)
            # if relate_false_line['index'] == false_step:






# if  os.path.exists(save_dir):
#     with open(save_train_path, 'w', encoding='utf-8') as f:
#         for line in sub_texts:
#             f.write(json.dumps(line, ensure_ascii=False) + '\n')