import json
import os

save_dir = '../data/'
save_train_path = save_dir + 'false.json'
data_line = []
with open("../data/false.json", 'r', encoding='utf-8') as fp1 , open("../data/relation_mentions_test.json", 'r', encoding='utf-8') as fp2:
    relation_mentions_test_lines = fp2.readlines()
    false_lines = fp1.readlines()
    false_step = 0
    while false_step < len(false_lines) and false_step < len(relation_mentions_test_lines):
        relate_false_line = json.loads(false_lines[false_step])
        relation_test_line = json.loads(relation_mentions_test_lines[false_step])
        if relation_test_line:
            relate_false_line['relationMentions'] = relation_test_line
        data_line.append(relate_false_line)
        false_step += 1
if  os.path.exists(save_dir):
    with open(save_train_path, 'w', encoding='utf-8') as f:
        for line in data_line:
            # if line:
            f.write(json.dumps(line, ensure_ascii=False) + '\n')