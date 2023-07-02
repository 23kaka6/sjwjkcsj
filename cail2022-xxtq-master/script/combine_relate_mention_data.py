import json
import os

save_dir = '../data/'
save_train_path = save_dir + 'relation_mentions_test.json'
data_line = []
with open("../data/relate_false.json", 'r', encoding='utf-8') as fp1 , open("../data/relation_test.json", 'r', encoding='utf-8') as fp2:
    relation_test_lines = fp2.readlines()
    relate_false_lines = fp1.readlines()
    false_step = 0
    data_line = len(relate_false_lines) if len(relate_false_lines) > len(relation_test_lines) else len(relation_test_lines)
    data_line = [0] * data_line
    while false_step < len(relate_false_lines) and false_step < len(relation_test_lines):
        relate_false_line = json.loads(relate_false_lines[false_step])
        relation_test_line = json.loads(relation_test_lines[false_step])
        if relate_false_line["index"] == 9628:
            print('stop',false_step)
        relationMention = {
            "em1Text": relation_test_line["em1Text"],
            "em2Text": relation_test_line["em2Text"],
            "e1start": relation_test_line["e1start"],
            "e21start": relation_test_line["e21start"],
            "label": relate_false_line["label"]
        }
        if not data_line[relate_false_line["index"]]:
            data_line[relate_false_line["index"]] = []
            data_line[relate_false_line["index"]].append(relationMention)
        else:
            data_line[relate_false_line["index"]].append(relationMention)
        false_step += 1
step = 0
if  os.path.exists(save_dir):
    with open(save_train_path, 'w', encoding='utf-8') as f:
        for line in data_line:
            # if line:
            if step < 9999 + 1:
                f.write(json.dumps(line, ensure_ascii=False) + '\n')
            step += 1
