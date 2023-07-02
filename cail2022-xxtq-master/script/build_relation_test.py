import itertools
import json
import os

save_dir = './data/'
save_train_path = save_dir + 'relation_test.json'
sub_texts = []
step = -1
# for file_index in range(train_file_list_index, train_file_clip_num):
with open("./checkpoint/ernie/false.json", 'r', encoding='utf-8') as fp:
    for line in fp.readlines():
        step += 1
        line = json.loads(line)
        entityMentions = line['entityMentions']
        words = line["sentText"]
        # 去掉换行符
        words = words[:-1]
        start_to_end = dict()
        starts = []
        for (key, entityMention) in enumerate(entityMentions):
            start = entityMention["start"]
            end = entityMention["end"]
            start_to_end[start] = end
            starts.append(start)
        starts.sort()
        starts_comb = list(itertools.combinations(starts, 2))
        for (e1start, e21start) in starts_comb:
            start = e1start
            end = start_to_end[e21start]
            sub_text = words[start:end]
            em1Text = words[e1start:start_to_end[e1start]]
            em2Text= words[e21start:start_to_end[e21start]]
            if em1Text != "" and em2Text!="" and sub_texts!="":
                sub_texts.append({
                    "text":sub_text,
                    "step":step, "em1Text":em1Text,"em2Text":em2Text,
                    "e1start":e1start,"e21start":e21start,
                })


if  os.path.exists(save_dir):
    with open(save_train_path, 'w', encoding='utf-8') as f:
        for line in sub_texts:
            f.write(json.dumps(line, ensure_ascii=False) + '\n')