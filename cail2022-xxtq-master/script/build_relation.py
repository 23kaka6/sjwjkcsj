import json
import os

save_dir = '../data/'
save_train_path = save_dir + 'relation_train.json'
clip_data = []
# for file_index in range(train_file_list_index, train_file_clip_num):
with open("../data/step1_train.json", 'r', encoding='utf-8') as fp:
    for line in fp.readlines():
        line_json = json.loads(line)
        entityMentions =  line_json['entityMentions']
        relationMentions = line_json['relationMentions']
        words = line_json["sentText"]
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
            clip_data.append({
                "words": sub_text,
                "labels": label
            })


if  os.path.exists(save_dir):
    with open(save_train_path, 'w', encoding='utf-8') as f:
        for line in clip_data:
            f.write(json.dumps(line, ensure_ascii=False) + '\n')