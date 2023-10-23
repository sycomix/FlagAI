import json

fout = open(
    '/sharefs/baai-mrnd/xw/cpm3_train_data/cpm3_train_data.jsonl',
    "w",
    encoding='utf-8',
)
fin = open(
    '/sharefs/webbrain-lijijie/data/CEPSUM/test_public.jsonl',
    'r',
    encoding='utf-8',
)

def random_mask(source: str):
    if type(source) == list:
        return source
    length = len(source)
    half = length // 3
    return [source[:half],source[half * 2:]]

for line in fin:
    instance = {
        'mode': 'lm',
        'source': [],
        'target': "",
        'control': {
            'keywords': [],
            'genre': "",
            'relations': [],
            'events': []
        }
    }
    res = json.loads(line)
    for key in instance:
        if key == 'source':
            instance[key] = random_mask(res.get(key, instance[key]))
        elif key == 'target':
            instance[key] = res.get('source')
        else:
            instance[key] = res.get(key, instance[key])
    fout.write(json.dumps(instance) + '\n')
    fout.flush()