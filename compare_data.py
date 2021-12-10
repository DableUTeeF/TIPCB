import pickle
import transformers as ppb
import json
import numpy as np
import pandas as pd


def find_indice(data, caption_all, tokenizer, seen={}):
    """
    :param data: the loaded pickle data (yes it's pickle even though the extension is npz)
    :param caption_all: the json with the same name
    :param tokenizer: just in case this function is called elsewhere
    :param seen: so that we don't need to call tokenizer over and over
    :return: 1. list of pair of indice. use by `caption_all[idx[0]]['captions'][idx[1]]`
             2. the updated `seen`. sometime pointer doesn't work
    """
    indice = []
    used_path = []
    for idx in range(len(data['caption_id'])):
        image_path = data['images_path'][idx]
        if image_path in used_path:
            continue
        for i, caption in enumerate(caption_all):
            if caption['file_path'] == image_path:
                used_path = image_path
                for j, text in enumerate(caption['captions']):
                    encoded = tokenizer.encode(text, add_special_tokens=True, max_length=128, padding='max_length')
                    if len(encoded) > 128:  # this is why the npzs have "64" in their names
                        encoded = encoded[:128]  # encoded[:63] + encoded[-1:]
                    seen[text] = encoded
                    indice.append((i, j))
                break
    return indice, seen


if __name__ == '__main__':
    caption_all = json.load(open('/media/palm/BiggerData/caption/thai_data/caption_all_thai.json'))
    print('a')
    model_class, tokenizer_class, pretrained_weights = (ppb.BertModel, ppb.AutoTokenizer, 'airesearch/wangchanberta-base-att-spm-uncased')
    tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
    seen = {}
    for stage in ['test', 'train', 'val']:
        with open(f'data/BERT_encode/BERT_id_{stage}_64_new.npz', 'rb') as f_pkl:
            data = pickle.load(f_pkl)
        indice, seen = find_indice(data, caption_all, tokenizer, seen)
        out_data = {
            'caption_id': [],
            'attention_mask': [],
            'images_path': [],
            'labels': []
        }
        unique_id = {}  # because we need to reassign the id. although called "labels" in the `data`.
        for idx in indice:
            image_data = caption_all[idx[0]]
            if image_data['id'] not in unique_id:
                unique_id[image_data['id']] = len(unique_id) + 1  # it starts from 1 btw
            caption = caption_all[idx[0]]['captions'][idx[1]]
            encoded = seen[caption]  # called "caption_id" in `data`
            attention_mask = (np.array(encoded) > 0).astype('int64')
            image_path = image_data['file_path']
            caption_id = unique_id[image_data['id']]  # called "labels" in the `data`

            out_data['caption_id'].append(encoded)
            out_data['attention_mask'].append(attention_mask)
            out_data['images_path'].append(image_path)
            out_data['labels'].append(caption_id)
        out_data['caption_id'] = np.array(out_data['caption_id'])
        out_data['attention_mask'] = np.array(out_data['attention_mask'])
        # I have no idea why these have to be `pd.Series` but whatever
        out_data['images_path'] = pd.Series(out_data['images_path'])
        out_data['labels'] = pd.Series(out_data['labels'])
        # print(out_data)
        with open(f'data/BERT_encode/thai_{stage}_128.npz', 'wb') as f_pkl:
            pickle.dump(out_data, f_pkl)
