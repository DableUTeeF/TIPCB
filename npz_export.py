import json
import numpy as np
import transformers as ppb
import pandas as pd
import pickle
import os


def export(json_path, output_path):
    caption_all = json.load(open(json_path))
    folders = {}
    for caption in caption_all:
        f = os.path.split(caption['file_path'])[0]
        if f not in folders:
            folders[f] = 0
    model_class, tokenizer_class, pretrained_weights = (ppb.BertModel, ppb.AutoTokenizer, 'airesearch/wangchanberta-base-att-spm-uncased')
    tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
    encoded_text = {
        'train': {'caption_id': [],
                  'attention_mask': [],
                  'images_path': [],
                  'labels': []},
        'test': {'caption_id': [],
                  'attention_mask': [],
                  'images_path': [],
                  'labels': []},
        'val': {'caption_id': [],
                  'attention_mask': [],
                  'images_path': [],
                  'labels': []}
    }
    seen_id = []
    for image in caption_all:
        folder = os.path.split(image['file_path'])[0]
        if image['id'] not in seen_id:
            seen_id.append(image['id'])
            folders[folder] += 1

        if folders[folder] % 10 == 1:
            stage = 'test'
        elif folders[folder] % 10 == 2:
            stage = 'val'
        else:
            stage = 'train'

        for text in image['captions']:
            data = {}
            encoded = tokenizer.encode(text, add_special_tokens=True, max_length=128, padding='max_length')
            if len(encoded) > 64:  # this is why the npzs have "64" in their names
                encoded = encoded[:64]  # encoded[:63] + encoded[-1:]

            attention_mask = (np.array(encoded) > 0).astype('int64')
            image_path = image['file_path']

            encoded_text[stage]['caption_id'].append(encoded)
            encoded_text[stage]['attention_mask'].append(attention_mask)
            encoded_text[stage]['images_path'].append(image_path)
            encoded_text[stage]['labels'].append(image['id'])
            print(image['id'])

    for stage in encoded_text:
        encoded_text[stage]['caption_id'] = np.array(encoded_text[stage]['caption_id'])
        encoded_text[stage]['attention_mask'] = np.array(encoded_text[stage]['attention_mask'])
        encoded_text[stage]['images_path'] = pd.Series(encoded_text[stage]['images_path'])
        encoded_text[stage]['labels'] = pd.Series(encoded_text[stage]['labels'])
        with open(os.path.join(output_path, f'{stage}_64.npz'), 'wb') as f_pkl:
            pickle.dump(encoded_text[stage], f_pkl)

    return encoded_text



if __name__ == '__main__':
    export('/media/palm/BiggerData/caption/thai_data/caption_all_thai.json', '/home/palm/PycharmProjects/TIPCB/data/BERT_encode')
