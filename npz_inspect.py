from transformers import AutoTokenizer
import pickle
import json
import numpy as np
from timm.data.dataset import ImageDataset

if __name__ == '__main__':
    jsfile = json.load(open('/media/palm/BiggerData/caption/CUHK-PEDES/CUHK-PEDES/caption_all_encn.json'))
    old_tk = AutoTokenizer.from_pretrained('bert-base-uncased', model_max_length=64)
    for stage in ['test', 'train', 'val', ]:
        with open(f'/media/palm/BiggerData/caption/BERT_tokens/BERT_id_{stage}_64_new.npz', 'rb') as f_pkl:
            old_data = pickle.load(f_pkl)
            old_labels = list(old_data['labels'])
            old_captions = old_data['caption_id']
            old_images = list(old_data['images_path'])
            old_attention_mask = old_data['attention_mask']
        tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/distiluse-base-multilingual-cased', model_max_length=64)
        labels = []
        captions = []
        images = []
        attention_mask = []
        used = []
        for ann in jsfile:
            if ann['file_path'] in old_images and ann['file_path'] not in used:
                # used.append(ann['file_path'])

                tokens = tokenizer(ann['captions'], padding='max_length', truncation=True)
                captions.extend(tokens['input_ids'])
                attention_mask.extend(tokens['attention_mask'])

                images.extend([ann['file_path'] for _ in ann['captions']])
                labels.extend([old_labels[old_images.index(ann['file_path'])] for _ in ann['captions']])

        data = {
            'labels': labels,
            'caption_id': captions,
            'images_path': images,
            'attention_mask': attention_mask
        }
        with open(f'BERT_id_{stage}_64_revised.npz', 'wb') as f_pkl:
            pickle.dump(data, f_pkl)
