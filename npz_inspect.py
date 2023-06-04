from transformers import AutoTokenizer
import pickle
import json
import numpy as np


if __name__ == '__main__':
    jsfile = json.load(open('/media/palm/BiggerData/caption/CUHK-PEDES/CUHK-PEDES/caption_all.json'))
    old_tk = AutoTokenizer.from_pretrained('bert-base-uncased', model_max_length=64)
    for stage in ['test', 'train', 'val', ]:
        with open(f'/media/palm/BiggerData/caption/BERT_tokens/BERT_id_{stage}_64_new.npz', 'rb') as f_pkl:
            old_data = pickle.load(f_pkl)
            old_labels = list(old_data['labels'])
            old_captions = old_data['caption_id']
            old_images = old_data['images_path']
            old_attention_mask = old_data['attention_mask']
        tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L12-v2', model_max_length=64)
        labels = []
        captions = []
        images = []
        attention_mask = []
        for ann in jsfile:
            if ann['id'] in old_labels:
                oc = np.array(old_captions[old_data['labels'] == ann['id']].tolist())
                caps = old_tk.batch_decode(oc)
                for cap in ann['captions']:
                    tokens = tokenizer(cap, padding='max_length', truncation=True)
                    captions.append(tokens['input_ids'])
                    attention_mask.append(tokens['attention_mask'])
                    labels.extend(ann['id'])
                    images.append(ann['file_path'])
        data = {
            'labels': labels,
            'caption_id': captions,
            'images_path': images,
            'attention_mask': attention_mask
        }
        with open(f'BERT_id_{stage}_64_revised.npz', 'wb') as f_pkl:
            pickle.dump(data, f_pkl)
