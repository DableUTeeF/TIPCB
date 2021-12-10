from torchreid.utils import FeatureExtractor
import pickle
import torch
import os
import time
import hnswlib
import numpy as np

if __name__ == '__main__':
    with open(f'data/BERT_encode/BERT_id_test_64_new.npz', 'rb') as f_pkl:
        data = pickle.load(f_pkl)

    extractor = FeatureExtractor(
        model_name='osnet_x1_0',
        model_path='/media/palm/BiggerData/deep-person-reid/cp/osnet_ms_d_c.pth.tar',
        device='cuda'
    )
    filedir = '/media/palm/BiggerData/caption/CUHK-PEDES/CUHK-PEDES/imgs'

    image_list = [
        os.path.join(filedir, data['images_path'][0]),
        os.path.join(filedir, data['images_path'][1]),
        os.path.join(filedir, data['images_path'][5]),
        os.path.join(filedir, data['images_path'][6]),
        os.path.join(filedir, data['images_path'][7]),
        os.path.join(filedir, data['images_path'][12]),
        os.path.join(filedir, data['images_path'][13]),
        os.path.join(filedir, data['images_path'][14]),
        os.path.join(filedir, data['images_path'][15]),
        os.path.join(filedir, data['images_path'][16]),
        os.path.join(filedir, data['images_path'][17]),
        os.path.join(filedir, data['images_path'][18]),
        os.path.join(filedir, data['images_path'][19]),
        os.path.join(filedir, data['images_path'][20]),
    ]

    _ = extractor(image_list)
    p = hnswlib.Index(space='cosine', dim=512)
    p.init_index(max_elements=24, ef_construction=200, M=16)
    p.set_ef(50)
    t = time.time()
    with torch.no_grad():
        features = extractor(image_list)
        print(time.time() - t)
    features = features.cpu().numpy()
    print(time.time() - t)
    p.add_items(features[1:], np.arange(13))
    print(time.time() - t)
    labels, distances = p.knn_query(features[0], k=5)
    print(time.time() - t)
