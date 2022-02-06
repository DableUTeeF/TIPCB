from mmdet.apis import inference_detector, init_detector, show_result_pyplot
from mmcv import Config
import cv2
import os
import time
import pickle
import torch
from inference import get_model, get_transform
from PIL import Image, ImageFont, ImageDraw
import json
import hnswlib
import numpy as np
import transformers as ppb
from torchreid.utils import FeatureExtractor


def gallery_image_vectors(tipcb,  # todo: make a dedicated vector generator function with batch predict, and dedicated gallery loader without try/except
                          image_path,
                          image_folder='/media/palm/BiggerData/caption/CUHK-PEDES/CUHK-PEDES/imgs',
                          vector_folder='/media/palm/BiggerData/caption/vectors',
                          ):
    try:
        vector = torch.load(os.path.join(vector_folder, image_path + '.pth'))
    except:
        image = cv2.imread(os.path.join(image_folder, image_path))
        image = Image.fromarray(image)
        image = transform(image).unsqueeze(0)
        vector = tipcb.forward_img(image.to(device))
        os.makedirs(os.path.dirname(os.path.join(vector_folder, image_path)), exist_ok=True)
        torch.save(vector, os.path.join(vector_folder, image_path + '.pth'))
    return vector

def detect(detector, image):
    result = inference_detector(detector, image)
    result = [x[x[:, -1] > 0.3] for x in result]  # filter confidence > 0.3
    result = result[0]  # class 0 is person
    images = []
    for box in result:
        x1, y1, x2, y2, _ = box.astype('int')
        images.append(image[y1:y2, x1:x2])
    return images, result


def video_vector(tipcb, detector, frame, frame_id, video_name, extractor,
                 vector_folder='/media/palm/BiggerData/caption/vectors'):
    # os.makedirs(os.path.join(vector_folder, video_name), exist_ok=True)
    cropped_images, boxes = detect(detector, frame)
    tipcb_vectors = []
    reid_vectors = []
    for i, image in enumerate(cropped_images):
        image = Image.fromarray(image)
        image = transform(image).unsqueeze(0)
        tipcb_vector = tipcb.forward_img(image.to(device))
        tipcb_vectors.append(tipcb_vector)
        reid_vector = extractor(image)
        reid_vectors.append(reid_vector)
        # os.makedirs(os.path.join(vector_folder, video_name), exist_ok=True)
        # torch.save(vector, os.path.join(vector_folder, video_name, f'{frame_id:07d}_{i:04d}.pth'))
    return cropped_images, tipcb_vectors, reid_vectors


def text2vector(text):
    token = tokenizer.encode(text, add_special_tokens=True, max_length=64, padding='max_length')
    mask = (np.array(token) > 0).astype('int64')
    vector = tipcb.forward_text(torch.tensor(token).unsqueeze(0).to(device), torch.tensor(mask).unsqueeze(0).to(device)).cpu().numpy()
    return vector


def get_detail(text, show=True):
    vector = text2vector(text)
    labels, distances = tipcb_ann.knn_query(vector, k=5)
    for i in range(len(labels[0])):
        label = labels[0][i]
        if show:
            cv2.imshow(str(distances[0][i]) +'_' + str(timestamps[i]), image_indice[i])
    if show:
        cv2.waitKey()
    return labels, distances


device = 'cuda'
transform = get_transform()
if __name__ == '__main__':
    # with open(f'data/BERT_encode/thai_train_64.npz', 'rb') as f_pkl:
    #     data = pickle.load(f_pkl)
    video = cv2.VideoCapture('/home/palm/dwhelper/VIRAT Video Data-1.mp4')
    fps = video.get(cv2.CAP_PROP_FPS)
    cfg = Config.fromfile('/media/palm/BiggerData/mmdetection/configs/cascade_rcnn/cascade_rcnn_r101_fpn_1x_coco.py')
    detector = init_detector(cfg, '/home/palm/PycharmProjects/mmdetection/checkpoints/cascade_rcnn_r101_fpn_20e_coco_bbox_mAP-0.425_20200504_231812-5057dcc5.pth', device='cuda')

    tipcb = get_model('bert-base-uncased', '/media/palm/BiggerData/caption/cp/eng/model_best.pth.tar').cuda()
    tipcb_ann = hnswlib.Index(space='cosine', dim=2048)
    tipcb_ann.init_index(max_elements=2000, ef_construction=200, M=16)
    tipcb_ann.set_ef(50)

    tokenizer = ppb.AutoTokenizer.from_pretrained('bert-base-uncased')
    reid_extractor = FeatureExtractor(
        model_name='osnet_x1_0',
        model_path='/media/palm/BiggerData/deep-person-reid/cp/osnet_ms_d_c.pth.tar',
        device='cuda'
    )
    reid_ann = hnswlib.Index(space='cosine', dim=512)
    reid_ann.init_index(max_elements=2000, ef_construction=200, M=16)
    reid_ann.set_ef(50)

    reid_fetures = {}
    image_indice = {}
    timestamps = {}
    i = -1
    count = 0
    while True:
        ret, frame = video.read()
        if not ret:
            break
        i += 1
        if i % int(fps) != 0:
            continue

        cropped_images, tipcb_vectors, reid_vectors = video_vector(tipcb, detector, frame, i, 'Data-1', reid_extractor)
        for im, tipcb_vector, reid_vector in zip(cropped_images, tipcb_vectors, reid_vectors):
            if i==0:
                tipcb_ann.add_items(tipcb_vector[0].cpu().numpy())
                reid_ann.add_items(reid_vector[0].cpu().numpy())
                image_indice[count] = im
                timestamps[count] = [i]
                count += 1
            else:
                labels, distances = reid_ann.knn_query(reid_vector[0].cpu().numpy(), k=1)
                if distances[0] > 0.1:
                    tipcb_ann.add_items(tipcb_vector[0].cpu().numpy())
                    reid_ann.add_items(reid_vector[0].cpu().numpy())
                    image_indice[count] = im
                    timestamps[count] = [i]
                    count += 1
                else:
                    timestamps[int(labels[0])].append(i)

        # if i > 100:
        #     break

    text = 'a woman with blue shirt'
    vector = text2vector(text)
    labels, distances = tipcb_ann.knn_query(vector, k=50)
    for i in range(5):
        label = labels[0][i]
        cv2.imshow(str(distances[0][i]) +'_' + str(timestamps[i]), image_indice[i])
    cv2.waitKey()
    cv2.destroyAllWindows()
