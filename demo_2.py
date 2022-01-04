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


def video_vector(tipcb, detector, frame, frame_id, video_name,
                 vector_folder='/media/palm/BiggerData/caption/vectors'):
    os.makedirs(os.path.join(vector_folder, video_name), exist_ok=True)
    cropped_images, boxes = detect(detector, frame)
    vectors = []
    for i, image in enumerate(cropped_images):
        image = Image.fromarray(image)
        image = transform(image).unsqueeze(0)
        vector = tipcb.forward_img(image.to(device))
        os.makedirs(os.path.join(vector_folder, video_name), exist_ok=True)
        vectors.append(vector)
        # torch.save(vector, os.path.join(vector_folder, video_name, f'{frame_id:07d}_{i:04d}.pth'))
    return cropped_images, vectors


def text2vector(text):
    text = 'ผู้ชายใส่เสื้อกล้าม'
    token = tokenizer.encode(text, add_special_tokens=True, max_length=64, padding='max_length')
    mask = (np.array(token) > 0).astype('int64')
    vector = tipcb.forward_text(torch.tensor(token).unsqueeze(0).to(device), torch.tensor(mask).unsqueeze(0).to(device)).cpu().numpy()
    return vector


device = 'cuda'
transform = get_transform()
if __name__ == '__main__':
    # with open(f'data/BERT_encode/thai_train_64.npz', 'rb') as f_pkl:
    #     data = pickle.load(f_pkl)
    video = cv2.VideoCapture('/home/palm/dwhelper/VIRAT Video Data-1.mp4')
    fps = video.get(cv2.CAP_PROP_FPS)
    cfg = Config.fromfile('/media/palm/BiggerData/mmdetection/configs/cascade_rcnn/cascade_rcnn_r101_fpn_1x_coco.py')
    detector = init_detector(cfg, '/home/palm/PycharmProjects/mmdetection/checkpoints/cascade_rcnn_r101_fpn_20e_coco_bbox_mAP-0.425_20200504_231812-5057dcc5.pth', device='cuda')
    tipcb = get_model().cuda()
    ann = hnswlib.Index(space='cosine', dim=2048)
    ann.init_index(max_elements=200, ef_construction=200, M=16)
    ann.set_ef(50)
    tokenizer = ppb.AutoTokenizer.from_pretrained('airesearch/wangchanberta-base-att-spm-uncased')

    image_indeice = {}
    timestamps = {}
    i = 0
    count = 0
    while True:
        ret, frame = video.read()
        if not ret:
            break
        i += 1
        if i % int(fps) != 0:
            continue

        cropped_images, vectors = video_vector(tipcb, detector, frame, i, 'Data-1')
        for im, vector in zip(cropped_images, vectors):
            ann.add_items(vector[0].cpu().numpy())
            image_indeice[count] = im
            timestamps[count] = i
            count += 1
        if i > 100:
            break

    text = 'ผู้หญิงเสื้อน้ำเงิน'
    vector = text2vector(text)
    labels, distances = ann.knn_query(vector, k=5)
    for i in range(len(labels[0])):
        label = labels[0][i]
        cv2.imshow(str(distances[0][i])+'_'+str(timestamps[i]), image_indeice[i])
    cv2.waitKey()
