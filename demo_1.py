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


fontpath = 'THSarabun.ttf'
font = ImageFont.truetype(fontpath, 32)

def add_bbox(img, bbox, labels, color, conf=None, show_txt=True, pos='top'):
    # bbox = np.array(bbox, dtype=np.int32)
    # cat = (int(cat) + 1) % 80
    if conf:
        txt = '{}:{:.1f}'.format(labels, conf)
    else:
        txt = labels
    cat_size = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
    cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
    if show_txt:
        if pos == 'top':
            y1 = bbox[1] - cat_size[1] - 2
            y2 = bbox[1]
        else:
            y1 = bbox[3]
            y2 = bbox[3] + cat_size[1]
        img_pil = Image.fromarray(img)
        draw = ImageDraw.Draw(img_pil)
        draw.text((y1, y2), labels, font=font, fill=color)
        img = np.array(img_pil)
    return img


def detect(detector, image):
    result = inference_detector(detector, image)
    result = [x[x[:, -1] > 0.3] for x in result]  # filter confidence > 0.3
    result = result[0]  # class 0 is person
    images = []
    for box in result:
        x1, y1, x2, y2, _ = box.astype('int')
        images.append(image[y1:y2, x1:x2])
    return images, result


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


def gallery_text_vector(tipcb,
                        token,
                        mask,
                        vector_folder='/media/palm/BiggerData/caption/vectors',
                        ):
    if os.path.exists(os.path.join(vector_folder, 'text_embed_filenames.json')):
        names = json.load(open(os.path.join(vector_folder, 'text_embed_filenames.json')))
    else:
        names = {}
    try:
        vector = torch.load(os.path.join(vector_folder, 'text', names[str(token)]))
    except Exception as e:
        print(e)
        vector = tipcb.forward_text(torch.tensor(token).unsqueeze(0).to(device), torch.tensor(mask).unsqueeze(0).to(device))
        os.makedirs(os.path.join(vector_folder, 'text'), exist_ok=True)
        names[str(token)] = f'{len(names):07d}.pth'
        json.dump(names, open(os.path.join(vector_folder, 'text_embed_filenames.json'), 'w'))
        torch.save(vector, os.path.join(vector_folder, 'text', names[str(token)]))
    return vector

device = 'cuda'
transform = get_transform()
if __name__ == '__main__':
    with open(f'data/BERT_encode/thai_train_64.npz', 'rb') as f_pkl:
        data = pickle.load(f_pkl)

    video = cv2.VideoCapture('/home/palm/dwhelper/VIRAT Video Data-1.mp4')
    fps = video.get(cv2.CAP_PROP_FPS)
    cfg = Config.fromfile('/media/palm/BiggerData/mmdetection/configs/cascade_rcnn/cascade_rcnn_r101_fpn_1x_coco.py')
    detector = init_detector(cfg, '/home/palm/PycharmProjects/mmdetection/checkpoints/cascade_rcnn_r101_fpn_20e_coco_bbox_mAP-0.425_20200504_231812-5057dcc5.pth', device='cuda')
    tipcb = get_model().cuda()
    ann = hnswlib.Index(space='cosine', dim=2048)
    ann.init_index(max_elements=len(data['caption_id']), ef_construction=200, M=16)
    ann.set_ef(50)
    tokenizer = ppb.AutoTokenizer.from_pretrained('airesearch/wangchanberta-base-att-spm-uncased')

    demo_out_dir = '/media/palm/BiggerData/caption/demo'
    # gallery_vectors = []
    for i in range(len(data['caption_id'])):
        # gallery_vectors.append(gallery_text_vector(tipcb, data['caption_id'][i], data['attention_mask'][i]).cpu().numpy())
        vector = gallery_text_vector(tipcb, data['caption_id'][i], data['attention_mask'][i]).cpu().numpy()
        ann.add_items(vector[0])

    for i in range(400):
        ret, frame = video.read()
        if i % int(fps) != 0:
            continue
        # 1. predict
        t1 = time.time()
        raw_images, boxes = detect(detector, frame)
        # todo: reid

        # 2. pre-embed
        # todo: optimize pre-process
        images = [transform(Image.fromarray(im)) for im in raw_images]
        images = torch.stack(images)

        image_vectors = tipcb.forward_img(images.cuda()).cpu().numpy()

        labels, distances = ann.knn_query(image_vectors, k=5)
        for j in range(len(labels)):
            image = raw_images[j]
            caption = tokenizer.decode(data['caption_id'][labels[j][0]]).replace('<pad>', '').replace('<s>', '').replace('</s>', '')
            if caption.startswith(' '):
                caption = caption[1:]
            frame_res = add_bbox(frame, boxes[j].astype('int'), caption, (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 160), 0))
            cv2.imshow('a', frame_res)
            ch = cv2.waitKey()
            if ch == 27 or ch == ord("q") or ch == ord("Q"):
                break
