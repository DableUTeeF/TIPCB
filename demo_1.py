from mmdet.apis import inference_detector, init_detector, show_result_pyplot
from mmcv import Config
import cv2
import os
import time
import pickle


if __name__ == '__main__':
    with open(f'data/BERT_encode/BERT_id_test_64_new.npz', 'rb') as f_pkl:
        data = pickle.load(f_pkl)
    filedir = '/media/palm/BiggerData/caption/CUHK-PEDES/CUHK-PEDES/imgs'
    cfg = Config.fromfile('/media/palm/BiggerData/mmdetection/configs/queryinst/queryinst_r101_fpn_300_proposals_crop_mstrain_480-800_3x_coco.py')
    cfg.load_from = 'https://download.openmmlab.com/mmdetection/v2.0/queryinst/queryinst_r101_fpn_300_proposals_crop_mstrain_480-800_3x_coco/queryinst_r101_fpn_300_proposals_crop_mstrain_480-800_3x_coco_20210904_153621-76cce59f.pth'
    model = init_detector(cfg, None, device='cuda')
    model.CLASSES = None

    for image_path in data['images_path']:
        filename = os.path.join(filedir, image_path)

        # test a single image
        t1 = time.time()
        result = inference_detector(model, filename)
        # show the results
        img = model.show_result(filename,
                                result,
                                score_thr=0.8, show=False)

        t2 = time.time() - t1
        cv2.imshow('a', cv2.resize(img, None, None, 0.5, 0.5))
        cv2.waitKey()
        cv2.destroyAllWindows()

