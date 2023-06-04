from torch.utils.data import Dataset
import torchvision.transforms as transforms
import functools
import os
from PIL import Image
import json
from sklearn.model_selection import train_test_split
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import create_transform
import torch

def split(args):
    with open(args.json_path, "rb") as f:
        caption_all = json.load(f)

    group_by_id = dict()
    for record in caption_all:
        if record["id"] not in group_by_id.keys():
            group_by_id[record["id"]] = []
        for caption in record["captions"]:
            group_by_id[record["id"]].append({
                "id": record["id"],
                "file_path": record["file_path"],
                "caption": caption
            })

    train, val = train_test_split([group_by_id[key] for key in group_by_id.keys()], test_size=0.076, random_state=42, shuffle=False)
    train = functools.reduce(lambda a, b: a + b, train)
    val = functools.reduce(lambda a, b: a + b, val)
    return train, val

def build_transform(is_train, args):
    if is_train:
        transform = create_transform(
            input_size=384,
            is_training=True,
            color_jitter=0.4,
            auto_augment='rand-m9-mstd0.5-inc1',
            re_prob=0.25,
            re_mode='pixel',
            re_count=1,
            interpolation='bicubic',
        )
        transform.transforms[0] = transforms.Compose([
            transforms.Resize((args.height, args.width), interpolation=Image.BICUBIC),
            transforms.Pad(10),
            transforms.RandomCrop((args.height, args.width)),
        ])
        return transform

    t = [transforms.Resize((args.height, args.width), interpolation=Image.BICUBIC),
         transforms.ToTensor(),
         transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)]

    return transforms.Compose(t)


class JSONDataset(Dataset):
    def __init__(self, args, data, train):
        self.train = train
        self.args = args
        self.data = data
        self.stage = 'train' if train else 'val'
        self.transform = build_transform(train, args)

    def __getitem__(self, index):
        item = self.data[index]
        img = Image.open(os.path.join(self.args.image_root_path, item["file_path"]))
        img = self.transform(img)
        label = item["id"]
        text = item["caption"]
        return img, text, label

    def __len__(self):
        return len(self.data)

    @staticmethod
    def collate_fn(batch):
        images, texts, labels = list(zip(*batch))
        return torch.stack(images), texts, torch.tensor(labels)
