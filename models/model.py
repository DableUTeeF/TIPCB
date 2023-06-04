from torchvision import models
from models.CNN_text import ResNet_text_50
import transformers as ppb
from torch.nn import init
import torch.nn as nn
import torch
import torch.nn.functional as F

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in') # For old pytorch, you may use kaiming_normal.
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm1d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal_(m.weight.data, std=0.001)
        init.constant_(m.bias.data, 0.0)


class ResNet_image_50(nn.Module):
    def __init__(self):
        super(ResNet_image_50, self).__init__()
        efficientnet = models.efficientnet_b4(True)
        efficientnet.features[6][0].block[1][0].stride = (1, 1)
        self.base1 = nn.Sequential(
            efficientnet.features[0],
            efficientnet.features[1],
            efficientnet.features[2],
        )
        self.base2 = nn.Sequential(
            efficientnet.features[3],
        )
        self.base3 = nn.Sequential(
            efficientnet.features[4],
            efficientnet.features[5],
        )
        self.base4 = nn.Sequential(
            efficientnet.features[6],
            efficientnet.features[7],
            efficientnet.features[8],
        )
        self.pad3 = nn.Sequential(
            nn.Conv2d(160, 1024, 1),
            nn.BatchNorm2d(1024),
            nn.ReLU()
        )
        self.pad4 = nn.Sequential(
            nn.Conv2d(1792, 2048, 1),
            nn.BatchNorm2d(2048),
            nn.ReLU()
        )

    def forward(self, x):
        x1 = self.base1(x)
        x2 = self.base2(x1)
        x3 = self.base3(x2)
        x4 = self.base4(x3)
        x3 = self.pad3(x3)
        x4 = self.pad4(x4)
        return x1, x2, x3, x4


class Network(nn.Module):
    def __init__(self, args, bert='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'):
        super(Network, self).__init__()

        self.model_img = ResNet_image_50()
        # print('a')
        self.model_txt = ResNet_text_50(args)
        # print('b')

        if args.embedding_type == 'BERT':
            model_class, tokenizer_class, pretrained_weights = (ppb.AutoModel, ppb.AutoTokenizer, bert)
            self.text_embed = model_class.from_pretrained(pretrained_weights)
            self.text_embed.eval()
            self.BERT = True
            # for p in self.text_embed.parameters():
            #     p.requires_grad = False
        # print('c')
        self.max_pool = nn.AdaptiveMaxPool2d((1, 1))

    @torch.no_grad()
    def forward_img(self, img):
        _, _, img3, img4 = self.model_img(img)  # img4: batch x 2048 x 24 x 8
        img_f4 = self.max_pool(img4).squeeze(dim=-1).squeeze(dim=-1)
        return img_f4

    @torch.no_grad()
    def forward_text(self, txt, mask):
        txt = self.text_embed(txt, attention_mask=mask)
        txt = txt[0]
        txt = txt.unsqueeze(1)
        txt = txt.permute(0, 3, 1, 2)
        txt3, txt41, txt42, txt43, txt44, txt45, txt46 = self.model_txt(txt)  # txt4: batch x 2048 x 1 x 64
        txt_f3 = self.max_pool(txt3).squeeze(dim=-1).squeeze(dim=-1)
        txt_f41 = self.max_pool(txt41)
        txt_f42 = self.max_pool(txt42)
        txt_f43 = self.max_pool(txt43)
        txt_f44 = self.max_pool(txt44)
        txt_f45 = self.max_pool(txt45)
        txt_f46 = self.max_pool(txt46)
        txt_f4 = self.max_pool(torch.cat([txt_f41, txt_f42, txt_f43, txt_f44, txt_f45, txt_f46], dim=2)).squeeze(dim=-1).squeeze(dim=-1)
        return txt_f4

    def forward(self, img, txt, mask):
        with torch.no_grad():
            txt = self.text_embed(txt, attention_mask=mask)
            txt = txt[0]
            txt = txt.unsqueeze(1)
            txt = txt.permute(0, 3, 1, 2)

        _, _, img3, img4 = self.model_img(img)  # img4: batch x 2048 x 24 x 8
        img_f3 = self.max_pool(img3).squeeze(dim=-1).squeeze(dim=-1)
        img_f41 = self.max_pool(img4[:, :, 0:4, :]).squeeze(dim=-1).squeeze(dim=-1)
        img_f42 = self.max_pool(img4[:, :, 4:8, :]).squeeze(dim=-1).squeeze(dim=-1)
        img_f43 = self.max_pool(img4[:, :, 8:12, :]).squeeze(dim=-1).squeeze(dim=-1)
        img_f44 = self.max_pool(img4[:, :, 12:16, :]).squeeze(dim=-1).squeeze(dim=-1)
        img_f45 = self.max_pool(img4[:, :, 16:20, :]).squeeze(dim=-1).squeeze(dim=-1)
        img_f46 = self.max_pool(img4[:, :, 20:, :]).squeeze(dim=-1).squeeze(dim=-1)
        img_f4 = self.max_pool(img4).squeeze(dim=-1).squeeze(dim=-1)

        txt3, txt41, txt42, txt43, txt44, txt45, txt46 = self.model_txt(txt)  # txt4: batch x 2048 x 1 x 64
        txt_f3 = self.max_pool(txt3).squeeze(dim=-1).squeeze(dim=-1)
        txt_f41 = self.max_pool(txt41)
        txt_f42 = self.max_pool(txt42)
        txt_f43 = self.max_pool(txt43)
        txt_f44 = self.max_pool(txt44)
        txt_f45 = self.max_pool(txt45)
        txt_f46 = self.max_pool(txt46)
        txt_f4 = self.max_pool(torch.cat([txt_f41, txt_f42, txt_f43, txt_f44, txt_f45, txt_f46], dim=2)).squeeze(dim=-1).squeeze(dim=-1)
        txt_f41 = txt_f41.squeeze(dim=-1).squeeze(dim=-1)
        txt_f42 = txt_f42.squeeze(dim=-1).squeeze(dim=-1)
        txt_f43 = txt_f43.squeeze(dim=-1).squeeze(dim=-1)
        txt_f44 = txt_f44.squeeze(dim=-1).squeeze(dim=-1)
        txt_f45 = txt_f45.squeeze(dim=-1).squeeze(dim=-1)
        txt_f46 = txt_f46.squeeze(dim=-1).squeeze(dim=-1)

        if self.training:
            return img_f3, img_f4, img_f41, img_f42, img_f43, img_f44, img_f45, img_f46, \
                   txt_f3, txt_f4, txt_f41, txt_f42, txt_f43, txt_f44, txt_f45, txt_f46
        else:
            return img_f4, txt_f4
