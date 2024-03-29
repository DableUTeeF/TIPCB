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
        resnet50 = models.resnet50(pretrained=True)
        resnet50.layer4[0].downsample[0].stride = (1, 1)
        resnet50.layer4[0].conv2.stride = (1, 1)
        self.base1 = nn.Sequential(
            resnet50.conv1,
            resnet50.bn1,
            resnet50.relu,
            resnet50.maxpool,
            resnet50.layer1,  # 256 64 32
        )
        self.base2 = nn.Sequential(
            resnet50.layer2,  # 512 32 16
        )
        self.base3 = nn.Sequential(
            resnet50.layer3,  # 1024 16 8
        )
        self.base4 = nn.Sequential(
            resnet50.layer4  # 2048 16 8
        )

    def forward(self, x):
        x1 = self.base1(x)
        x2 = self.base2(x1)
        x3 = self.base3(x2)
        x4 = self.base4(x3)
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
        # print(f'txt3: {txt3.size()} txt41: {txt41.size()} txt42: {txt42.size()} txt43: {txt43.size()} txt44: {txt44.size()} txt45: {txt45.size()} txt46: {txt46.size()} ')
        txt_f3, _ = txt3.max(2)
        txt_f41, _ = txt41.max(2)
        txt_f42, _ = txt42.max(2)
        txt_f43, _ = txt43.max(2)
        txt_f44, _ = txt44.max(2)
        txt_f45, _ = txt45.max(2)
        txt_f46, _ = txt46.max(2)
        # print(f'txt3: {txt_f3.size()} txt41: {txt_f41.size()} txt42: {txt_f42.size()} txt43: {txt_f43.size()} txt44: {txt_f44.size()} txt45: {txt_f45.size()} txt46: {txt_f46.size()} ')
        txt_f4, _ = torch.stack([txt_f41, txt_f42, txt_f43, txt_f44, txt_f45, txt_f46], dim=2).max(2)
        txt_f4 = txt_f4.squeeze(dim=-1).squeeze(dim=-1)
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
