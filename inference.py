from models.model import Network
from test_config import parse_args
import torchvision.transforms as transforms
from function import *
import cv2
import transformers as ppb
import torch


def get_model():
    args = parse_args()
    model = Network(args)
    start, model = load_checkpoint(model, '/home/palm/PycharmProjects/TIPCB/log/Experiment04/64.pth.tar')
    model.eval()
    return model


def get_transform():
    test_transform = transforms.Compose([
        transforms.Resize((args.height, args.width), interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return test_transform


args = parse_args()
if __name__ == '__main__':
    model = Network(args)
    start, model = load_checkpoint(model, '/home/palm/PycharmProjects/TIPCB/log/Experiment04/64.pth.tar')
    model.eval()
    tokenizer = ppb.AutoTokenizer.from_pretrained('airesearch/wangchanberta-base-att-spm-uncased')
    # print('d')
    s = torch.tensor([0.229, 0.224, 0.225])
    m = torch.tensor([0.485, 0.456, 0.406])
    test_transform = transforms.Compose([
        transforms.Resize((args.height, args.width), interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    # print('e')
    test_loaders = data_config(args.dir, batch_size=args.batch_size, split='test', max_length=args.max_length,
                              embedding_type=args.embedding_type, transform=test_transform)
    # print('f')
    with torch.no_grad():
        x = test_loaders[0]
        image_embeddings, text_embeddings = model(x[0].unsqueeze(0), x[1].unsqueeze(0), x[3].unsqueeze(0))
        image_embeddings = image_embeddings / (image_embeddings.norm(dim=1, keepdim=True) + 1e-12)
        inp = x[0].permute([1, 2, 0]).cpu().mul_(s).add_(m).numpy()[..., ::-1]
        inp *= 255
        inp = inp.astype('uint8')

        x2 = test_loaders[2]
        image_embeddings2, text_embeddings2 = model(x2[0].unsqueeze(0), x2[1].unsqueeze(0), x2[3].unsqueeze(0))
        image_embeddings2 = image_embeddings2 / (image_embeddings2.norm(dim=1, keepdim=True) + 1e-12)
        inp2 = x2[0].permute([1, 2, 0]).cpu().mul_(s).add_(m).numpy()[..., ::-1]
        inp2 *= 255
        inp2 = inp2.astype('uint8')

        x3 = test_loaders[500]
        image_embeddings3, text_embeddings3 = model(x3[0].unsqueeze(0), x3[1].unsqueeze(0), x3[3].unsqueeze(0))
        image_embeddings3 = image_embeddings3 / (image_embeddings3.norm(dim=1, keepdim=True) + 1e-12)
        inp3 = x3[0].permute([1, 2, 0]).cpu().mul_(s).add_(m).numpy()[..., ::-1]
        inp3 *= 255
        inp3 = inp3.astype('uint8')

    print('same:', torch.dist(image_embeddings, image_embeddings2))
    print('diff:', torch.dist(image_embeddings, image_embeddings3))
    print('diff:', torch.dist(image_embeddings2, image_embeddings3))
    print(tokenizer.decode(x[1]))
    print(tokenizer.decode(x2[1]))
    print(tokenizer.decode(x3[1]))

    cv2.imshow('a', inp)
    cv2.imshow('b', inp2)
    cv2.imshow('c', inp3)
    cv2.waitKey()
