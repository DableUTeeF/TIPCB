import sys

from transformers import AutoTokenizer
import torchvision.transforms as transforms
import os
os.environ['TOKENIZERS_PARALLELISM'] = 'true'
import yaml
import time
# from tensorboard_logger import configure, log_value
from test_model import start_test
from train_config import parse_args
from function import data_config, optimizer_function, load_checkpoint, lr_scheduler, AverageMeter, save_checkpoint, \
    gradual_warmup, fix_seed, Logger
from models.model import Network
from CMPM import Loss
from torch.utils.tensorboard import SummaryWriter
import tensorflow as tf
from test_model import test


def train(epoch, train_loader, network, opitimizer, compute_loss, args, checkpoint_dir, tokenizer, writer):
    train_loss = AverageMeter()
    # switch to train mode
    network.train()
    progbar = tf.keras.utils.Progbar(len(train_loader))

    for step, (images, text, labels) in enumerate(train_loader):
        images = images.to(args.device)
        labels = labels.to(args.device)
        tokens = tokenizer(text, truncation=True, padding='max_length', return_tensors='pt').to(args.device)
        captions = tokens['input_ids']
        mask = tokens['attention_mask']
        opitimizer.zero_grad()

        # compute loss
        img_f3, img_f4, img_f41, img_f42, img_f43, img_f44, img_f45, img_f46, \
        txt_f3, txt_f4, txt_f41, txt_f42, txt_f43, txt_f44, txt_f45, txt_f46 = network(images, captions, mask)
        loss = compute_loss(
            img_f3, img_f4, img_f41, img_f42, img_f43, img_f44, img_f45, img_f46,
            txt_f3, txt_f4, txt_f41, txt_f42, txt_f43, txt_f44, txt_f45, txt_f46, labels)
        train_loss.update(loss, images.shape[0])
        writer.add_scalar('Loss/train', loss.cpu().detach().numpy(), step + 1 + (epoch * len(train_loader)))

        # graduate
        loss.backward()
        opitimizer.step()
        printlog = [
            ('loss', loss.cpu().detach().numpy()),
        ]
        progbar.update(step + 1, printlog)

    state = {"epoch": epoch + 1,
             "state_dict": network.state_dict(),
             "W": compute_loss.W
             }

    save_checkpoint(state, epoch+1, checkpoint_dir)


def main(network, dataloader, compute_loss, optimizer, scheduler, start_epoch, args, checkpoint_dir, writer):
    start = time.time()
    tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2', model_max_length=args.max_length)
    for epoch in range(start_epoch, args.num_epoches):
        print("**********************************************************")

        if epoch < args.warm_epoch:
            print('learning rate warm_up')
            if args.optimizer == 'sgd':
                optimizer = gradual_warmup(epoch, args.sgd_lr, optimizer, epochs=args.warm_epoch)
            else:
                optimizer = gradual_warmup(epoch, args.adam_lr, optimizer, epochs=args.warm_epoch)

        train(epoch, dataloader['train'], network, optimizer, compute_loss, args, checkpoint_dir, tokenizer, writer)
        test(dataloader['test'], network, args, tokenizer, writer, epoch+1)
        scheduler.step()

        Epoch_time = time.time() - start
        start = time.time()
        print('Epoch_training complete in {:.0f}m {:.0f}s'.format(
            Epoch_time // 60, Epoch_time % 60))


if __name__=='__main__':
    args = parse_args()

    # load GPU
    str_ids = args.gpus.split(',')
    gpu_ids = []
    for str_id in str_ids:
        gid = int(str_id)
        if gid >= 0:
            gpu_ids.append(gid)

    # # set gpu ids
    # if len(gpu_ids) > 0:
    #     torch.cuda.set_device(gpu_ids[0])
    #     cudnn.benchmark = True  # make the training speed faster
    fix_seed(args.seed)

    name = args.name
    # set some paths
    checkpoint_dir = args.checkpoint_dir
    checkpoint_dir = os.path.join(checkpoint_dir, name)
    log_dir = args.log_dir
    log_dir = os.path.join(log_dir, name)

    sys.stdout = Logger(os.path.join(log_dir, "train_log.txt"))
    opt_dir = os.path.join('log', name)
    if not os.path.exists(opt_dir):
        os.makedirs(opt_dir)
    with open('%s/opts_train.yaml' % opt_dir, 'w') as fp:
        yaml.dump(vars(args), fp, default_flow_style=False)

    # pre-process the dataset
    transform_train_list = [
        transforms.Resize((args.height, args.width), interpolation=3),
        transforms.Pad(10),
        transforms.RandomCrop((args.height, args.width)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
    transform_val_list = [
        transforms.Resize((args.height, args.width), interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]

    # define dictionary: data_transforms
    data_transforms = {
        'train': transforms.Compose(transform_train_list),
        'val': transforms.Compose(transform_val_list),
    }

    dataloaders = data_config(args)
    writer = SummaryWriter(log_dir=os.path.join(args.log_dir, 'log'))

    # loss function
    if args.CMPM:
        print("import CMPM")

    compute_loss = Loss(args).cuda()
    model = Network(args).cuda()

    # compute the model size:
    print('Number of model parameters: {}'.format(
        sum([p.data.nelement() for p in model.parameters()])))

    # load checkpoint:
    if args.resume is not None:
        start_epoch, model = load_checkpoint(model, args.resume)
    else:
        print("Do not load checkpoint,Epoch start from 0")
        start_epoch = 0

    # opitimizer:
    opitimizer = optimizer_function(args, model)
    exp_lr_scheduler = lr_scheduler(opitimizer, args)
    main(model, dataloaders, compute_loss, opitimizer, exp_lr_scheduler, start_epoch, args, checkpoint_dir, writer)
