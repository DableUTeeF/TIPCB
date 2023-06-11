import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='command for train on CUHK-PEDES')
    parser.add_argument('--name', default='Experiment01', type=str, help='output model name')

    #dataset_Directory
    #CUHK-PEDES
    parser.add_argument('--dir', type=str,
                        default=r'/content',
                        help='directory to store dataset')
    parser.add_argument('--dataset', type=str,
                        default="CUHKPEDES")

    ##save_Directory
    parser.add_argument('--checkpoint_dir', type=str,
                        default="/content/cp",
                        help='directory to store checkpoint')
    parser.add_argument('--log_dir', type=str,
                        default="/content/tb_log/",
                        help='directory to store log')
    parser.add_argument('--image_root_path', type=str,
                        default='/content/CUHK-PEDES/imgs',
                        help='directory to store log')
    parser.add_argument('--json_path', type=str,
                        default='/media/palm/BiggerData/caption/CUHK-PEDES/CUHK-PEDES/caption_all.json',
                        help='directory to store log')

    #word_embedding
    parser.add_argument('--max_length', type=int, default=64)
    parser.add_argument('--embedding_type', type=str,
                        default='BERT')

    #image setting
    parser.add_argument('--width', type=int, default=128)
    parser.add_argument('--height', type=int, default=384)

    #CNN setting
    parser.add_argument('--num_classes', type=int, default=11003)
    parser.add_argument('--feature_size', type=int, default=2048)
    parser.add_argument('--pretrained', action='store_false',
                       help='whether or not to restore the pretrained visual model')
    parser.add_argument('--droprate', default=0, type=float, help='drop rate')

    #experiment setting
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--num_epoches', type=int, default=80)
    parser.add_argument('--resume', action='store_true',
                        help='whether or not to restore the pretrained whole model')

    #loss function setting
    parser.add_argument('--CMPM', default=True)

    #Optimization setting
    parser.add_argument('--optimizer', type=str, default='adam')
    parser.add_argument('--wd', type=float, default=0.00004)

    #adam_setting
    parser.add_argument('--adam_lr', type=float, default=1e-4, help='the learning rate of adam')
    parser.add_argument('--adam_alpha', type=float, default=0.9)
    parser.add_argument('--adam_beta', type=float, default=0.999)
    parser.add_argument('--epsilon', type=float, default=1e-8)

    parser.add_argument('--lr_decay_type', type=str, default='MultiStepLR',
                        help='One of "MultiStepLR" or "StepLR" or "ReduceLROnPlateau"')
    parser.add_argument('--lr_decay_ratio', type=float, default=0.1)
    parser.add_argument('--epoches_decay', type=str, default='50', help='#epoches when learning rate decays')
    parser.add_argument('--warm_epoch', default=4, type=int, help='the first K epoch that needs warm up')

    # Default setting
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--gpus', type=str, default='6')
    args = parser.parse_args()
    return args
