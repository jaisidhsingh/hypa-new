import argparse


def setup_args_for_separate_mapping():
    parser = argparse.ArgumentParser()
    # overall experiment settings
    parser.add_argument("--experiment-name", type=str, default="coco_vitb_st5b_val_best_0")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--results-folder", type=str, default="/home/mila/s/sparsha.mishra/scratch/hyperalignment/results")
    parser.add_argument("--logs-folder", type=str, default="/home/mila/s/sparsha.mishra/scratch/hyperalignment/logs")
    parser.add_argument("--checkpoint-folder", type=str, default="/home/mila/s/sparsha.mishra/scratch/hyperalignment/checkpoints")
    # model and dataset settings
    parser.add_argument("--image-encoder", type=str, default="vit_base_patch16_224")
    parser.add_argument("--text-encoder", type=str, default="sentence-t5-base")
    parser.add_argument("--feature-dataset", type=str, default="cc3m300k")
    parser.add_argument("--val-dataset", type=str, default="mscoco_val")
    parser.add_argument("--image-embed-dim", type=int, default=768)
    parser.add_argument("--text-embed-dim", type=int, default=768)
    parser.add_argument("--use-bias", type=bool, default=True)
    parser.add_argument("--logit-scale", type=float, default=100.0)
    parser.add_argument("--use-wandb", type=bool, default=False)
    # training settings
    parser.add_argument("--batch-size", type=int, default=int(pow(2, 14)))
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--num-epochs", type=int, default=500)
    parser.add_argument("--warmup-steps", type=int, default=50)
    parser.add_argument("--shuffle-data", type=bool, default=True)
    
    args = parser.parse_args()
    return args


def setup_args_for_joint_mapping():
    parser = argparse.ArgumentParser()
    # overall experiment settings
    parser.add_argument("--experiment-name", type=str, default="hnet_2_cos_sched")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--results-folder", type=str, default="../results")
    parser.add_argument("--logs-folder", type=str, default="../logs")
    parser.add_argument("--checkpoint-folder", type=str, default="../checkpoints")
    parser.add_argument("--save-every", type=int, default=5)
    parser.add_argument("--eval-every", type=int, default=2000)
    parser.add_argument("--use-wandb", type=bool, default=False)
    # model and dataset settings
    parser.add_argument("--num-image-encoders", type=int, default=3)
    parser.add_argument("--num-text-encoders", type=int, default=1)
    parser.add_argument("--feature-dataset", type=str, default="mscoco")
    parser.add_argument("--image-embed-dim", type=int, default=768)
    parser.add_argument("--text-embed-dim", type=int, default=768)
    # training settings
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--num-epochs", type=int, default=30)
    parser.add_argument("--shuffle-data", type=bool, default=True)
    parser.add_argument("--warmup-steps", type=int, default=500)
    
    args = parser.parse_args()
    return args

def setup_args_for_ft_joint_mapping():
    parser = argparse.ArgumentParser()
    # overall experiment settings
    parser.add_argument("--experiment-name", type=str, default="ft_hnet_1_epoch_1_vitb_st5b")
    parser.add_argument("--ckpt-experiment-name", type=str, default="hnet_1")
    parser.add_argument("--ckpt-epoch", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--results-folder", type=str, default="/workspace/jaisidh/hyperalignment/results")
    parser.add_argument("--logs-folder", type=str, default="/workspace/jaisidh/hyperalignment/logs")
    parser.add_argument("--checkpoint-folder", type=str, default="/workspace/jaisidh/hyperalignment/checkpoints")
    # model and dataset settings
    parser.add_argument("--image-encoder", type=str, default="vit_base_patch16_224")
    parser.add_argument("--text-encoder", type=str, default="sentence-t5-base")
    parser.add_argument("--hnet-cond-id", type=int, default=0)
    parser.add_argument("--mode", type=str, default="ghn3")
    parser.add_argument("--ghn3-name", type=str, default="ghn3tm8.pt")
    parser.add_argument("--node-k", type=int, default=12)
    parser.add_argument("--feature-dataset", type=str, default="cc3m300k_id_vitr_var")
    parser.add_argument("--val-dataset", type=str, default="mscoco_val_id_vitr_val")
    parser.add_argument("--image-embed-dim", type=int, default=768)
    parser.add_argument("--text-embed-dim", type=int, default=768)
    parser.add_argument("--use-bias", type=bool, default=True)
    parser.add_argument("--logit-scale", type=float, default=100.0)
    parser.add_argument("--use-wandb", type=bool, default=False)
    # training settings
    parser.add_argument("--batch-size", type=int, default=int(pow(2, 14)))
    parser.add_argument("--learning-rate", type=float, default=1e-1)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--num-epochs", type=int, default=50)
    parser.add_argument("--warmup-steps", type=int, default=80)
    parser.add_argument("--shuffle-data", type=bool, default=True)
    
    args = parser.parse_args()
    return args

def setup_args_for_ghn3():
    parser = argparse.ArgumentParser()
    # overall experiment settings
    parser.add_argument("--experiment-name", type=str, default="ghn3_hnet_test_0")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--results-folder", type=str, default="/workspace/jaisidh/hyperalignment/results")
    parser.add_argument("--logs-folder", type=str, default="/workspace/jaisidh/hyperalignment/logs")
    parser.add_argument("--checkpoint-folder", type=str, default="/workspace/jaisidh/hyperalignment/checkpoints")
    parser.add_argument("--save-every", type=int, default=5)
    parser.add_argument("--eval-every", type=int, default=1)
    # model and dataset settings
    parser.add_argument("--num-image-encoders", type=int, default=3)
    parser.add_argument("--num-text-encoders", type=int, default=1)
    parser.add_argument("--feature-dataset", type=str, default="mscoco")
    parser.add_argument("--eval-dataset", type=str, default="cifar10")
    parser.add_argument("--image-embed-dim", type=int, default=768)
    parser.add_argument("--text-embed-dim", type=int, default=768)
    parser.add_argument("--logit-scale", type=float, default=100.0)
    parser.add_argument("--use-wandb", type=bool, default=False)
    parser.add_argument("--ghn3-name", type=str, default="ghn3xlm16.pt")
    parser.add_argument("--node-k", type=int, default=12)
    parser.add_argument("--node-downsampling", type=str, default="mean")
    # training settings
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--eval-batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--num-epochs", type=int, default=30)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--shuffle-data", type=bool, default=True)
    
    args = parser.parse_args()
    return args
