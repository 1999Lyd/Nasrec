import argparse
import os
import sys
sys.path.append(os.getcwd())
import warnings
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset,DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.nn.utils.prune as prune
# Project imports
from nasrec.supernet.supernet import (
    SuperNet,
    ops_config_lib,
)
from nasrec.torchrec.utils import ParallelReadConcat

# FB internal for manifold read. Can be removed upon code release.
from nasrec.utils.config import (
    NUM_EMBEDDINGS_CRITEO,
    NUM_EMBEDDINGS_AVAZU,
    NUM_EMBEDDINGS_KDD,
)

from nasrec.utils.data_pipes import (
    get_criteo_kaggle_pipes,
    get_avazu_kaggle_pipes,
    get_kdd_kaggle_pipes,
)

from nasrec.utils.lr_schedule import (
    CosineAnnealingWarmupRestarts,
    ConstantWithWarmup,
)

from nasrec.utils.train_utils import (
    get_l2_loss,
    train_and_test_one_epoch,
    warmup_model,
    get_model_flops_and_params,
    init_weights,
)
from nasrec.utils.io_utils import (
    load_json,
    create_dir,
    dump_pickle_data,
    load_pickle_data,
    load_model_checkpoint,
    save_model_checkpoint
)
from nasrec.supernet.modules import flags
from nasrec.supernet.modules import ElasticLinear
warnings.simplefilter("ignore", ResourceWarning)
warnings.simplefilter("ignore", UserWarning)


# define the model 
_num_sparse_inputs_dict = {
    'criteo-kaggle': 26,
    'avazu': 23,
    'kdd': 10
}
_num_embedding_dict = {
    'criteo-kaggle': NUM_EMBEDDINGS_CRITEO,
    'avazu': NUM_EMBEDDINGS_AVAZU,
    "kdd": NUM_EMBEDDINGS_KDD,
}

def get_model(args):
    if args.net == "sparse_nn":
        model = DLRM(
            sparse_input_size=_num_sparse_inputs_dict[args.dataset],
            num_embeddings=_num_embedding_dict[args.dataset],
            embedding_dim=16,
            dense_arch_dims=[20, 16],
            over_arch_dims=[16],
            arch_interaction_itself=False,
            activation=args.activation,
        )
    elif args.net == "supernet":
        # Train the full supernet using DLRM spec.
        model = SuperNet(
            sparse_input_size=_num_sparse_inputs_dict[args.dataset],
            num_blocks=7,
            ops_config=ops_config_lib["xlarge"],
            use_layernorm=True,
            activation=args.activation,
            num_embeddings=_num_embedding_dict[args.dataset],
            path_sampling_strategy="full-path",
        )
    elif args.net == "supernet-config":
        choice = load_json(args.supernet_config)
        print(choice)
        model = SuperNet(
            sparse_input_size=_num_sparse_inputs_dict[args.dataset],
            num_blocks=choice["num_blocks"],
            ops_config=ops_config_lib[choice["config"]],
            use_layernorm=False,
            # use_layernorm=(choice["use_layernorm"] == 1),
            activation=args.activation,
            num_embeddings=_num_embedding_dict[args.dataset],
            path_sampling_strategy="fixed-path",
            fixed=True,
            fixed_choice=choice,
        )
    else:
        raise NotImplementedError("Model {} is not implemented!".format(args.net))
    return model

def main(args):
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True

    #create_dir(args.logging_dir)

    #define the model
    model = get_model(args)
    
    #warm up the model by running one step inference
    int_feats = torch.randn((1, 13), dtype=torch.float)
    cat_feats = torch.randint(low=0, high=100, size=(1, 26), dtype=torch.long)
    label = torch.randn((1,1), dtype=torch.float)
    train_set = TensorDataset(int_feats, cat_feats,label)
    train_loader = DataLoader(
            train_set,
            batch_size=1,
            num_workers=0
            )
    model(int_feats,cat_feats)
    # Load the model state dict from a .pt file
    checkpoint = torch.load("supernet-config_checkpoint.pt")

    model_state_dict = checkpoint['model_state_dict']

    model.load_state_dict(model_state_dict, strict=True)
    
    model.eval()
    
    output = model(int_feats,cat_feats)
    print(output)
    Flops = get_model_flops_and_params(model,train_loader,0)
    print("model before pruning:{}".format(Flops))

    Pruned_state_dict = torch.load("pruned_supernet.pt")
    model.load_state_dict(Pruned_state_dict, strict=True)
    model.to('cpu').eval()
    output = model(int_feats.to('cpu'),cat_feats.to('cpu'))
    print(output)
    Flops = get_model_flops_and_params(model,train_loader,0)
    print("model after pruning:{}".format(Flops))

if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="criteo-kaggle",
        help="Choice of datasets",
        choices=["criteo-kaggle", "avazu", "kdd"],
    )
    parser.add_argument(
        "--root_dir",
        type=str,
        default="/home/tunhouzhang/local/datasets/criteo_kaggle_sharded",
    )
    parser.add_argument(
        "--logging_dir", type=str, default=None, help="Directory to put loggings."
    )
    parser.add_argument(
        "--net",
        type=str,
        default="dlrm",
        help="Network backbone name",
        choices=["supernet", "supernet-config"],
    )
    parser.add_argument(
        "--supernet_config", type=str, default=None, help="Supernet configuration."
    )
    # Hyperparameters.
    parser.add_argument(
        "--wd", type=float, default=1e-8, help="L2 Weight decay")
    parser.add_argument(
        "--learning_rate", type=float, default=0.01, help="Learning rate"
    )
    parser.add_argument(
        "--learning_rate_decay", type=float, default=0, help="Learning rate decay."
    )
    parser.add_argument(
        "--num_epochs", type=int, default=1, help="Number of epochs for training."
    )
    parser.add_argument(
        "--train_split",
        type=str,
        default="trainval",
        help="Data split for training. Can be one of ['train', 'trainval']",
        choices=["train", "trainval"],
    )
    parser.add_argument(
        "--validate_split",
        type=str,
        default="test",
        help="Data split for validation (evaluation). Can be one of ['val', 'test']",
        choices=["val", "test"],
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=200, help="Training batch size."
    )
    parser.add_argument(
        "--test_batch_size", type=int, default=16368, help="Testing batch size."
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="adagrad",
        help="Optimizer",
        choices=["adagrad", "sgd", "adam", "rmsprop", "ds-optimizer"],
    )
    #-------------------Criteo------------------
    # Train: 36672495 Val: 4584061 Test: 4584061 Trainval: 41256556
    # ------------------Avazu-------------------
    # Train: 32343175 Val: 4042896 Test: 4042896 Trainval: 36386071
    # ------------------KDD---------------------
    # Train: 119711284 Val: 14963910 Test: 14963910 Trainval: 134675194
    parser.add_argument(
        "--train_limit",
        type=int,
        default=41256556,
        help="Maximum number of training examples.",
    )
    parser.add_argument(
        "--test_limit",
        type=int,
        default=4584061,
        help="Maximum number of testing examples.",
    )
    parser.add_argument(
        "--lr_schedule", default="cosine", help="Learning rate schedule",
        choices=["cosine", "constant", "constant-no-warmup"])
    parser.add_argument(
        "--display_interval",
        type=int,
        default=100,
        help="Interval to display tensorboard curve/training stats.",
    )
    parser.add_argument(
        "--test_interval", type=int, default=2000, help="Testing intervals."
    )
    # Currently useful in SparseNN-V2.
    parser.add_argument(
        "--activation",
        type=str,
        default="relu",
        help="Activation function to use in this work.",
        choices=["relu", "silu"],
    )
    parser.add_argument(
        "--no-reg-param-name",
        type=str,
        default=None,
        help="Name of the parameters that do not need to be regularized.",
    )
    # loss functions
    parser.add_argument(
        "--loss_function",
        type=str,
        default="bce",
        help="Loss function to perform the task.",
        choices=["bce"],
    )
    parser.add_argument(
        "--test_only_at_last_step",
        type=int,
        default=0,
        help="Whether only test the last step.",
    )
    # NOTE: not implemented.
    parser.add_argument(
        "--ema", type=float, default=0.0,
        help="EMA strength ranging from 0 to 1. 0.9/0.99/... is recommended.")
    # GPU utils
    parser.add_argument("--gpu", type=int, default=None, help="GPU ID to use.")
    args = parser.parse_args()
    main(args)

