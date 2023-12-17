import argparse
import os
import sys
sys.path.append(os.getcwd())
import warnings
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import math
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
from nasrec.supernet.modules import ElasticLinear,ElasticLinear3D,SigmoidGating,DotProduct

#sys.path.append("/home/yl817/nas")

#from NasRec.xlarge_prune.prune_module import Projected_MaskedLinear
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

import torch.nn.init as init

class STEFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, scale, zero_point, Nbits):
        levels = 2 ** Nbits
        return quantize(input, scale, zero_point, levels)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None, None

def quantize(x, scale=1.0, zero_point=0, levels=256):
    alpha = x.max()
    # Compute beta (bias) for dynamic scaling, it's zero for non-symmetric quantization
    beta = x.min()
                # Scale w with alpha so that all elements in ws are between 0 and 1
    ws = (x-beta) / (alpha-beta)
                
    step = levels - 1
                # Quantize ws with a linear quantizer to "bit" bits
    R = torch.round(ws * step) / step
                # Scale the quantized weight R back with alpha
    wq = R * (alpha-beta) + beta

    return wq

class QuantizedLinear(torch.nn.Module):
    def __init__(self, in_features, out_features, Nbits, device='cuda:0'):
        super(QuantizedLinear, self).__init__()
        self.linear = torch.nn.Linear(in_features, out_features).to(device)
        # Define scale, zero_point for quantization
        self.scale = 1.0
        self.zero_point = 0
        self.Nbits = Nbits

    def forward(self, x):
        quantized_weights = STEFunction.apply(self.linear.weight, self.scale, self.zero_point, self.Nbits)
        return torch.nn.functional.linear(x, quantized_weights, self.linear.bias)


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
def train_and_eval_one_model(model, args):
    """Evaluate one certain model with the training pipeline and validation pipeline.
    Args:
        :params model: Backbone models used to train.
        :params args: Other hyperparameter settings.
    """

    get_pipes = {
        "criteo-kaggle": lambda: get_criteo_kaggle_pipes(args),
        "avazu": lambda: get_avazu_kaggle_pipes(args),
        "kdd": lambda: get_kdd_kaggle_pipes(args),
    }
    train_datapipes, test_datapipes, num_train_workers, num_test_workers = get_pipes[
        args.dataset
    ]()

    # Wrap up data-loader.
    train_loader = DataLoader(
        ParallelReadConcat(*train_datapipes),
        batch_size=None,
        num_workers=num_train_workers,
    )
    
    test_loader = DataLoader(
        ParallelReadConcat(*test_datapipes),
        batch_size=None,
        num_workers=num_test_workers,
    )
    
    '''
    data = pd.read_csv('nasrec/tools/data/random.csv')
    data = data.values
    int_input = torch.FloatTensor(data[1,1:14])[None,:]    
    cat_input = torch.LongTensor(data[1,14:])[None,:]

    label_train = torch.FloatTensor(data[:8000,0])
    label_train = label_train[:,None]
    
    Int_feature_train = torch.FloatTensor(data[:8000,1:14])
    
    Cat_feature_train = torch.LongTensor(data[:8000,14:])
    
    train_set = TensorDataset(Int_feature_train,Cat_feature_train,label_train)
    train_loader = DataLoader(
            train_set,
            batch_size=2,
            num_workers=0
            )
    label_test = torch.FloatTensor(data[8000:,0])
    label_test = label_test[:,None]
    Int_feature_test = torch.FloatTensor(data[8000:,1:14])
    Cat_feature_test = torch.LongTensor(data[8000:,14:])
    test_set = TensorDataset(Int_feature_test,Cat_feature_test,label_test)
    test_loader = DataLoader(
            test_set,
            batch_size=2,
            num_workers=0
            )
    '''
    '''
    # Debugging
    train_loader = DataLoader(train_datapipes[0],
        batch_size=None,
        num_workers=0)

    test_loader = DataLoader(test_datapipes[0],
        batch_size=None,
        num_workers=0)
    '''
    # Warmup models.
    with torch.no_grad():
        model = warmup_model(model, train_loader, args.gpu)
    # Get flops.
    flops, params = get_model_flops_and_params(model, train_loader, args.gpu)
    print("FLOPS: {:.4f} M \t Params: {:.4f} M".format(flops / 1e6, params / 1e6))
    # Functional headers for training purposes.
    if args.loss_function == "bce":
        loss_fn = torch.nn.BCEWithLogitsLoss()
    else:
        raise NotImplementedError(
            "Loss function {} is not implemented!".format(args.loss_function)
        )


    # Add tensorboard graph
    #flags.config_debug(True)
    #with torch.no_grad():
    #    int_x, cat_x, _ = next(iter(train_loader))
    #    
    #    int_x, cat_x = int_x.to(args.gpu)[0].unsqueeze_(0), cat_x.to(args.gpu)[0].unsqueeze_(0)
        
        #int_x = torch.rand((1, INT_FEATURE_COUNT), dtype=torch.float32).to(args.gpu)
        #cat_x = torch.randint(0, 1, size=(1, CAT_FEATURE_COUNT), dtype=torch.int32).to(args.gpu)
        #writer = SummaryWriter(args.logging_dir)
    #    writer.add_graph(model, (int_x, cat_x))
    
    # Training starts
    flags.config_debug(False)
    # L2 loss function.
    def _l2_loss_fn(model):
        """
        Customized Loss function. 
        Has an optional choice to discard embedding regularization and disabling bias decay.
        """
        return get_l2_loss(model, args.wd, args.no_reg_param_name, gpu=args.gpu)

    l2_loss_fn = _l2_loss_fn
    # Optimizer
    optimizer_lib = {
        "adagrad": torch.optim.Adagrad(
            model.parameters(), lr=args.learning_rate, eps=1e-2,
        ),
        "adam": torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate, eps=1e-8),
        "sgd": torch.optim.SGD(
            model.parameters(), lr=args.learning_rate, nesterov=True, momentum=0.9
        ),
    }
    optimizer = optimizer_lib[args.optimizer]
    # Scheduler
    num_train_steps_per_epoch = args.train_limit // args.train_batch_size
    num_train_steps = num_train_steps_per_epoch * args.num_epochs
    num_warmup_steps = num_train_steps_per_epoch // 10 // args.num_epochs
    if args.lr_schedule == "cosine":
        lr_scheduler = CosineAnnealingWarmupRestarts(
            optimizer,
            first_cycle_steps=num_train_steps,
            warmup_steps=num_warmup_steps,
            max_lr=args.learning_rate,
            min_lr=1e-8,
        )
    elif args.lr_schedule == "constant":
        lr_scheduler = ConstantWithWarmup(
            optimizer, num_warmup_steps=num_warmup_steps)
    else:
        # Do nothing.
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[num_train_steps * 10], gamma=0.1
        )

    model.apply(init_weights)
    print(model)

    create_dir(args.logging_dir)
    # Dump config args first.
    args_dump_path = os.path.join(args.logging_dir, "configs_args.json")
    with open(args_dump_path, 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    epoch_logs = []

    # Save the initial weights
    torch.save(model.state_dict(), 'initial_weights.pth')

    for epoch in range(args.num_epochs):
        # Train the model for 1 epoch.
        logs = train_and_test_one_epoch(
            model,
            epoch,
            optimizer,
            lr_scheduler,
            train_loader,
            test_loader,
            loss_fn,
            l2_loss_fn,
            args.train_batch_size,
            args.gpu,
            display_interval=args.display_interval,
            test_interval=args.test_interval if epoch != 1 else 100,
            max_train_steps=num_train_steps_per_epoch if epoch != 1 else 500,
            test_only_at_last_step=(args.test_only_at_last_step) == 1,
            tb_writer=None,
            use_amp=False,       # Disable-on-Pascal-Archs
            grad_clip_value=5.0,
        )
        epoch_logs.append(logs)
    # Dumping logs.
    print("Dumping logs to {}!".format(args.logging_dir))
    save_model_checkpoint(
        model, os.path.join(args.logging_dir, "{}_checkpoint.pt".format(args.net)), optimizer
    )
    dump_pickle_data(
        os.path.join(args.logging_dir, "masked_train_test_logs.pickle"), epoch_logs
    )





def mask_linear(layer, device, Nbits=8):  # Default to 8 bits
    self_linear = layer._left_self_linear._linear
    linear_proj = layer._linear_proj

    if self_linear is not None:
        masked_self_linear = QuantizedLinear(self_linear.in_features, self_linear.out_features, Nbits, device=device)
        masked_self_linear.linear.weight.data = self_linear.weight.data
        masked_self_linear.linear.bias.data = self_linear.bias.data
        layer._left_self_linear._linear = masked_self_linear

    if linear_proj is not None:
        masked_linear_proj = QuantizedLinear(linear_proj.in_features, linear_proj.out_features, Nbits, device=device)
        masked_linear_proj.linear.weight.data = linear_proj.weight.data
        masked_linear_proj.linear.bias.data = linear_proj.bias.data
        layer._linear_proj = masked_linear_proj


def mask_dot_linear(layer,device,Nbits=8):
    #dense_proj = layer._dense_proj
    #sparse_input_proj = layer._sparse_inp_proj
    linear_proj = layer._linear_proj
    #masked_dense_proj = Projected_MaskedLinear(dense_proj.in_features, dense_proj.out_features,device = 'cuda:1')
    #masked_sparse_input_proj = Projected_MaskedLinear(sparse_input_proj.in_features, sparse_input_proj.out_features,device = 'cuda:1')
    masked_linear_proj = QuantizedLinear(linear_proj.in_features, linear_proj.out_features, Nbits, device=device)
    #masked_dense_proj.weight.data = dense_proj.weight.data
    #masked_dense_proj.bias.data = dense_proj.bias.data
    #masked_sparse_input_proj.weight.data = sparse_input_proj.weight.data
    #masked_sparse_input_proj.bias.data = sparse_input_proj.bias.data
    masked_linear_proj.linear.weight.data = linear_proj.weight.data
    masked_linear_proj.linear.bias.data = linear_proj.bias.data
    # Replace the nn.LazyLinear layer in the ElasticLinear module with the pruned nn.Linear layer
    #layer._dense_proj = masked_dense_proj
    #layer._sparse_inp_proj = masked_sparse_input_proj
    layer._linear_proj = masked_linear_proj    
    

def mask_elastic_layer(elastic_layer,device,Nbits=4):
    layer = elastic_layer._linear

    # Compute the L1-norm of the neuron weights
    

    # Create a new nn.Linear layer with pruned neurons
    masked_layer = QuantizedLinear(layer.in_features, layer.out_features, Nbits, device=device)
    masked_layer.linear.weight.data = layer.weight.data
    masked_layer.linear.bias.data = layer.bias.data

    # Replace the nn.LazyLinear layer in the ElasticLinear module with the pruned nn.Linear layer
    elastic_layer._linear = masked_layer



   
def main(args):
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = True
    torch.manual_seed(42)
    #create_dir(args.logging_dir)

    #define the model
    model = get_model(args)
    
    #warm up the model by running one step inference
    int_feats = torch.randn((1, 13), dtype=torch.float)
    cat_feats = torch.randint(low=0, high=3, size=(1, 26), dtype=torch.long)
    model(int_feats,cat_feats)
    
   
    '''
    label = torch.randn((1,1), dtype=torch.float)
    train_set = TensorDataset(int_feats, cat_feats,label)
    train_loader = DataLoader(
            train_set,
            batch_size=1,
            num_workers=0
            )
    
    Flops = get_model_flops_and_params(model,train_loader,'cpu')
    print("model before pruning:{}".format(Flops))
    '''
    for idx, block in enumerate(model._blocks):
        
        if block.project_emb_dim is not None:
            Nbits=4
            layer = block.project_emb_dim
            masked_layer = QuantizedLinear(layer.in_features, layer.out_features, Nbits, device=args.gpu)
            masked_layer.linear.weight.data = layer.weight.data
            masked_layer.linear.bias.data = layer.bias.data
            block.project_emb_dim = masked_layer
        
        for node in block._nodes:
            if isinstance(node, ElasticLinear) or isinstance(node, ElasticLinear3D):
                mask_elastic_layer(node,args.gpu)
                    
            if isinstance(node, SigmoidGating):
                mask_linear(node,args.gpu)
            
            elif isinstance(node, DotProduct):
                
                mask_dot_linear(node,args.gpu)
                
            
    for name, param in model.named_parameters():
        print(name)
        param.requires_grad = True       
    


    model = model.to(args.gpu)
    
    train_and_eval_one_model(model, args)
    
   

# Save the pruned model
 #   torch.save(model.state_dict(), "pruned_supernet.pt")



#export the pruned model to onnx format
    
    
    

#run inference and get flop
    
    '''
    model.to('cpu').eval()
    output = model(int_feats.to('cpu'),cat_feats.to('cpu'))
    print(output)

    Flops = get_model_flops_and_params(model,train_loader,'cpu')
    print("model after pruning:{}".format(Flops))
    '''
    #model = model.to(args.gpu)
    
    
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

