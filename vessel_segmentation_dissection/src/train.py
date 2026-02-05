import sys
import json
import os
import os.path as osp
import random
import argparse
from datetime import datetime, timedelta
import operator
from tqdm import tqdm
import numpy as np
import torch
import json
import time
from models.get_model import get_arch
from utils.get_loaders import get_train_val_loaders
from utils.evaluation import evaluate
from utils.model_saving_loading import save_model

def compare_op(metric):
    """
    This should return an operator that given a, b returns True if a is better than b
    Also, let us return which is an appropriately terrible initial value for such metric
    """

    if metric == 'auc':
        op, init = operator.gt, 0
    elif metric == 'tr_auc':
        op, init = operator.gt, 0
    elif metric == 'dice':
        op, init = operator.gt, 0
    elif metric == 'loss':
        op, init = operator.lt, np.inf
    else:
        raise NotImplementedError

    return op, init

def set_seeds(seed_value, use_cuda):
    np.random.seed(seed_value)  # cpu vars
    torch.manual_seed(seed_value)  # cpu  vars
    random.seed(seed_value)  # Python
    if use_cuda:
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)  # gpu vars
        torch.backends.cudnn.deterministic = True  # needed
        torch.backends.cudnn.benchmark = False

def reduce_lr(optimizer, epoch, factor=0.1, verbose=True):
    for i, param_group in enumerate(optimizer.param_groups):
        old_lr = float(param_group['lr'])
        new_lr = old_lr * factor
        param_group['lr'] = new_lr
        if verbose:
            print(f'Epoch {epoch:5d}: reducing learning rate of group {i} to {new_lr:.4e}.')

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def run_one_epoch(
        loader,
        model,
        criterion,
        optimizer=None,
        scheduler=None,
        assess=False
        ):

    device='cuda' if next(model.parameters()).is_cuda else 'cpu'
    train = optimizer is not None  # if we are in training mode there will be an optimizer and train=True here

    if train:
        model.train()
    else:
        model.eval()

    if assess:
        logits_all, labels_all = [], []
    n_elems, running_loss, tr_lr = 0, 0, 0


    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        #inputs, labels = inputs, labels
        logits = model(inputs)
        if model.n_classes == 1:
            loss = criterion(logits, labels.unsqueeze(dim=1).float())  # BCEWithLogitsLoss()/DiceLoss()
        else:
            loss = criterion(logits, labels)  # CrossEntropyLoss()

        if train:  # only in training mode
            optimizer.zero_grad()
            loss.backward()
            tr_lr = get_lr(optimizer)
            optimizer.step()
            scheduler.step()
        if assess:
            logits_all.extend(logits)
            labels_all.extend(labels)

        # Compute running loss
        running_loss += loss.item() * inputs.size(0)
        n_elems += inputs.size(0)
        run_loss = running_loss / n_elems

    if assess:
        return logits_all, labels_all, run_loss, tr_lr
    return None, None, run_loss, tr_lr

def train_one_cycle(train_loader, model, criterion, optimizer=None, scheduler=None, cycle=0):

    model.train()
    cycle_len = scheduler.cycle_lens[cycle]

    with tqdm(range(cycle_len)) as t:
        for epoch in t:
            if epoch == cycle_len-1:
                assess=True # only get logits/labels on last cycle
            else: assess = False
            tr_logits, tr_labels, tr_loss, tr_lr = run_one_epoch(train_loader, model, criterion, optimizer=optimizer,
                                                          scheduler=scheduler, assess=assess)
            t.set_postfix(tr_loss_lr=f"{float(tr_loss):.4f}/{tr_lr:.6f}")

    return tr_logits, tr_labels, tr_loss

def train_model(model, optimizer, criterion, train_loader, val_loader, epochs, evaluate_every, scheduler, metric, exp_path, seed = 2):

    best_auc, best_dice, best_epoch = 0, 0, 0
    is_better, best_monitoring_metric = compare_op(metric)

    with tqdm(range(1, epochs+1), initial=1) as t:
        for epoch in t:
            model.train()
            if epoch%evaluate_every==0:
                assess=True # only get logits/labels when evaluating
            else:
                assess = False
            tr_logits, tr_labels, tr_loss, tr_lr = run_one_epoch(train_loader, model, criterion, optimizer=optimizer,
                                                        scheduler=scheduler, assess=assess)
            
            t.set_postfix(epoch=f"{epoch:d}/{epochs:d}", tr_loss_lr=f"{float(tr_loss):.4f}/{tr_lr:.6f}")

            if assess:
                print(25 * '-' + '  Evaluating ' + 25 * '-')
                tr_auc, tr_dice = evaluate(tr_logits, tr_labels)

                with torch.no_grad():
                    assess=True
                    vl_logits, vl_labels, vl_loss, _ = run_one_epoch(val_loader, model, criterion, assess=assess)
                    vl_auc, vl_dice = evaluate(vl_logits, vl_labels)
                    del vl_logits, vl_labels
                msg = f'Train/Val Loss: {tr_loss:.4f}/{vl_loss:.4f}  -- Train/Val AUC: {tr_auc:.4f}/{vl_auc:.4f}  -- Train/Val DICE: {tr_dice:.4f}/{vl_dice:.4f} -- LR={get_lr(optimizer):.6f}'
                print(msg.rstrip('0'))

                # check if performance was better than anyone before and checkpoint if so
                if metric == 'auc':
                    monitoring_metric = vl_auc
                elif metric == 'tr_auc':
                    monitoring_metric = tr_auc
                elif metric == 'loss':
                    monitoring_metric = vl_loss
                elif metric == 'dice':
                    monitoring_metric = vl_dice
                if is_better(monitoring_metric, best_monitoring_metric):
                    print(f'Best {metric} attained. {100*best_monitoring_metric:.2f} --> {100*monitoring_metric:.2f}')
                    best_auc, best_dice, best_epoch = vl_auc, vl_dice, epoch
                    best_monitoring_metric = monitoring_metric
                    if exp_path is not None:
                        print(25 * '-', ' Checkpointing ', 25 * '-')
                        save_model(exp_path, model, optimizer, seed)

    del model
    torch.cuda.empty_cache()
    
    return best_auc, best_dice, best_epoch

def show_kernels(model):
    import torch
    import torch.nn as nn
    # Iterando sobre as camadas e imprimindo os kernels
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            print(f"Kernels da camada {name}:")
            print(module.weight.shape)  # Imprime o shape do kernels
            print(module.weight.data)  # Imprime os kernels
            print("="*50)

def main(arg_dict=None):

    args = get_args(arg_dict)
    device = torch.device(args.device)

    # reproducibility
    seed_value = args.seed
    set_seeds(seed_value, args.device.startswith("cuda"))
    set_seeds(seed_value, False)

    # gather parser parameters
    transform = args.transform
    variation = args.variation
    model_id = args.model_id
    kernels = args.kernels
    model_name = args.model_name
    max_lr, bs = args.max_lr, args.batch_size
    epochs, evaluate_every = args.epochs, args.evaluate_every
    metric = args.metric
    in_c = args.in_c
    use_green = args.use_green

    if in_c==3 and use_green==True:
        raise ValueError('Parameter use_green is True, but in_c=3.')
    if in_c==3:
        channels = 'all'
    elif in_c==1:
        if use_green:
            channels = 'green'
        else:
            channels = 'gray'

    im_size = tuple([int(item) for item in args.im_size.split(',')])
    if isinstance(im_size, tuple) and len(im_size)==1:
        tg_size = (im_size[0], im_size[0])
    elif isinstance(im_size, tuple) and len(im_size)==2:
        tg_size = (im_size[0], im_size[1])
    else:
        sys.exit('im_size should be a number or a tuple of two numbers')

    do_not_save = args.do_not_save
    if do_not_save is False:
        experiment_path=args.save_path
        os.makedirs(experiment_path, exist_ok=True)
    else:
        experiment_path = None

    csv_train = args.csv_train
    csv_val = csv_train.replace('train', 'val')

    label_values = [0, 255]

    #print(f"* Creating Dataloaders, batch size = {bs}, workers = {args.num_workers}")
    train_loader, val_loader = get_train_val_loaders(csv_path_train=csv_train, csv_path_val=csv_val, batch_size=bs, 
                                                     tg_size=tg_size, label_values=label_values, channels=channels,
                                                     num_workers=args.num_workers, transform = transform, variation=variation)

    print(f'\n\n{'*'*10} Instantiating a {model_name} model - IDENTIFICATOR {model_id}')
    model = get_arch(model_name, kernels, in_c=in_c)
    model = model.to(device)
    #print(f"- Device: {device}")

    num_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    #print(f"- Total params: {num_p:,}")
    optimizer = torch.optim.Adam(model.parameters(), lr=max_lr)

    scheduler = torch.optim.lr_scheduler.PolynomialLR(optimizer, total_iters=epochs*len(train_loader), power=0.9)
    setattr(optimizer, 'max_lr', max_lr)  # store it inside the optimizer for accessing to it later

    criterion = torch.nn.BCEWithLogitsLoss()
    
    #print('- Loss function', str(criterion))
    #print('-' * 10,' Starting to train ')

    start = time.time()
    best_auc, best_dice, best_epoch = train_model(model, optimizer, criterion, train_loader, val_loader, epochs, evaluate_every, scheduler, metric, experiment_path, seed = seed_value)
    end = time.time()
    diff = end - start
    print(f"- Time to Train: {diff:.2f}")

    # Salva configurações do treinamento
    experiment_path=args.save_path
    config = vars(args)
    config["num_parameters"] = num_p  # Adiciona o número de parâmetros
    config['time_toTrain'] = f"{diff:.2f}"

    with open(osp.join(experiment_path,f'config{seed_value}.cfg'), 'w') as f:
            json.dump(config, f, indent=2)

    print(f"val_auc: {best_auc}")
    print(f"val_dice: {best_dice}")
    print(f"best_epoch: {best_epoch}")
    if do_not_save is False:
        
        dic = {"Best_AUC":f"{100*best_auc:.2f}","Best_DICE":f"{100*best_dice:.2f}","Best_epoch":f"{best_epoch}", "params": num_p }

        with open(osp.join(experiment_path,f'val_metrics{seed_value}.json'), 'w') as arquivo:
            json.dump(dic, arquivo, indent=4)
        

def get_parser():
    """Get parser for command line arguments."""

    parser = argparse.ArgumentParser()
    parser.add_argument('--transform', type=str, default=None, help='Transform type, can be \'erosion\', \'skeleton\' or \'contour_ratio\'')
    parser.add_argument('--variation', type=str, default=None, help='Skeletonization ratio variation and other preprocessing applied to the dataset')
    parser.add_argument('--model_id', type=str, default='1', help='number of the experiment')
    parser.add_argument('--seed', type=int, default=0, help='seed')
    parser.add_argument('--epochs', type=int, default=1000, help='number of epochs to train')
    parser.add_argument('--max_lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--csv_train', type=str, default='data/DRIVE/train.csv', help='path to training data csv')
    parser.add_argument('--kernels', type=str, default='1 3 3 2 1 3 3 2 1 3 3 2 3 1 3 3 2 3 1 3 3')    
    parser.add_argument('--model_name', type=str, default='unet', help='architecture: unet, dic_unet, wnet')
    parser.add_argument('--im_size', help='image size, e.g.: 600,400', type=str, default='512')
    parser.add_argument('--batch_size', type=int, default=4, help='batch Size')
    parser.add_argument('--evaluate_every', type=int, default=50, help='number of epochs between model evaluations')
    parser.add_argument('--metric', type=str, default='auc', help='which metric to use for monitoring progress (tr_auc/auc/loss/dice)')
    parser.add_argument('--in_c', type=int, default=3, help='number of image channels to use')
    parser.add_argument('--use_green', action='store_true', help='If set, uses green channel. Otherwise, convert image to gray.')
    parser.add_argument('--do_not_save', action='store_true', help='avoid saving anything')
    parser.add_argument('--save_path', type=str, default='date_time', help='path to save model (defaults to date/time')
    parser.add_argument('--num_workers', type=int, default=0, help='number of parallel (multiprocessing) workers to launch for data loading tasks (handled by pytorch) [default: %(default)s]')
    cuda = "cuda:0" if torch.cuda.is_available() else "cpu"
    parser.add_argument('--device', type=str, default= cuda, help='where to run the training code (e.g. "cpu" or "cuda:0") [default: %(default)s]')
    

    return parser

def get_args(arg_dict=None):
    """
    Get arguments from the command line or from a dictionary. To use the
    arguments from the command line, just call get_args(). Alternativally, you
    can create a dictionary like {"--arg1": "value1", "--arg2": "value2"} and
    pass to this function.
    """

    parser = get_parser()
    if arg_dict is None:
        args = parser.parse_args()
    else:
        vals = []
        for k, v in arg_dict.items():
            vals.extend([k, v])
        args = parser.parse_args(vals)

    return args

if __name__ == '__main__':
    main()
