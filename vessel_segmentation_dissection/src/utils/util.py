import json
from pathlib import Path
from utils.get_loaders import get_train_val_datasets

def get_experiment_data(exps_root, exp_id, ds_root):

    # Need this in order to import the model
    import sys
    sys.path.insert(0, str(exps_root/"code"))

    from models.get_model import get_arch
    from utils.model_saving_loading import load_model

    config_file = exps_root/"experiments"/f"{exp_id}"/"config.cfg"

    args = json.load(open(config_file, 'r'))
    
    experiment_path = args["experiment_path"]
    model_name = args["model_name"]
    kernels = args["kernels"]
    in_c = args["in_c"]
    tg_size = (int(args["im_size"]), int(args["im_size"]))

    experiment_path_full = exps_root/Path(experiment_path).relative_to("..")
    ds_train, ds_test = get_train_val_datasets("../data/VessMAP/train.csv", 
                                               "../data/VessMAP/test_all.csv",
                                                tg_size=tg_size)
    model = get_arch(model_name, kernels, in_c=in_c)
    model, stats, _ = load_model(model, experiment_path_full)

    return ds_train, ds_test, model
