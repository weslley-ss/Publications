import json
from pathlib import Path
import numpy as np
import shutil

def load_json(path: Path):
    with open(path, "r") as f:
        return json.load(f)

def save_json(path: Path, data: dict):
    with open(path, "w") as f:
        json.dump(data, f, indent=4)

def same_core(old: dict, new: dict, CHECK_KEYS) -> bool:
    flag = True
    for k in CHECK_KEYS:
        if(old.get(k) != new.get(k)):
            flag = False
            print(f"Key '{k}' mismatch: old='{old.get(k)}' vs new='{new.get(k)}'")
    return all(old.get(k) == new.get(k) for k in CHECK_KEYS)

def merge_old_into_new(old: dict, new: dict, EXCLUDED_KEYS) -> dict:
    for key, value in old.items():
        if key in EXCLUDED_KEYS:
            continue
        # só sobrescreve se a chave existir no novo
        if key in new:
            new[key] = value
    return new

def merge_configs(old_path: Path, new_path: Path, CHECK_KEYS, EXCLUDED_KEYS, identificador = None) -> bool:
    if old_path.exists() and new_path.exists():
        old_config = load_json(old_path)
        new_config = load_json(new_path)
        
        if same_core(old_config, new_config, CHECK_KEYS):
            merged_config = merge_old_into_new(old_config, new_config, EXCLUDED_KEYS)
            save_json(new_path, merged_config)
            print(f"Config merged successfully | {identificador}")
            return True
        else:
            print(f"Core configuration mismatch | {identificador}")
            return False
    else:
        print(f"File not found | {identificador}")
        return False
    
def merge_metrics(old_path: Path, new_path: Path, CHECK_KEYS = ["params"] , EXCLUDED_KEYS = [], identificador = None) -> bool:
    if old_path.exists() and new_path.exists():
        old_metrics = load_json(old_path)
        new_metrics = load_json(new_path)
        
        if same_core(old_metrics, new_metrics, CHECK_KEYS):
            merged_metrics = merge_old_into_new(old_metrics, new_metrics, EXCLUDED_KEYS)
            save_json(new_path, merged_metrics)
            #print(f"Metrics merged successfully | {identificador}")
            return True
        else:
            print(f"Core Metrics mismatch | {identificador}")
            return False
    else:
        print(f"File not found | {identificador}")
        return False


def copy_files(old_path, new_path, identificador = None):
    if old_path.exists():
        new_path.parent.mkdir(parents=True, exist_ok=True) # confirm that the old file exists
        shutil.copy(old_path, new_path)
        #print(f"Model copied successfully | {identificador}")
    else:
        print(f"File not found | {identificador}")
        False
        
def delete_old_file(path: Path, identificador=None):
    if path.exists():
        path.unlink()
        # print(f"Old model deleted | {identificador}")
        return True
    else:
        print(f"Old FILE not found | {identificador}")
        return False

def make_path(data_set:str, transformation: str, model: str = None):
    base_old = Path(f"/home/weslley/Documentos/Experimentos-baseline/vessel_baseline/experiments_transform/{transformation}{str("_" + model) if model != "dic_unet" else ''}")
    base_new = Path(f"/home/weslley/Documentos/Publications/vessel_segmentation_dissection/experiments/{data_set}/{transformation}/{model}/baseline_variation")
    return base_old, base_new
    
def main():
    print("MIGRATE 001: RESTORE CONFIGURATION, MODELS AND METRICS FILES FOR SHAPE TRANSFORMATIONS")

    data_set = "VessMAP"
    transformations = ["erosion", "skeleton"]
    models = ["dic_unet", "wnet"]
    index = 0
    
    for transformation in transformations:
        for model in models:
            
            base_old, base_new = make_path(data_set, transformation=transformation, model=model)
            
            for seed in [2, 4, 8]: 
                     
                #  CONFIG FILES (The kernels on original files was mofied manually because differences in the wrote file)
                old_config_path = base_old / f"config{seed}.cfg"
                new_config_path = base_new / f"config_{index:03d}" / f"config_{index:03d}_{seed}.cfg"
                merge_configs(old_config_path, new_config_path, 
                            CHECK_KEYS = ["seed", "kernels", "transform"],
                            EXCLUDED_KEYS=["variation", "model_id","save_path"], identificador=f"Config file: {transformation}.{model}.{seed}")
                                
                # METRICS FILES
                old_metrics_path = base_old / f"val_metrics{seed}.json"
                new_metrics_path = base_new / f"config_{index:03d}" / f"model_validation_metrics_config_{index:03d}_{seed}.json"
                merge_metrics(old_metrics_path, new_metrics_path, identificador=f"Json file: {transformation}.{model}.{seed}")            
                
                # MODEL FILES
                old_model_path = base_old / f"model_checkpoint{seed}.pth"
                new_model_path = base_new / f"config_{index:03d}" / f"model_checkpoint{seed}.pth"
                copy_files(old_model_path, new_model_path, identificador=f"Model checkpoint: {transformation}.{model}.{seed}")
                
                # DELETE FILES
                wrong_model_path = base_new / f"config_{index:03d}" / f"model_checkpoint_{index:03d}_{seed}.pth"
                #delete_old_file(wrong_model_path, f"{index} | model_checkpoint_{index:03d}_{seed}.pth")
    
    print("MIGRATE 001 DONE!!!")

main()