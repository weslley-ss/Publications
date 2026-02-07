import os
import os.path as osp
import json

def generate_layer_combinations(base_values, new_values, flag = "kernels"):

    n_layers = len(base_values)
    all_confs = {"base": base_values}

    for v in new_values:
        # [CH1] Change only one layer
        for idx in range(n_layers):
            values = base_values[:]  # [:] is used to copy the list
            values[idx] = v
            dic = {f"CH1.{v}.{idx}": values}
            all_confs.update(dic)

        # [CH2] Change all layers up to idx
        values = base_values[:]
        for idx in range(n_layers):
            values[idx] = v
            dic = {f"CH2.{v}.{idx}": values[:]}
            all_confs.update(dic)

        if flag == "kernels":
            # [CH3] Change first conv of residual blocks
            values = base_values[:]
            for idx in range(n_layers):
                if idx%2==0:
                    values[idx] = v
            dic = {f"CH3.{v}.{1}": values[:]}
            all_confs.update(dic)

            # [CH4] Change second conv of residual blocks
            values = base_values[:]
            for idx in range(n_layers):
                if idx%2==1:
                    values[idx] = v
            dic = {f"CH4.{v}.{1}": values[:]}
            all_confs.update(dic)

            # [CH5] Change each residual block
            for idx in range(0, n_layers, 2):
                values = base_values[:]
                values[idx] = v
                values[idx+1] = v
                dic = {f"CH5.{v}.{idx}": values[:]}
                all_confs.update(dic)

            # [CH6] Change encoder
            values = base_values[:]
            for idx in range(0, 6):
                values[idx] = v
            dic = {f"CH6.{v}.{idx}": values[:]}
            all_confs.update(dic)

            # Change decoder
            values = base_values[:]
            for idx in range(6, n_layers):
                values[idx] = v
            dic = {f"CH7.{v}.{idx}": values[:]}
            all_confs.update(dic)
        
        elif flag == "residuals":
            # Change encoder residuals
            values = base_values[:]
            values[1] = v
            values[2] = v
            dic = {f"CH3.{v}.{idx}": values[:]}
            all_confs.update(dic)

            # Change decoder residuals
            values = base_values[:]
            values[3] = v
            values[4] = v
            dic = {f"CH4.{v}.{idx}": values[:]}
            all_confs.update(dic)
    
    return all_confs

def generate_models_configurations(base_k, base_dil, base_res, k_vals, dil_vals, res_vals, return_base = True):

    # Kernel configurations
    all_kernels = generate_layer_combinations(base_k, k_vals)

    # Dilation configurations
    all_dilations = generate_layer_combinations(base_dil, dil_vals)
    
    # Residual configurations
    all_residuals = generate_layer_combinations(base_res, res_vals, flag = "residuals")
    #print(all_residuals)

    # Downs configurations
    all_downs = [(2 ,2), (2, 1), (1, 2), (1, 1)]

    # Only for visualization
    base_dil = tuple(base_dil)
    base_k = tuple(base_k)
    base_downs = (2, 2)
    base_res = tuple(base_res)
    base_config = (base_k, base_dil, base_res, base_downs)
    if return_base == True:
        return [base_config]
    
    bases_kernels = [base_k]
    for v in k_vals:
        bases_kernels.append(all_kernels[f"CH2.{v}.9"])


    all_configs = []
    for kernels in all_kernels.values():
        all_configs.append((kernels, base_dil, base_res, base_downs))
        
    for dilations in all_dilations.values():
        all_configs.append((base_k, dilations, base_res, base_downs))
    
    for downs in all_downs:
        all_configs.append((base_k, base_dil, base_res, downs))
    return all_configs

def ciclo_treino_transform(configs:list, seed:int = 0, data="VessMAP", model="dic_unet", transform=None, variation=None):

    for index, config in enumerate(configs):
        # PATHS CONFIGURATION
        config_id = f"config_{index:03d}"
        transform_dir = "baseline" if transform is None else transform
        variation_dir = "baseline" if variation is None else variation
        base_path = f"../experiments/{data}/{transform_dir}/{variation_dir}/{config_id}"
        os.makedirs(base_path, exist_ok=True)
        
        # DATASET CONFIGURATION
        csv_train = f"../data/{data}/train.csv"
        im_size = 256 if data == "VessMAP" else 512
        in_c = 1 if data == "VessMAP" else 3
        
        # MODEL CONFIGURATION
        dic_kernel = str(config)
        
        # TRAINING MODELS
        cmd = f"""
        python train.py \
            --csv_train {csv_train} \
            --seed {seed} \
            --save_path {base_path} \
            --model_id {config_id} \
            --im_size {im_size} \
            --in_c {in_c} \
            --model_name {model} \
            --kernels "{dic_kernel}" \
            --transform "{transform}" \
            --variation "{variation}" \
            --epochs 2500
        """
        os.system(cmd)
        
        
        # GENERATE PREDICTIONS
        config_file = osp.join(base_path, f"{config_id}_{seed}.cfg")
        result_path = osp.join(base_path, f"results_{seed}")
        cmd = f"""
        python generate_results.py \
        --config_file {config_file} \
        --dataset {data} \
        --seed {seed} \
        --im_size {im_size} \
        --in_c {in_c} \
        --result_path {result_path}"""
        os.system(cmd)
        
        
        # ANALYZE RESULTS
        cmd = f"""
        python analyze_results.py \
        --path_preds {result_path} \
        --dataset {data} \
        --seed {seed}"""
        os.system(cmd)
        
        

        
        
def main():
    # LAYERS CONFIGURATION
    n_layers = 10 # only convolutional layers
    base_k = [3]*n_layers
    k_vals = [1, 5, 7]

    base_dil = [1]*n_layers
    dil_vals = [2, 3, 7]

    base_res = [1]*5 
    res_vals = [3]

    standard_config = [[(3, 3, 3, 3, 3, 3, 3, 3, 3, 3), (1, 1, 1, 1, 1, 1, 1, 1, 1, 1), (1, 1, 1, 1, 1), (2, 2)]]
    configs = generate_models_configurations(base_k, base_dil, base_res, k_vals, dil_vals, res_vals, return_base = False)

    for seed in [2,4,8]:
        ciclo_treino_transform(configs, seed=seed, data="VessMAP", model="dic_unet")
        #ciclo_treino_transform(configs, seed=seed, data="DRIVE", model="dic_unet")
        #ciclo_treino_transform(standard_config, seed=2, data="VessMAP", model="dic_unet", transform="erosion")
        #ciclo_treino_transform(standard_config, seed=2, data="VessMAP", model="dic_unet", transform="skeleton")


main()