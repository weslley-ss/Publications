import os
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

def main():

    n_layers = 10 # only convolutional layers
    base_k = [3]*n_layers
    k_vals = [1, 5, 7]

    base_dil = [1]*n_layers
    dil_vals = [2, 3, 7]

    base_res = [1]*5 
    res_vals = [3, 5, 7]

    configs = generate_models_configurations(base_k, base_dil, base_res, k_vals, dil_vals, res_vals, return_base = False)
    print(configs)

main()