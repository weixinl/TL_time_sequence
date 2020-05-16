import os 
import numpy as np
import pandas as pd

data_name_list=["ucihar_small","emg_normal","emg_aggressive"]
data_abbrev__list=["HAR","EMGN","EMGA"]
group_num_list=[5,4,4]
har_group_num=5
emgn_group_num=4
emga_group_num=4

def plot_pretrain_domain_acc():
    cwd_abs_path=os.path.abspath(os.path.dirname(__file__))
    # har_acc_list=
    # emgn_acc=
    tar_name_list=[]
    for dataset_i in range(3):
        data_name=data_name_list[dataset_i]
        data_abbrev=data_abbrev_list[dataset_i]
        group_num=group_num_list[dataset_i]
        for group_i in range(group_num):
            tar_name=data_abbrev+str(group_i)
            tar_name_list.append(tar_name)

    print("tar_name_list:")
    print(tar_name_list)
        
    for dataset_i in range(3):
        data_name=data_name_list[dataset_i]
        data_abbrev=data_abbrev_list[dataset_i]
        group_num=group_num_list[dataset_i]
        domain_best_info_dir=cwd_abs_path+"/domain_pretrain_results/"+data_name+"/best_model_infos"
        for group_i in range(group_num):
            tar_info_path=domain_best_info_dir+"/best_domain_branch_info_tar_group_id_"+str(group_id)\
                +".csv"
            


