from my_env import *
import utils


def get_zipped_datasets(_groups_dir,_group_num,_tar_group_id):

    src_a_zipped_dataset ,src_b_zipped_dataset=\
        utils.get_src_zipped_dataset(_groups_dir,_tar_group_id,_group_num)
    tar_group_dir=_groups_dir+"/group_"+str(_tar_group_id)
    tar_feature_path=tar_group_dir+"/feature.txt"
    tar_label_path=tar_group_dir+"/label.txt"
    tar_domain_path=tar_group_dir+"/domain.txt"
    tar_zipped_dataset=utils.load_zipped_dataset(tar_feature_path,tar_label_path,tar_domain_path)
    return src_a_zipped_dataset,src_b_zipped_dataset,tar_zipped_dataset



def get_dataloaders(_groups_dir,_group_num,_tar_group_id):
    src_a_zipped_dataset,src_b_zipped_dataset,tar_zipped_dataset=\
        get_zipped_datasets(_groups_dir,_group_num,_tar_group_id)
    train_dataloader=utils.zipped_dataset_to_dataloader(src_a_zipped_dataset)
    valid_dataloader=utils.zipped_dataset_to_dataloader(src_b_zipped_dataset)
    test_dataloader=utils.zipped_dataset_to_dataloader(tar_zipped_dataset)

    # print("train dataloader shape:")
    # print(train_dataloader.dataset.feature_list.shape)
    return train_dataloader,valid_dataloader,test_dataloader

def get_domain_specific_dataloader_lists(_groups_dir,_group_num,_tar_group_id):
    cwd_abs_path=os.path.abspath(os.path.dirname(__file__))
    src_group_id_list=[]
    for group_id in range(_group_num):
        if(group_id==_tar_group_id):
            continue
        src_group_id_list.append(group_id)
    src_domain_specific_a_dataloader_list=[]
    src_domain_specific_b_dataloader_list=[]
    for src_group_id in src_group_id_list:
        group_dir=_groups_dir+"/group_"+str(src_group_id)
        feature_a_path=group_dir+"/feature_a.txt"
        label_a_path=group_dir+"/label_a.txt"
        domain_a_path=group_dir+"/domain_a.txt"
        feature_b_path=group_dir+"/feature_b.txt"
        label_b_path=group_dir+"/label_b.txt"
        domain_b_path=group_dir+"/domain_b.txt"
        specific_group_a_zipped_dataset=utils.load_zipped_dataset(feature_a_path,label_a_path,domain_a_path)
        specific_group_b_zipped_dataset=utils.load_zipped_dataset(feature_b_path,label_b_path,domain_b_path)
        specific_group_a_dataloader=utils.zipped_dataset_to_dataloader(specific_group_a_zipped_dataset)
        specific_group_b_dataloader=utils.zipped_dataset_to_dataloader(specific_group_b_zipped_dataset)
        src_domain_specific_a_dataloader_list.append(specific_group_a_dataloader)
        src_domain_specific_b_dataloader_list.append(specific_group_b_dataloader)
    return src_domain_specific_a_dataloader_list,src_domain_specific_b_dataloader_list
