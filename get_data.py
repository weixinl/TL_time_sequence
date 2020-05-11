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
