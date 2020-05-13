import utils
from my_env import *
import my_config
# from network import *
import models
import savers
import get_data

def baseline_task(_tar_group_id):
    cwd_abs_path=os.path.abspath(os.path.dirname(__file__))
    subject_num_each_group=6
    group_num=UCIHAR_SUBJECT_NUM//subject_num_each_group

    groups_dir=cwd_abs_path+"/data/ucihar/splitted/each_group_"+str(subject_num_each_group)+"_subject"

    log_save_dir=cwd_abs_path+"/results/log/baseline"
    utils.os_check_dir(log_save_dir)
    log_save_path=log_save_dir+"/log_tar_group_id_"+str(_tar_group_id)+".csv"

    best_model_info_dir=cwd_abs_path+"/results/best_model_infos/baseline"
    utils.os_check_dir(best_model_info_dir)
    best_model_info_path=best_model_info_dir+"/best_model_info_tar_group_id_"+str(_tar_group_id)+".csv"

    best_model_dir=cwd_abs_path+"/results/best_models/baseline"
    utils.os_check_dir(best_model_dir)
    best_model_path=best_model_dir+"/best_model_tar_group_id_"+str(_tar_group_id)+".pkl"
    
    src_train_dataloader,src_valid_dataloader,tar_dataloader=utils.get_dataloader(groups_dir,_tar_group_id,group_num)
    print("size of src_train_dataloader: "+str(len(src_train_dataloader.dataset)))
    saver=savers.Baseline_Saver(log_save_path,best_model_info_path,best_model_path)
    
    baseline_obj=models.Baseline(config_dict,saver)
    baseline_obj.train(src_train_dataloader,src_valid_dataloader,tar_dataloader)
    baseline_obj.save_log()

def baseline_model_test(_tar_group_id):
    cwd_abs_path=os.path.abspath(os.path.dirname(__file__))
    subject_num_each_group=6
    group_num=UCIHAR_SUBJECT_NUM//subject_num_each_group

    groups_dir=cwd_abs_path+"/data/ucihar/splitted/each_group_"+str(subject_num_each_group)+"_subject"
    group_dir=groups_dir+"/group_"+str(_tar_group_id)

    feature_b_path=group_dir+"/feature_b.txt"
    label_b_path=group_dir+"/label_b.txt"
    domain_b_path=group_dir+"/domain_b.txt"
    feature_b_list=np.loadtxt(feature_b_path,dtype=np.float)
    label_b_list=np.loadtxt(label_b_path,dtype=np.int)
    domain_b_list=np.loadtxt(domain_b_path,dtype=np.int)
    tar_b_zipped_dataset=utils.Zipped_Dataset(feature_b_list,label_b_list,domain_b_list)
    tar_b_dataloader=utils.zipped_dataset_to_dataloader(tar_b_zipped_dataset)



    best_model_dir=cwd_abs_path+"/results/best_models/baseline"
    best_model_path=best_model_dir+"/best_model_tar_group_id_"+str(_tar_group_id)+".pkl"
    
    
    baseline_obj=models.Baseline(config_dict,None)
    baseline_obj.test(best_model_path,tar_b_dataloader)


def finetune_task(_tar_group_id,_device_id=0):
    cwd_abs_path=os.path.abspath(os.path.dirname(__file__))
    subject_num_each_group=6
    group_num=UCIHAR_SUBJECT_NUM//subject_num_each_group

    prev_model_path=cwd_abs_path+"/results/best_models/baseline/best_model_tar_group_id_"+str(_tar_group_id)+".pkl"

    groups_dir=cwd_abs_path+"/data/ucihar/splitted/each_group_"+str(subject_num_each_group)+"_subject"
    group_dir=groups_dir+"/group_"+str(_tar_group_id)

    log_save_dir=cwd_abs_path+"/results/log/finetune"
    utils.os_check_dir(log_save_dir)
    log_save_path=log_save_dir+"/log_tar_group_id_"+str(_tar_group_id)+".csv"

    best_model_info_dir=cwd_abs_path+"/results/best_model_infos/finetune"
    utils.os_check_dir(best_model_info_dir)
    best_model_info_path=best_model_info_dir+"/best_model_info_tar_group_id_"+str(_tar_group_id)+".csv"

    best_model_dir=cwd_abs_path+"/results/best_models/finetune"
    utils.os_check_dir(best_model_dir)
    best_model_path=best_model_dir+"/best_model_tar_group_id_"+str(_tar_group_id)+".pkl"
    
    _,src_valid_zipped_dataset,_ =utils.get_zipped_dataset(groups_dir,_tar_group_id,group_num)
    src_valid_dataloader=utils.zipped_dataset_to_dataloader(src_valid_zipped_dataset)

    feature_a_path=group_dir+"/feature_a.txt"
    label_a_path=group_dir+"/label_a.txt"
    domain_a_path=group_dir+"/domain_a.txt"
    tar_a_zipped_dataset=utils.load_zipped_dataset(feature_a_path,label_a_path,domain_a_path)
    tar_a_dataloader=utils.zipped_dataset_to_dataloader(tar_a_zipped_dataset)

    feature_b_path=group_dir+"/feature_b.txt"
    label_b_path=group_dir+"/label_b.txt"
    domain_b_path=group_dir+"/domain_b.txt"
    tar_b_zipped_dataset=utils.load_zipped_dataset(feature_b_path,label_b_path,domain_b_path)
    tar_b_dataloader=utils.zipped_dataset_to_dataloader(tar_b_zipped_dataset)

    train_dataloader=tar_a_dataloader
    valid_dataloader=src_valid_dataloader
    test_dataloader=tar_b_dataloader

    saver=savers.Finetune_Saver(log_save_path,best_model_info_path,best_model_path)
    
    finetune_obj=models.Finetune(config_dict,saver,prev_model_path,_device_id)
    finetune_obj.train(train_dataloader,valid_dataloader,test_dataloader)
    finetune_obj.save_results()

def finetune_model_test(_tar_group_id,_device_id=0):
    cwd_abs_path=os.path.abspath(os.path.dirname(__file__))
    subject_num_each_group=6
    group_num=UCIHAR_SUBJECT_NUM//subject_num_each_group

    prev_model_path=cwd_abs_path+"/results/best_models/baseline/best_model_tar_group_id_"+str(_tar_group_id)+".pkl"

    groups_dir=cwd_abs_path+"/data/ucihar/splitted/each_group_"+str(subject_num_each_group)+"_subject"
    group_dir=groups_dir+"/group_"+str(_tar_group_id)

    log_save_dir=cwd_abs_path+"/results/log/finetune"
    utils.os_check_dir(log_save_dir)
    log_save_path=log_save_dir+"/log_tar_group_id_"+str(_tar_group_id)+".csv"

    best_model_info_dir=cwd_abs_path+"/results/best_model_infos/finetune"
    utils.os_check_dir(best_model_info_dir)
    best_model_info_path=best_model_info_dir+"/best_model_info_tar_group_id_"+str(_tar_group_id)+".csv"

    best_model_dir=cwd_abs_path+"/results/best_models/finetune"
    utils.os_check_dir(best_model_dir)
    best_model_path=best_model_dir+"/best_model_tar_group_id_"+str(_tar_group_id)+".pkl"
    
    feature_b_path=group_dir+"/feature_b.txt"
    label_b_path=group_dir+"/label_b.txt"
    domain_b_path=group_dir+"/domain_b.txt"
    tar_b_zipped_dataset=utils.load_zipped_dataset(feature_b_path,label_b_path,domain_b_path)
    tar_b_dataloader=utils.zipped_dataset_to_dataloader(tar_b_zipped_dataset)
    test_dataloader=tar_b_dataloader
    
    best_model_dir=cwd_abs_path+"/results/best_models/finetune"
    best_model_path=best_model_dir+"/best_model_tar_group_id_"+str(_tar_group_id)+".pkl"

    finetune_obj=models.Finetune(config_dict,None,prev_model_path,_device_id)
    finetune_obj.test(best_model_path,test_dataloader)

def transfer_task(_tar_group_id,_device_id=0):
    _device_id=_tar_group_id
    label_num=6
    cwd_abs_path=os.path.abspath(os.path.dirname(__file__))
    subject_num_each_group=6
    group_num=UCIHAR_SUBJECT_NUM//subject_num_each_group


    groups_dir=cwd_abs_path+"/data/ucihar/splitted/each_group_"+str(subject_num_each_group)+"_subject"
    group_dir=groups_dir+"/group_"+str(_tar_group_id)

    log_save_dir=cwd_abs_path+"/results/log/transfer"
    utils.os_check_dir(log_save_dir)
    log_save_path=log_save_dir+"/log_tar_group_id_"+str(_tar_group_id)+".csv"

    best_model_info_dir=cwd_abs_path+"/results/best_model_infos/transfer"
    utils.os_check_dir(best_model_info_dir)
    best_model_info_path=best_model_info_dir+"/best_model_info_tar_group_id_"+str(_tar_group_id)+".csv"

    best_model_dir=cwd_abs_path+"/results/best_models/transfer"
    utils.os_check_dir(best_model_dir)
    best_model_path=best_model_dir+"/best_model_tar_group_id_"+str(_tar_group_id)+".pkl"
    
    src_train_zipped_dataset ,src_valid_zipped_dataset,_ =utils.get_zipped_dataset(groups_dir,_tar_group_id,group_num)
    src_train_dataloader=utils.zipped_dataset_to_dataloader(src_train_zipped_dataset)
    src_valid_dataloader=utils.zipped_dataset_to_dataloader(src_valid_zipped_dataset)


    feature_b_path=group_dir+"/feature_b.txt"
    label_b_path=group_dir+"/label_b.txt"
    domain_b_path=group_dir+"/domain_b.txt"
    tar_b_zipped_dataset=utils.load_zipped_dataset(feature_b_path,label_b_path,domain_b_path)
    tar_b_dataloader=utils.zipped_dataset_to_dataloader(tar_b_zipped_dataset)

    train_dataloader=src_train_dataloader
    valid_dataloader=src_valid_dataloader
    test_dataloader=tar_b_dataloader

    saver=savers.Transfer_Saver(log_save_path,best_model_info_path,best_model_path)
    
    transfer_obj=models.Transfer(config_dict,saver,group_num-1,label_num,_device_id)
    transfer_obj.train(train_dataloader,valid_dataloader,test_dataloader,_tar_group_id)
    transfer_obj.save_results()




def two_step_transfer_task(_tar_group_id,_device_id=0):
    _device_id=_tar_group_id
    label_num=6
    cwd_abs_path=os.path.abspath(os.path.dirname(__file__))
    subject_num_each_group=6
    group_num=UCIHAR_SUBJECT_NUM//subject_num_each_group


    groups_dir=cwd_abs_path+"/data/ucihar/splitted/each_group_"+str(subject_num_each_group)+"_subject"
    group_dir=groups_dir+"/group_"+str(_tar_group_id)

    log_save_dir=cwd_abs_path+"/results/log/transfer"
    utils.os_check_dir(log_save_dir)
    log_save_path=log_save_dir+"/log_tar_group_id_"+str(_tar_group_id)+".csv"

    best_model_info_dir=cwd_abs_path+"/results/best_model_infos/transfer"
    utils.os_check_dir(best_model_info_dir)
    best_model_info_path=best_model_info_dir+"/best_model_info_tar_group_id_"+str(_tar_group_id)+".csv"

    best_model_dir=cwd_abs_path+"/results/best_models/transfer"
    utils.os_check_dir(best_model_dir)
    best_model_path=best_model_dir+"/best_model_tar_group_id_"+str(_tar_group_id)+".pkl"

    tmp_models_dir=cwd_abs_path+"/tmp_models"
    best_domain_branch_path=tmp_models_dir+"/best_domain_tar_group_id_"+str(_tar_group_id)+".pkl"
    
    src_train_zipped_dataset ,src_valid_zipped_dataset,_ =utils.get_zipped_dataset(groups_dir,_tar_group_id,group_num)
    src_train_dataloader=utils.zipped_dataset_to_dataloader(src_train_zipped_dataset)
    src_valid_dataloader=utils.zipped_dataset_to_dataloader(src_valid_zipped_dataset)


    feature_b_path=group_dir+"/feature_b.txt"
    label_b_path=group_dir+"/label_b.txt"
    domain_b_path=group_dir+"/domain_b.txt"
    tar_b_zipped_dataset=utils.load_zipped_dataset(feature_b_path,label_b_path,domain_b_path)
    tar_b_dataloader=utils.zipped_dataset_to_dataloader(tar_b_zipped_dataset)

    train_dataloader=src_train_dataloader
    # print("train dataloader shape:")
    # print(train_dataloader.dataset.feature_list.shape)
    
    valid_dataloader=src_valid_dataloader
    test_dataloader=tar_b_dataloader

    
 

    transfer_obj=models.Two_Step_Transfer(_config_dict=config_dict,\
        _domain_num=group_num-1,\
        _label_num=label_num,_device_id=_device_id)

    # transfer_domain_branch_saver=savers.Transfer_Domain_Branch_Saver(best_domain_branch_path)
    # transfer_obj.domain_branch_pretrain(_train_dataloader=train_dataloader,_valid_dataloader=valid_dataloader,_tar_group_id=_tar_group_id,\
    #     _domain_branch_saver=transfer_domain_branch_saver)



    # src_domain_specific_a_dataloader_list,src_domain_specific_b_dataloader_list=get_domain_specific_dataloader_lists(_tar_group_id,group_num)
    tmp_models_dir=cwd_abs_path+"/tmp_models"
    label_subbranches_dir=tmp_models_dir+"/tar_group_"+str(_tar_group_id)
    # utils.os_check_dir(label_subbranches_dir)
    # for subbranch_id in range(group_num-1):
    #     subbranch_dataloader_a=src_domain_specific_a_dataloader_list[subbranch_id]
    #     subbranch_dataloader_b=src_domain_specific_b_dataloader_list[subbranch_id]
    #     label_subbranch_save_path=label_subbranches_dir+"/label_subbranch_"+str(subbranch_id)+".pkl"
    #     label_subbranch_saver=savers.Transfer_Label_Subbranch_Saver(label_subbranch_save_path)
    #     transfer_obj.domain_specific_label_subbranch_pretrain(subbranch_id,label_subbranch_saver,subbranch_dataloader_a,subbranch_dataloader_b,_tar_group_id)
    #     print("label subbranch "+str(subbranch_id)+":")
    #     transfer_obj.label_subbranch_saver.final_print()

    saver=savers.Transfer_Saver(log_save_path,best_model_info_path,best_model_path)
    transfer_obj.train_model(_pretrain_domain_branch_path=best_domain_branch_path,_pretrain_subbranches_dir=label_subbranches_dir,\
        _train_dataloader=train_dataloader,_valid_dataloader=valid_dataloader,_test_dataloader=test_dataloader,_tar_group_id=_tar_group_id,\
        _saver=saver)
    
def get_zipped_dataset_from_specific_group(_group_id,_item_num=100):
    cwd_abs_path=os.path.abspath(os.path.dirname(__file__))
    groups_dir=cwd_abs_path+"/data/ucihar/splitted/each_group_6_subject"
    group_dir=groups_dir+"/group_"+str(_group_id)

    feature_path=group_dir+"/feature_a.txt"
    label_path=group_dir+"/label_a.txt"
    domain_path=group_dir+"/domain_a.txt"

    feature_list=np.loadtxt(feature_path,dtype=np.float)
    label_list=np.loadtxt(label_path,dtype=np.int)
    domain_list=np.loadtxt(domain_path,dtype=np.int)

    feature_list_concat=feature_list[:]
    label_list_concat=label_list[:]
    domain_list_concat=domain_list[:]
    return utils.Zipped_Dataset(feature_list_concat,label_list_concat,domain_list_concat)

def test_task():
    _tar_group_id=4
    device_id=_tar_group_id
    label_num=6
    cwd_abs_path=os.path.abspath(os.path.dirname(__file__))
    subject_num_each_group=6
    group_num=UCIHAR_SUBJECT_NUM//subject_num_each_group

    groups_dir=cwd_abs_path+"/data/ucihar/splitted/each_group_"+str(subject_num_each_group)+"_subject"
    group_dir=groups_dir+"/group_"+str(_tar_group_id)

    log_save_dir=cwd_abs_path+"/results/log/transfer"
    utils.os_check_dir(log_save_dir)
    log_save_path=log_save_dir+"/log_tar_group_id_"+str(_tar_group_id)+".csv"

    best_model_info_dir=cwd_abs_path+"/results/best_model_infos/transfer"
    utils.os_check_dir(best_model_info_dir)
    best_model_info_path=best_model_info_dir+"/best_model_info_tar_group_id_"+str(_tar_group_id)+".csv"

    best_model_dir=cwd_abs_path+"/results/best_models/transfer"
    utils.os_check_dir(best_model_dir)
    best_model_path=best_model_dir+"/best_model_tar_group_id_"+str(_tar_group_id)+".pkl"

    tmp_models_dir=cwd_abs_path+"/tmp_models"
    best_domain_branch_path=tmp_models_dir+"/best_domain_branch.pkl"

    # groups_dir=cwd_abs_path+"/data/ucihar/splitted/each_group_"+str(subject_num_each_group)+"_subject"
    # group_2_dir=groups_dir+"/group_2"
    # group_3_dir=groups_dir+"/group_3"

    # feature_g2_path=group_2_dir+"/feature_a.txt"
    # label_g2_path=group_2_dir+"/label_a.txt"
    # domain_g2_path=group_2_dir+"/domain_a.txt"

    # feature_g3_path=group_3_dir+"/feature_a.txt"
    # label_g3_path=group_3_dir+"/label_a.txt"
    # domain_g3_path=group_3_dir+"/domain_a.txt"


    # tar_g2_zipped_dataset=utils.load_zipped_dataset(feature_g2_path,label_g2_path,domain_g2_path)
    # # tar_g2_dataloader=utils.zipped_dataset_to_dataloader(tar_g2_zipped_dataset)
    # tar_g3_zipped_dataset=utils.load_zipped_dataset(feature_g3_path,label_g3_path,domain_g3_path)
    zipped_dataset_group_0=get_zipped_dataset_from_specific_group(0)
    zipped_dataset_group_1=get_zipped_dataset_from_specific_group(1)
    zipped_dataset_group_2=get_zipped_dataset_from_specific_group(2)
    zipped_dataset_group_3=get_zipped_dataset_from_specific_group(3)
    zipped_dataset_group_4=get_zipped_dataset_from_specific_group(4)

    zipped_dataset_0_1=utils.merge_zipped_dataset(zipped_dataset_group_0,zipped_dataset_group_1)
    zipped_dataset_2_3=utils.merge_zipped_dataset(zipped_dataset_group_2,zipped_dataset_group_3)
    merged_zipped_dataset=utils.merge_zipped_dataset(zipped_dataset_0_1,zipped_dataset_2_3)
    merged_dataloader=utils.zipped_dataset_to_dataloader(merged_zipped_dataset)
    dataloader_3=utils.zipped_dataset_to_dataloader(zipped_dataset_group_3)
    # tar_g3_dataloader=utils.zipped_dataset_to_dataloader(tar_g3_zipped_dataset)

    train_dataloader=merged_dataloader
    # print("train dataloader shape:")
    # print(train_dataloader.dataset.feature_list.shape)
    
    valid_dataloader=dataloader_3
    test_dataloader=None

    transfer_domain_branch_saver=savers.Transfer_Domain_Branch_Saver(best_domain_branch_path)
    saver=savers.Transfer_Saver(log_save_path,best_model_info_path,best_model_path)
    transfer_obj=models.Two_Step_Transfer(_config_dict=config_dict,_domain_branch_saver=transfer_domain_branch_saver,_saver=saver,_domain_num=group_num-1,\
        _label_num=label_num,_device_id=device_id)

    transfer_obj.domain_branch_pretrain(train_dataloader,valid_dataloader,_tar_group_id)

    return

def transfer_share_encoder_task(_tar_group_id,_device_id=0):
    _device_id=_tar_group_id
    label_num=6
    cwd_abs_path=os.path.abspath(os.path.dirname(__file__))
    subject_num_each_group=6
    group_num=UCIHAR_SUBJECT_NUM//subject_num_each_group


    groups_dir=cwd_abs_path+"/data/ucihar/splitted/each_group_"+str(subject_num_each_group)+"_subject"
    group_dir=groups_dir+"/group_"+str(_tar_group_id)

    log_save_dir=cwd_abs_path+"/results/log/transfer"
    utils.os_check_dir(log_save_dir)
    log_save_path=log_save_dir+"/log_tar_group_id_"+str(_tar_group_id)+".csv"

    best_model_info_dir=cwd_abs_path+"/results/best_model_infos/transfer"
    utils.os_check_dir(best_model_info_dir)
    best_model_info_path=best_model_info_dir+"/best_model_info_tar_group_id_"+str(_tar_group_id)+".csv"

    best_model_dir=cwd_abs_path+"/results/best_models/transfer"
    utils.os_check_dir(best_model_dir)
    best_model_path=best_model_dir+"/best_model_tar_group_id_"+str(_tar_group_id)+".pkl"

    tmp_models_dir=cwd_abs_path+"/tmp_models"
    best_domain_branch_path=tmp_models_dir+"/best_domain_tar_group_id_"+str(_tar_group_id)+".pkl"
    
    src_train_zipped_dataset ,src_valid_zipped_dataset,_ =utils.get_zipped_dataset(groups_dir,_tar_group_id,group_num)
    src_train_dataloader=utils.zipped_dataset_to_dataloader(src_train_zipped_dataset)
    src_valid_dataloader=utils.zipped_dataset_to_dataloader(src_valid_zipped_dataset)


    feature_b_path=group_dir+"/feature_b.txt"
    label_b_path=group_dir+"/label_b.txt"
    domain_b_path=group_dir+"/domain_b.txt"
    tar_b_zipped_dataset=utils.load_zipped_dataset(feature_b_path,label_b_path,domain_b_path)
    tar_b_dataloader=utils.zipped_dataset_to_dataloader(tar_b_zipped_dataset)

    train_dataloader=src_train_dataloader
    valid_dataloader=src_valid_dataloader
    test_dataloader=tar_b_dataloader

    
    



    # transfer_domain_branch_saver=savers.Transfer_Domain_Branch_Saver(best_domain_branch_path)
    # transfer_obj.domain_branch_pretrain(_train_dataloader=train_dataloader,_valid_dataloader=valid_dataloader,_tar_group_id=_tar_group_id,\
    #     _domain_branch_saver=transfer_domain_branch_saver)

    src_domain_specific_a_dataloader_list,src_domain_specific_b_dataloader_list=get_domain_specific_dataloader_lists(_tar_group_id,group_num)
    tmp_models_dir=cwd_abs_path+"/tmp_models"
    label_subbranches_dir=tmp_models_dir+"/tar_group_"+str(_tar_group_id)
    utils.os_check_dir(label_subbranches_dir)

    label_branch_obj=models.Label_Branch_Pretrain_Share_Encoder(_config_dict=config_dict,\
        _domain_num=group_num-1,\
        _label_num=label_num,\
        _src_dataloader_a_list=src_domain_specific_a_dataloader_list,_src_dataloader_b_list=src_domain_specific_b_dataloader_list,\
        _label_subbranches_dir=label_subbranches_dir,_device_id=_device_id)
    label_branch_obj.initialize_pretrain_subbranches()
    label_branch_obj.subbranches_pretrain()

    saver=savers.Transfer_Saver(log_save_path,best_model_info_path,best_model_path)
    transfer_obj.train_model(_pretrain_domain_branch_path=best_domain_branch_path,_pretrain_subbranches_dir=label_subbranches_dir,\
        _train_dataloader=train_dataloader,_valid_dataloader=valid_dataloader,_test_dataloader=test_dataloader,_tar_group_id=_tar_group_id,\
        _saver=saver)

def transfer_with_reconstruct_task(_data_name,_config,_tar_group_id,_device_id=0):
    # _device_id=_tar_group_id
    label_num=int(_config["label_num"])
    group_num=int(_config["group_num"])
    cwd_abs_path=os.path.abspath(os.path.dirname(__file__))
    # subject_num_each_group=6
    # group_num=UCIHAR_SUBJECT_NUM//subject_num_each_group

    groups_dir=cwd_abs_path+"/data/"+_data_name
    # tar_group_dir=groups_dir+"/group_"+str(_tar_group_id)

    log_save_dir=cwd_abs_path+"/results/"+_data_name+"/log/transfer_with_reconstruct"
    utils.os_check_dir(log_save_dir)
    log_save_path=log_save_dir+"/log_tar_group_id_"+str(_tar_group_id)+".csv"

    best_model_info_dir=cwd_abs_path+"/results/"+_data_name+"/best_model_infos/transfer_with_reconstruct"
    utils.os_check_dir(best_model_info_dir)
    best_model_info_path=best_model_info_dir+"/best_model_info_tar_group_id_"+str(_tar_group_id)+".csv"

    best_model_dir=cwd_abs_path+"/results/"+_data_name+"/best_models/transfer_with_reconstruct"
    utils.os_check_dir(best_model_dir)
    best_model_path=best_model_dir+"/best_model_tar_group_id_"+str(_tar_group_id)+".pkl"

    tmp_models_dir=cwd_abs_path+"/tmp_models/"+_data_name
    best_domain_branch_path=tmp_models_dir+"/best_domain_tar_group_id_"+str(_tar_group_id)+".pkl"

    train_dataloader,valid_dataloader,test_dataloader=\
        get_data.get_dataloaders(groups_dir,group_num,_tar_group_id)
    
    transfer_domain_branch_pretrain_obj=models.Transfer_Domain_Branch_Pretrain(_config=_config,\
        _device_id=_device_id)

    transfer_domain_branch_saver=savers.Transfer_Domain_Branch_Saver(best_domain_branch_path)
    transfer_domain_branch_pretrain_obj.domain_branch_pretrain(_train_dataloader=train_dataloader,\
        _valid_dataloader=valid_dataloader,_tar_group_id=_tar_group_id,\
        _domain_branch_saver=transfer_domain_branch_saver)

    src_domain_specific_a_dataloader_list,src_domain_specific_b_dataloader_list=\
        get_data.get_domain_specific_dataloader_lists(groups_dir,group_num,_tar_group_id)
    tmp_models_dir=cwd_abs_path+"/tmp_models/"+"data_name"
    label_subbranches_dir=tmp_models_dir+"/tar_group_"+str(_tar_group_id)
    utils.os_check_dir(label_subbranches_dir)
    label_branch_pretrain_obj=models.Transfer_Label_Branch_Pretrain(_config,_device_id)
    label_branch_pretrain_obj.subbranches_pretrain(_train_dataloader_list=src_domain_specific_a_dataloader_list,_valid_dataloader_list=src_domain_specific_b_dataloader_list,\
        _subbranches_dir=label_subbranches_dir)

    saver=savers.Transfer_Saver(log_save_path,best_model_info_path,best_model_path)
    transfer_obj=models.Transfer_With_Reconstruct(_config=_config,_device_id=_device_id)
    transfer_obj.train_model(_pretrain_domain_branch_path=best_domain_branch_path,_pretrain_subbranches_dir=label_subbranches_dir,\
        _train_dataloader=train_dataloader,_valid_dataloader=valid_dataloader,_test_dataloader=test_dataloader,_tar_group_id=_tar_group_id,\
        _saver=saver)
    
def small_ucihar_task():
