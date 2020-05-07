
import utils
from my_env import *
from config import config_dict 
import models
from tasks import *



if __name__ == "__main__":
    torch.manual_seed(10)
    subject_num_each_group=6
    group_num=UCIHAR_SUBJECT_NUM//subject_num_each_group
#     for tar_group_id in range(group_num):
#          baseline_task(tar_group_id)
    tar_group_id=4

    # use part of tar set to finetune
    # for tar_group_id in range(group_num):
    #     finetune_task(tar_group_id)
    # for tar_group_id in range(group_num):
    #     baseline_model_test(tar_group_id)
    # baseline_model_test(tar_group_id)
    # two_step_transfer_task(0)
    # test_task()
    # transfer_share_encoder_task(tar_group_id)
    transfer_with_reconstruct_task(tar_group_id)


