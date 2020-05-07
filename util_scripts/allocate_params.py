import numpy as np
import json

def os_check_dir(_dir):
    if not os.path.exists(_dir):
        os.makedirs(_dir)

task_name="baseline"
device_num=8
cwd_abs_path=os.path.abspath(os.path.dirname(__file__))

param_file_cnt=0

shells_folder=cwd_abs_path+"/../shells_by_device/"+task_name+"/"
allocated_params_folder=cwd_abs_path+"/../allocated_params/"+task_name+"/"


sh_files_by_device=[]
for device_id in range(device_num):
    sh_path=shells_folder+"cmd_device"+str(device_id)+".sh"
    sh_file=open(sh_path,"w")
    sh_file.write("source activate py37\n")
    sh_files_by_device.append(sh_file)

device_cnt=0

# for optimizer_name in optimizer_name_list:
#     for lr in lr_list:
#         device_id=device_cnt%device_num
#         # cmd_line="python test_fintech.py --task "+test_task+" --epoch_num "+str(epoch_num)+" --optimizer_name "+str(optimizer_name)\
#         #     +" --learning_rate "+str(lr)+" --params \'{\"momentum\":"+str(momentum) +"}\' --device_id "+str(device_id)+"\n"
#         # cmd_line="python ../algorithms/explicit/test_fintech.py " +" --epoch_num "+str(epoch_num)+" --optimizer_name sgd "\
#         #     +" --learning_rate "+str(lr)+" --momentum "+str(momentum)+" --coeff "+ str(coeff) +" --device_id "+str(device_id)+"\n"
#         cmd_line="python ../algorithms/explicit/test_fintech.py " +" --task "+task_name+" --epoch_num "+str(epoch_num)+" --optimizer_name adam "\
#             +" --learning_rate "+str(lr)+" --device_id "+str(device_id)+"\n"
#         sh_file=sh_files_by_device[device_id]
#         sh_file.write(cmd_line)
#         device_cnt+=1




for sh_file in sh_files_by_device:
    sh_file.close()





