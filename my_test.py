import numpy as np
from my_env import *


data_path="data/ucihar/train_prev/Inertial Signals/body_acc_x_train.txt"
data_list=np.loadtxt(data_path,dtype=float)
print(data_list.shape)



# merged_feature_path="data/ucihar/merged/merged_feature.txt"
# merged_feature_list=np.loadtxt(merged_feature_path,dtype=np.float)
# print(merged_feature_list.shape)
# class My_Classifier(nn.Module):
#     def __init__(self,_domain_num,_device):
#         super(My_Classifier,self).__init__()

#         self.fc= nn.Sequential(
#             nn.Linear(in_features=9*128, out_features=_domain_num),
#             nn.ReLU()
#         )
#         self.softmax_obj=nn.Softmax(dim=1)
    
#     def forward(self,_in):
#         new_in=_in.reshape(-1,9*128)
#         out=self.fc(new_in)
#         out=self.softmax_obj(out)
#         return out



