import numpy as np
from my_env import *


# data_path="data/ucihar/train_prev/Inertial Signals/body_acc_x_train.txt"
# data_list=np.loadtxt(data_path,dtype=float)
# print(data_list.shape)

def shuffle_feature_and_label(_feature_list,_label_list):
    # feature list and label list is in the order of subjects
    feature_list_len=len(_feature_list)
    label_list_len=len(_label_list)
    assert feature_list_len==label_list_len
    id_list=list(range(feature_list_len))
    print("id_list: ",id_list)
    np.random.shuffle(id_list)
    shuffled_feature_list=_feature_list[id_list]
    shuffled_label_list=_label_list[id_list]
    return shuffled_feature_list,shuffled_label_list

a_list=np.array([200,300,400])
b_list=np.array([2,3,4])
a_new,b_new=shuffle_feature_and_label(a_list,b_list)
print(a_new)
print(b_new)


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



