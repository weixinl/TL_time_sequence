import torch.nn as nn
import torch.nn.functional as F
from my_env import *

class Base_Extractor(nn.Module):
    def __init__(self):
        super(Base_Extractor, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=9, out_channels=32, kernel_size=(1, 9)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2), stride=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(1, 9)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2), stride=2)
        )
        
        self.fc1 = nn.Sequential(
            nn.Linear(in_features=64 * 26, out_features=1000),
            nn.ReLU()
        )

    def forward(self,_feature):
        feature=self.conv1(_feature)
        feature=self.conv2(feature)
        feature = feature.reshape(-1, 64 * 26)
        feature=self.fc1(feature)
        return feature


class Label_Classifier(nn.Module):
    def __init__(self):
        super(Label_Classifier,self).__init__()

        self.fc2 = nn.Sequential(
            nn.Linear(in_features=1000, out_features=500),
            nn.ReLU()
        )
        self.fc3 = nn.Sequential(
            nn.Linear(in_features=500, out_features=6)
        )
    
    def forward(self,_feature):
        out = self.fc2(_feature)
        out = self.fc3(out)
        one_hot_label = F.softmax(out, dim=1)
        return one_hot_label

class Classify_Network(nn.Module):
    def __init__(self):
        super(Classify_Network,self).__init__()
        self.base_extractor=Base_Extractor()
        self.label_classifier=Label_Classifier()
    
    def forward(self,_feature):
        out=self.base_extractor(_feature)
        out=self.label_classifier(out)
        return out

class Domain_Branch(nn.Module):
    def __init__(self,_config,_device):   
        self.domain_num=_config["domain_num"]
        self.attr_num=_config["attr_num"]
        self.window_size=_config["window_size"]

        super(Domain_Branch,self).__init__()
        self.bn=nn.BatchNorm2d(self.attr_num)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=self.attr_num, out_channels=16, kernel_size=(1, 9)),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2), stride=2)
        )
        conv1_out_len=(self.window_size-8)/2
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=4, kernel_size=(1, 5)),
            nn.BatchNorm2d(4),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2), stride=2)
        )
        conv2_out_len=int((conv1_out_len-4)/2)
        self.fc_in_feature_num=4*conv2_out_len
        self.fc1 = nn.Sequential(
            nn.Linear(in_features=self.fc_in_feature_num, out_features=32),
            nn.ReLU()
        )
        self.fc2= nn.Sequential(
            nn.Linear(in_features=32, out_features=self.domain_num),
            nn.ReLU()
        )

    
    def forward(self,_in):


        out=self.bn(_in)
        out=self.conv1(out)
        out=self.conv2(out)

        out=out.reshape(-1,self.fc_in_feature_num)
        out=self.fc1(out)
        out=self.fc2(out)

        return out

class Transfer_Encoder(nn.Module):
    def __init__(self,_config):
        super(Transfer_Encoder,self).__init__()
        self.domain_num=_config["domain_num"]
        self.attr_num=_config["attr_num"]
        self.window_size=_config["window_size"]
        self.bn1=nn.BatchNorm2d(self.attr_num)
        # 9*1*128
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=self.attr_num, out_channels=32, kernel_size=(1, 9)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2), stride=2)
        )
        conv1_out_len=(self.window_size-8)/2
        # 32*1*60
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(1, 9)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2), stride=2)
        )
        conv2_out_len=int((conv1_out_len-8)/2)
        # 64*1*26
    
    def forward(self,_in):
        new_in=self.bn1(_in)
        out=self.conv1(new_in)
        out=self.conv2(out)
        return out
    
class Transfer_Decoder(nn.Module):
    def __init__(self,_config):
        super(Transfer_Decoder,self).__init__()
        self.attr_num=_config["attr_num"]
        self.convtranspose1=nn.Sequential(
            nn.Upsample(size=[1,52], mode='bilinear'),
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=(1, 9)),
            nn.ReLU()
        )
        self.convtranspose2=nn.Sequential(
            nn.Upsample(size=[1,120], mode='bilinear'),
            nn.ConvTranspose2d(in_channels=32, out_channels=self.attr_num, kernel_size=(1, 9)),
            nn.ReLU()
        )

    def forward(self,_in):
        # print("_in:")
        # print(_in)
        out=self.convtranspose1(_in)
        out=self.convtranspose2(out)
        return out

class Base_Label_Extractor(nn.Module):
    def __init__(self,_config):
        super(Base_Label_Extractor,self).__init__()
        label_num=_config["label_num"]
        self.fc1 = nn.Sequential(
            nn.Linear(in_features=64 * 26, out_features=label_num),
            nn.ReLU()
        )
        # self.softmax_obj=nn.Softmax(dim=1)

    def forward(self,_in):
        # return one-hot
        in_vec = _in.reshape(-1, 64 * 26)
        out=self.fc1(in_vec)
        # out=self.softmax_obj(out)
        return out

class Label_Subbranch_Share_Encoder(nn.Module):
    def __init__(self,_encoder,_label_num,_device):
        # encoder is shared by all labelled subbranch
        super(Label_Subbranch_Share_Encoder,self).__init__()
        self.encoder=_encoder
        self.decoder=Transfer_Decoder().to(_device)
        self.label_extractor=Base_Label_Extractor(_label_num)
        self.reconstruct_loss_obj=nn.MSELoss()
        self.parameters_without_encoder=list(self.decoder.parameters())+list(self.label_extractor.parameters())
        self.parameters_inclu_encoder=self.parameters_without_encoder+list(self.encoder.parameters())

    def forward(self,_in):
        encoded_channels=self.encoder(_in)
        label_prob_vec=self.label_extractor(encoded_channels)
        decoded_channels=self.decoder(encoded_channels)
        reconstruct_loss=self.reconstruct_loss_obj(_in,decoded_channels)
        
        return label_prob_vec,reconstruct_loss
    
    def get_parameters_without_encoder(self):
        return self.parameters_without_encoder
    
    def get_parameters_inclu_encoder(self):
        return self.parameters_inclu_encoder
    

class Label_Subbranch(nn.Module):
    def __init__(self,_config,_device):
        super(Label_Subbranch,self).__init__()
        self.encoder=Transfer_Encoder(_config).to(_device)
        self.label_extractor=Base_Label_Extractor(_config).to(_device)
        # self.reconstruct_loss_obj=nn.MSELoss()

    def forward(self,_in):
        encoded_channels=self.encoder(_in)

        label_prob_vec=self.label_extractor(encoded_channels)
        return label_prob_vec,encoded_channels


class Label_Branch_Without_Reconstruct(nn.Module):
    def __init__(self,_domain_num,_label_num,_device):
        super(Label_Branch,self).__init__()
        self.domain_num=_domain_num
        self.label_num=_label_num
        self.device=_device
        # self.encoder=Transfer_Encoder().to(_device)
        self.label_subbranch_list=[]
        self.label_subbranch_parameters=[]
        for i in range(_domain_num):
            label_subbranch=Label_Subbranch(_label_num,_device).to(_device)
            self.label_subbranch_parameters+=list(label_subbranch.parameters())
            self.label_subbranch_list.append(label_subbranch)
    
    def forward(self,_in,_one_hot_domain_weight_by_entry):
        # one_hot_domain_weight=torch.mean(_one_hot_domain_weight_by_entry,0)
        batch_size=int(list(_one_hot_domain_weight_by_entry.shape)[0])
        weighted_sum_one_hot_label_by_entry=torch.zeros([batch_size,self.label_num],dtype=float).to(self.device)
        # weighted_sum_reconstruct_loss=torch.zeros([1],dtype=float).to(self.device)

        for domain_id in range(self.domain_num):
            subbranch_one_hot_label_by_entry=self.label_subbranch_list[domain_id](_in)
            for entry_id in range(batch_size):
                domain_weight=_one_hot_domain_weight_by_entry[entry_id][domain_id]
                weighted_sum_one_hot_label_by_entry[entry_id]+=domain_weight*subbranch_one_hot_label_by_entry[entry_id]
            # weighted_sum_reconstruct_loss+=one_hot_domain_weight[domain_id]*reconstruct_loss

        return weighted_sum_one_hot_label_by_entry
    
    def get_parameters(self):
        all_parameters=self.label_subbranch_parameters
        return all_parameters

class Label_Branch_With_Reconstruct(nn.Module):
    def __init__(self,_config,_device):
        super(Label_Branch_With_Reconstruct,self).__init__()
        self.config=_config
        self.domain_num=_config["domain_num"]
        self.label_num=_config["label_num"]
        self.device=_device
        # self.encoder=Transfer_Encoder().to(_device)
        self.label_subbranch_list=[]
        self.label_subbranch_parameters=[]
        for i in range(self.domain_num):
            label_subbranch=Label_Subbranch(_config,_device).to(_device)
            self.label_subbranch_parameters+=list(label_subbranch.parameters())
            self.label_subbranch_list.append(label_subbranch)
    
    def forward(self,_in,_domain_prob_vec_by_entry):
        # one_hot_domain_weight=torch.mean(_one_hot_domain_weight_by_entry,0)
        batch_size=int(list(_domain_prob_vec_by_entry.shape)[0])
        weighted_sum_encoded_layer_by_entry=torch.zeros([batch_size,64,1,26],dtype=torch.float).to(self.device)
        weighted_sum_label_vec_by_entry=torch.zeros([batch_size,self.label_num],dtype=torch.float).to(self.device)
        # weighted_sum_reconstruct_loss=torch.zeros([1],dtype=float).to(self.device)

        for domain_id in range(self.domain_num):
            label_prob_vec_by_entry,encoded_layer_by_entry=self.label_subbranch_list[domain_id](_in)
            for entry_id in range(batch_size):
                domain_weight=_domain_prob_vec_by_entry[entry_id][domain_id]
                weighted_sum_encoded_layer_by_entry[entry_id]+=domain_weight*encoded_layer_by_entry[entry_id]
                weighted_sum_label_vec_by_entry[entry_id]+=domain_weight*label_prob_vec_by_entry[entry_id]
            # weighted_sum_reconstruct_loss+=one_hot_domain_weight[domain_id]*reconstruct_loss

        return weighted_sum_label_vec_by_entry.float(),weighted_sum_encoded_layer_by_entry.float()
    
    def get_parameters(self):
        all_parameters=self.label_subbranch_parameters
        return all_parameters

class Label_Branch_Share_Encoder(nn.Module):
    def __init__(self,_domain_num,_label_num,_device):
        super(Label_Branch_Share_Encoder,self).__init__()
        self.domain_num=_domain_num
        self.label_num=_label_num
        self.device=_device
        # self.encoder=Transfer_Encoder().to(_device)
        self.label_subbranch_list=[]
        self.label_subbranch_parameters=[]
        for i in range(_domain_num):
            label_subbranch=Label_Subbranch_Share_Encoder(_label_num,_device).to(_device)
            self.label_subbranch_parameters+=list(label_subbranch.parameters())
            self.label_subbranch_list.append(label_subbranch)
    
    def forward(self,_in,_one_hot_domain_weight_by_entry):
        # one_hot_domain_weight=torch.mean(_one_hot_domain_weight_by_entry,0)
        batch_size=int(list(_one_hot_domain_weight_by_entry.shape)[0])
        weighted_sum_one_hot_label_by_entry=torch.zeros([batch_size,self.label_num],dtype=float).to(self.device)
        # weighted_sum_reconstruct_loss=torch.zeros([1],dtype=float).to(self.device)

        for domain_id in range(self.domain_num):
            subbranch_one_hot_label_by_entry=self.label_subbranch_list[domain_id](_in)
            for entry_id in range(batch_size):
                domain_weight=_one_hot_domain_weight_by_entry[entry_id][domain_id]
                weighted_sum_one_hot_label_by_entry[entry_id]+=domain_weight*subbranch_one_hot_label_by_entry[entry_id]
            # weighted_sum_reconstruct_loss+=one_hot_domain_weight[domain_id]*reconstruct_loss

        return weighted_sum_one_hot_label_by_entry
    
    def get_parameters(self):
        all_parameters=self.label_subbranch_parameters
        return all_parameters



# class Transfer_Network(nn.Module):
#     def __init__(self,_domain_num,_label_num,_device):
#         super(Transfer_Network,self).__init__()
#         self.domain_num=_domain_num
#         self.label_num=_label_num
#         self.domain_branch=Domain_Branch(_domain_num,_device).to(_device)
#         self.label_branch=Label_Branch_With_Reconstruct(_domain_num,_label_num,_device).to(_device)
    
#     def forward(self,_in):
#         domain_weight_vec=self.domain_branch(_in)
#         # print("==========================")
#         # print("domain_weight_vec:")
#         # print(domain_weight_vec)
#         one_hot_label=self.label_branch(_in,domain_weight_vec)
#         return one_hot_label,domain_weight_vec
    
#     def get_parameters(self):
#         domain_branch_parameters=list(self.domain_branch.parameters())
#         label_branch_parameters=self.label_branch.get_parameters()
#         all_parameters=domain_branch_parameters+label_branch_parameters
#         return all_parameters
class Baseline_Network(nn.Module):
    def __init__(self,_config,_device):
        super(Baseline_Network,self).__init__()
        self.config=_config
        self.domain_num=_config["domain_num"]
        self.label_num=_config["label_num"]
        self.encoder=Transfer_Encoder(_config).to(_device)
        self.label_extractor=Base_Label_Extractor(_config).to(_device)
    
    def forward(self,_in):
        return self.label_extractor(self.encoder(_in))
    
class Transfer_Network_With_Reconstruct(nn.Module):
    def __init__(self,_config,_device):
        super(Transfer_Network_With_Reconstruct,self).__init__()
        self.config=_config
        self.domain_num=_config["domain_num"]
        self.label_num=_config["label_num"]
        self.domain_branch=Domain_Branch(_config,_device).to(_device)
        self.label_branch=Label_Branch_With_Reconstruct(_config,_device).to(_device)
        self.decoder=Transfer_Decoder(_config).to(_device)
        self.reconstruct_loss_obj=nn.MSELoss()
    
    def forward(self,_in):
        
        domain_weight_vec=self.domain_branch(_in)
        # print("==========================")
        # print("domain_weight_vec:")
        # print(domain_weight_vec)
        label_vec_predict,encoded_layer=self.label_branch(_in,domain_weight_vec)
        decoded_feature=self.decoder(encoded_layer)
        reconstruct_loss=self.reconstruct_loss_obj(decoded_feature,_in)
        return label_vec_predict,domain_weight_vec,reconstruct_loss
    
    def get_parameters(self):
        domain_branch_parameters=list(self.domain_branch.parameters())
        label_branch_parameters=self.label_branch.get_parameters()
        decoder_parameters=list(self.decoder.parameters())
        all_parameters=domain_branch_parameters+label_branch_parameters+decoder_parameters
        return all_parameters




