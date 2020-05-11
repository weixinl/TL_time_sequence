from my_env import *
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from config import config_dict
from sklearn.preprocessing import StandardScaler

def os_check_dir(_dir):
    if not os.path.exists(_dir):
        os.makedirs(_dir)

def get_raw_feature_list(_feature_path_by_signal):
    feature_list = None
    entry_num=None
    for feature_spec_signal_path in _feature_path_by_signal:
        data_entries = np.loadtxt(feature_spec_signal_path, dtype=np.float)
        if feature_list is None:
            entry_num=len(data_entries)
            # vertical vector, height=entry_num
            feature_list = np.zeros((entry_num, 1))
        feature_list = np.hstack((feature_list, data_entries))
    feature_list = feature_list[:, 1:]

    # print(X.shape)
    return feature_list

def get_raw_label_list(_label_path):
    # class 0-5
    label_list = np.loadtxt(_label_path, dtype=np.int) - 1
    return label_list

def get_raw_subject_list(_subject_path):
    subject_list = np.loadtxt(_subject_path, dtype=np.int) - 1
    return subject_list


def onehot_to_label(y_onehot):
    a = np.argwhere(y_onehot == 1)
    return a[:, -1]


class data_loader(Dataset):
    def __init__(self, samples, labels, t):
        self.samples = samples
        self.labels = labels
        self.T = t

    def __getitem__(self, index):
        sample, target = self.samples[index], self.labels[index]
        if self.T:
            return self.T(sample), target
        else:
            return sample, target

    def __len__(self):
        return len(self.samples)

class My_Dataset(Dataset):
    def __init__(self,_feature_list,_label_list,_domain_list):
        self.feature_list=_feature_list
        self.label_list=_label_list
        self.domain_list=_domain_list
        # feature_size=len(self.feature_list)
        # print("feature size: "+str(feature_size))
        # label_size=len(self.label_list)
        # print("label size: "+str(label_size))
        # domain_size=len(self.domain_list)
        # print("domain size: "+str(domain_size))

    def __getitem__(self, _index):
        feature, label, domain = self.feature_list[_index], self.label_list[_index],self.domain_list[_index]
        return feature,label,domain

    def __len__(self):
        return len(self.label_list)

class Zipped_Dataset(object):
    def __init__(self,_feature_list,_label_list,_domain_list):
        self.feature_list=_feature_list
        self.label_list=_label_list
        self.domain_list=_domain_list
        self.size=len(self.label_list)

    def data_shuffle(self):
        id_list=list(range(self.size))
        np.random.shuffle(id_list)
        shuffled_feature_list=self.feature_list[id_list]
        shuffled_label_list=self.label_list[id_list]
        shuffled_domain_list=self.domain_list[id_list]
        self.feature_list=shuffled_feature_list
        self.label_list=shuffled_label_list
        self.domain_list=shuffled_domain_list
    
    def get_size(self):
        return self.size

    def save(self,_feature_path,_label_path,_domain_path):
        np.savetxt(_feature_path,self.feature_list)
        np.savetxt(_label_path,self.label_list,fmt="%d")
        np.savetxt(_domain_path,self.domain_list,fmt="%d")

def load_zipped_dataset(_feature_path,_label_path,_domain_path):
    feature_list=np.loadtxt(_feature_path,dtype=np.float)
    label_list=np.loadtxt(_label_path,dtype=np.int)
    domain_list=np.loadtxt(_domain_path,dtype=np.int)
    return Zipped_Dataset(feature_list,label_list,domain_list)

def split_zipped_dataset(_total_zipped_dataset,_a_ratio):
    total_size=_total_zipped_dataset.get_size()
    a_size=int(total_size*_a_ratio)
    tot_feature_list=_total_zipped_dataset.feature_list
    tot_label_list=_total_zipped_dataset.label_list
    tot_domain_list=_total_zipped_dataset.domain_list
    a_feature_list=tot_feature_list[:a_size]
    a_label_list=tot_label_list[:a_size]
    a_domain_list=tot_domain_list[:a_size]
    b_feature_list=tot_feature_list[a_size:]
    b_label_list=tot_label_list[a_size:]
    b_domain_list=tot_domain_list[a_size:]
    a_zipped_dataset=Zipped_Dataset(a_feature_list,a_label_list,a_domain_list)
    b_zipped_dataset=Zipped_Dataset(b_feature_list,b_label_list,b_domain_list)
    return a_zipped_dataset, b_zipped_dataset

def merge_zipped_dataset(_zipped_dataset_a,_zipped_dataset_b):
    new_feature_list=np.concatenate((_zipped_dataset_a.feature_list,_zipped_dataset_b.feature_list))
    new_domain_list=np.concatenate((_zipped_dataset_a.domain_list,_zipped_dataset_b.domain_list))
    new_label_list=np.concatenate((_zipped_dataset_a.label_list,_zipped_dataset_b.label_list))
    return Zipped_Dataset(new_feature_list,new_label_list,new_domain_list)
    

def normalize(x):
    x_min = x.min(axis=(0, 2, 3), keepdims=True)
    x_max = x.max(axis=(0, 2, 3), keepdims=True)
    x_norm = (x - x_min) / (x_max - x_min)
    return x_norm


def load(batch_size=64):
    x_train, y_train, x_test, y_test = load_data()
    x_train, x_test = x_train.reshape((-1, 9, 1, 128)), x_test.reshape((-1, 9, 1, 128))
    transform = None
    train_set = data_loader(x_train, y_train, transform)
    test_set = data_loader(x_test, y_test, transform)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

def merge_raw_dataset():
    dataset_folder = "data/"+config_dict['data_folder_raw'] + '/'
    
    raw_train_feature_path_by_signal = [dataset_folder + 'train/' + 'Inertial Signals/' + item + 'train.txt' for item in SIGNAL_TYPES]
    raw_test_feature_path_by_signal = [dataset_folder + 'test/' + 'Inertial Signals/' + item + 'test.txt' for item in SIGNAL_TYPES]
    raw_train_feature_list=get_raw_feature_list(raw_train_feature_path_by_signal)
    raw_test_feature_list=get_raw_feature_list(raw_test_feature_path_by_signal)
    raw_train_feature_list=np.array(raw_train_feature_list)
    raw_test_feature_list=np.array(raw_test_feature_list)
    # print(raw_train_feature_list.shape)
    # print(raw_test_feature_list.shape)
    merged_feature_list=np.concatenate((raw_train_feature_list,raw_test_feature_list))

    raw_train_label_path = dataset_folder + 'train/y_train.txt'
    raw_test_label_path = dataset_folder + 'test/y_test.txt'
    raw_train_label_list=get_raw_label_list(raw_train_label_path)
    raw_test_label_list=get_raw_label_list(raw_test_label_path)
    merged_label_list=np.concatenate((raw_train_label_list,raw_test_label_list))

    raw_train_subject_path=dataset_folder+"train/subject_train.txt"
    raw_test_subject_path=dataset_folder+"test/subject_test.txt"
    raw_train_subject_list=get_raw_subject_list(raw_train_subject_path)
    raw_test_subject_list=get_raw_subject_list(raw_test_subject_path)
    merged_subject_list=np.concatenate((raw_train_subject_list,raw_test_subject_list))
    
    merged_folder=dataset_folder+"merged"
    merged_feature_path=merged_folder+"/merged_feature.txt"
    merged_label_path=merged_folder+"/merged_label.txt"
    merged_subject_path=merged_folder+"/merged_subject.txt"

    np.savetxt(merged_feature_path,merged_feature_list)
    np.savetxt(merged_label_path,merged_label_list,fmt="%d")
    np.savetxt(merged_subject_path,merged_subject_list,fmt="%d")

def get_data_by_each_subject():
    cwd_abs_path=os.path.abspath(os.path.dirname(__file__))
    merged_dataset_folder=cwd_abs_path+"/data/ucihar/merged"
    merged_feature_path=merged_dataset_folder+"/merged_feature.txt"
    merged_label_path=merged_dataset_folder+"/merged_label.txt"
    merged_subject_path=merged_dataset_folder+"/merged_subject.txt"

    merged_feature_list=np.loadtxt(merged_feature_path, dtype=np.float)
    merged_label_list=np.loadtxt(merged_label_path,dtype=np.int)
    merged_subject_list=np.loadtxt(merged_subject_path,dtype=np.int)

    feature_list_by_subject=[]
    label_list_by_subject=[]
    for subject_id in range(UCIHAR_SUBJECT_NUM):
        feature_list_by_subject.append([])
        label_list_by_subject.append([])
    for feature,label,subject_id in zip(merged_feature_list,merged_label_list,merged_subject_list):
        feature_list_by_subject[subject_id].append(feature)
        label_list_by_subject[subject_id].append(label)
    return feature_list_by_subject,label_list_by_subject
    
def merge_lists(_lists):
    return np.concatenate(_lists,axis=0)
    # merged_list=[]
    # for _list in _lists:
    #     merged_list+=_list[:]
    # merged_list=np.array(merged_list)
    # return merged_list

def shuffle_feature_and_label(_feature_list,_label_list):
    # feature list and label list is in the order of subjects
    feature_list_len=len(_feature_list)
    label_list_len=len(_label_list)
    assert feature_list_len==label_list_len
    id_list=list(range(feature_list_len))
    np.random.shuffle(id_list)
    shuffled_feature_list=_feature_list[id_list]
    shuffled_label_list=_label_list[id_list]
    return shuffled_feature_list,shuffled_label_list

    
def split_dataset_by_subject_group(_valid_ratio=0.2,_subject_num_each_group=6):
    cwd_abs_path=os.path.abspath(os.path.dirname(__file__))
    splitted_dir=cwd_abs_path+"/data/ucihar/splitted/each_group_"+str(_subject_num_each_group)+"_subject"
    os_check_dir(splitted_dir)

    feature_list_by_subject,label_list_by_subject=get_data_by_each_subject()
    group_num=int(UCIHAR_SUBJECT_NUM/_subject_num_each_group)
    print("group_num: "+str(group_num))
    for group_id in range(group_num):
        src_tar_pair_id=group_id
        src_tar_pair_dir=splitted_dir+"/group_"+str(group_id)
        os_check_dir(src_tar_pair_dir)
        subject_id_left_range=group_id*_subject_num_each_group
        subject_id_right_range=(group_id+1)*_subject_num_each_group
        group_feature_list_by_sub=feature_list_by_subject[subject_id_left_range:subject_id_right_range]
        group_label_list_by_sub=label_list_by_subject[subject_id_left_range:subject_id_right_range]

        group_feature_list=merge_lists(group_feature_list_by_sub)
        group_label_list=merge_lists(group_label_list_by_sub)

        shuffled_feature_list,shuffled_label_list=shuffle_feature_and_label(group_feature_list,group_label_list)
        
        group_feature_path=src_tar_pair_dir+"/feature.txt"
        group_label_path=src_tar_pair_dir+"/label.txt"

        np.savetxt(group_feature_path,shuffled_feature_list)
        np.savetxt(group_label_path,shuffled_label_list,fmt="%d")

        # split to train and valid
        group_size=len(shuffled_label_list)
        valid_size=int(group_size*_valid_ratio)

        valid_feature_list=shuffled_feature_list[:valid_size]
        train_feature_list=shuffled_feature_list[valid_size:]
        valid_label_list=shuffled_label_list[:valid_size]
        train_label_list=shuffled_label_list[valid_size:]

        train_feature_path=src_tar_pair_dir+"/feature_train.txt"
        valid_feature_path=src_tar_pair_dir+"/feature_valid.txt"
        train_label_path=src_tar_pair_dir+"/label_train.txt"
        valid_label_path=src_tar_pair_dir+"/label_valid.txt"

        np.savetxt(train_feature_path,train_feature_list)
        np.savetxt(valid_feature_path,valid_feature_list)
        np.savetxt(train_label_path,train_label_list,fmt="%d")
        np.savetxt(valid_label_path,valid_label_list,fmt="%d")

def split_subject_group(_subject_num_each_group=6,_a_ratio=0.5):
    cwd_abs_path=os.path.abspath(os.path.dirname(__file__))
    splitted_dir=cwd_abs_path+"/data/ucihar/splitted/each_group_"+str(_subject_num_each_group)+"_subject"
    group_num=int(UCIHAR_SUBJECT_NUM/_subject_num_each_group)
    for group_id in range(group_num):
        src_tar_pair_id=group_id
        src_tar_pair_dir=splitted_dir+"/group_"+str(group_id)
        feature_path=src_tar_pair_dir+"/feature.txt"
        label_path=src_tar_pair_dir+"/label.txt"
        feature_list=np.loadtxt(feature_path, dtype=np.float)
        label_list=np.loadtxt(label_path,dtype=np.int)
        data_size=len(label_list)
        domain_list=np.zeros(data_size)
        domain_list.fill(group_id)
        zipped_dataset=Zipped_Dataset(feature_list,label_list,domain_list)
        zipped_dataset.data_shuffle()
        zipped_dataset_a,zipped_dataset_b=split_zipped_dataset(zipped_dataset,_a_ratio)

        feature_a_path=src_tar_pair_dir+"/feature_a.txt"
        label_a_path=src_tar_pair_dir+"/label_a.txt"
        domain_a_path=src_tar_pair_dir+"/domain_a.txt"
        feature_b_path=src_tar_pair_dir+"/feature_b.txt"
        label_b_path=src_tar_pair_dir+"/label_b.txt"
        domain_b_path=src_tar_pair_dir+"/domain_b.txt"
        zipped_dataset_a.save(feature_a_path,label_a_path,domain_a_path)
        zipped_dataset_b.save(feature_b_path,label_b_path,domain_b_path)


def get_src_zipped_dataset(_groups_dir,_tar_group_id,_group_num):
    # tar_group_dir=_groups_dir+"/group_"+str(_tar_group_id)
    # tar_feature_path=tar_group_dir+"/feature.txt"
    # tar_label_path=tar_group_dir+"/label.txt"
    # tar_feature_list=np.loadtxt(tar_feature_path,dtype=np.float)
    # tar_label_list=np.loadtxt(tar_label_path,dtype=np.int)
    # tar_size=len(tar_label_list)
    # tar_domain_list=np.zeros(tar_size,dtype=int)
    # tar_domain_list.fill(_tar_group_id)

    src_group_id_list=[]
    for group_id in range(_group_num):
        if(group_id==_tar_group_id):
            continue
        src_group_id_list.append(group_id)
    
    src_train_feature_list_by_group=[]
    src_train_label_list_by_group=[]
    src_train_domain_list_by_group=[]
    src_valid_feature_list_by_group=[]
    src_valid_label_list_by_group=[]
    src_valid_domain_list_by_group=[]

    for src_group_id in src_group_id_list:
        # print("src_group_id: "+str(src_group_id))
        
        group_dir=_groups_dir+"/group_"+str(src_group_id)
        group_train_feature_path=group_dir+"/feature_a.txt"
        group_train_label_path=group_dir+"/label_a.txt"
        group_valid_feature_path=group_dir+"/feature_b.txt"
        group_valid_label_path=group_dir+"/label_b.txt"

        group_train_feature_list=np.loadtxt(group_train_feature_path,dtype=np.float)
        group_train_label_list=np.loadtxt(group_train_label_path,dtype=np.int)
        group_train_size=len(group_train_label_list)
        # print("tmp group train size: "+str(group_train_size))
        group_train_domain_list=np.zeros(group_train_size)
        group_train_domain_list.fill(src_group_id)
        # print("group_train_size: "+str(group_train_size))
        # print(group_train_domain_list)
        src_train_feature_list_by_group.append(group_train_feature_list)
        src_train_label_list_by_group.append(group_train_label_list)
        src_train_domain_list_by_group.append(group_train_domain_list)

        group_valid_feature_list=np.loadtxt(group_valid_feature_path,dtype=np.float)
        group_valid_label_list=np.loadtxt(group_valid_label_path,dtype=np.int)
        group_valid_size=len(group_valid_label_list)
        group_valid_domain_list=np.zeros(group_valid_size)
        group_valid_domain_list.fill(src_group_id)
        src_valid_feature_list_by_group.append(group_valid_feature_list)
        src_valid_label_list_by_group.append(group_valid_label_list)
        src_valid_domain_list_by_group.append(group_valid_domain_list)

    src_train_feature_list=merge_lists(src_train_feature_list_by_group)
    src_train_label_list=merge_lists(src_train_label_list_by_group)
    src_train_domain_list=merge_lists(src_train_domain_list_by_group)
    src_valid_feature_list=merge_lists(src_valid_feature_list_by_group)
    src_valid_label_list=merge_lists(src_valid_label_list_by_group)
    src_valid_domain_list=merge_lists(src_valid_domain_list_by_group)

    # src_train_feature_list=src_train_feature_list.reshape((-1, 9, 1, 128))
    # src_valid_feature_list=src_valid_feature_list.reshape((-1, 9, 1, 128))
    # tar_feature_list=tar_feature_list.reshape((-1, 9, 1, 128))

    src_train_zipped_dataset=Zipped_Dataset(src_train_feature_list,src_train_label_list,src_train_domain_list)
    src_valid_zipped_dataset=Zipped_Dataset(src_valid_feature_list,src_valid_label_list,src_valid_domain_list)
    # tar_zipped_dataset=Zipped_Dataset(tar_feature_list,tar_label_list,tar_domain_list)
    src_train_zipped_dataset.data_shuffle()
    src_valid_zipped_dataset.data_shuffle()
    # tar_zipped_dataset.data_shuffle()

    return src_train_zipped_dataset,src_valid_zipped_dataset


def get_dataloader(_groups_dir,_tar_group_id,_group_num):
    tar_group_dir=_groups_dir+"/group_"+str(_tar_group_id)
    tar_feature_path=tar_group_dir+"/feature.txt"
    tar_label_path=tar_group_dir+"/label.txt"
    tar_feature_list=np.loadtxt(tar_feature_path,dtype=np.float)
    tar_label_list=np.loadtxt(tar_label_path,dtype=np.int)
    tar_size=len(tar_label_list)
    tar_domain_list=np.zeros(tar_size)
    tar_domain_list.fill(_tar_group_id)

    src_group_id_list=[]
    for group_id in range(_group_num):
        if(group_id==_tar_group_id):
            continue
        src_group_id_list.append(group_id)
    
    src_train_feature_list_by_group=[]
    src_train_label_list_by_group=[]
    src_train_domain_list_by_group=[]
    src_valid_feature_list_by_group=[]
    src_valid_label_list_by_group=[]
    src_valid_domain_list_by_group=[]

    for src_group_id in src_group_id_list:
        
        group_dir=_groups_dir+"/group_"+str(_tar_group_id)
        group_train_feature_path=group_dir+"/train_feature.txt"
        group_train_label_path=group_dir+"/train_label.txt"
        group_valid_feature_path=group_dir+"/valid_feature.txt"
        group_valid_label_path=group_dir+"/valid_label.txt"

        group_train_feature_list=np.loadtxt(group_train_feature_path,dtype=np.float)
        group_train_label_list=np.loadtxt(group_train_label_path,dtype=np.int)
        group_train_size=len(group_train_label_list)
        group_train_domain_list=np.zeros(group_train_size)
        group_train_domain_list.fill(src_group_id)
        # print("group_train_size: "+str(group_train_size))
        # print(group_train_domain_list)
        src_train_feature_list_by_group.append(group_train_feature_list)
        src_train_label_list_by_group.append(group_train_label_list)
        src_train_domain_list_by_group.append(group_train_domain_list)

        group_valid_feature_list=np.loadtxt(group_valid_feature_path,dtype=np.float)
        group_valid_label_list=np.loadtxt(group_valid_label_path,dtype=np.int)
        group_valid_size=len(group_valid_label_list)
        group_valid_domain_list=np.zeros(group_valid_size)
        group_valid_domain_list.fill(src_group_id)
        src_valid_feature_list_by_group.append(group_valid_feature_list)
        src_valid_label_list_by_group.append(group_valid_label_list)
        src_valid_domain_list_by_group.append(group_valid_domain_list)

    src_train_feature_list=merge_lists(src_train_feature_list_by_group)
    src_train_label_list=merge_lists(src_train_label_list_by_group)
    src_train_domain_list=merge_lists(src_train_domain_list_by_group)
    src_valid_feature_list=merge_lists(src_valid_feature_list_by_group)
    src_valid_label_list=merge_lists(src_valid_label_list_by_group)
    src_valid_domain_list=merge_lists(src_valid_domain_list_by_group)

    src_train_feature_list=src_train_feature_list.reshape((-1, 9, 1, 128))
    src_valid_feature_list=src_valid_feature_list.reshape((-1, 9, 1, 128))
    tar_feature_list=tar_feature_list.reshape((-1, 9, 1, 128))

    # src_train_dataset=My_Dataset(src_train_feature_list,src_train_label_list,src_train_domain_list)
    # src_valid_dataset=My_Dataset(src_valid_feature_list,src_valid_label_list,src_valid_domain_list)
    # tar_dataset=My_Dataset(tar_feature_list,tar_label_list,tar_domain_list)

    # src_train_dataloader = DataLoader(src_train_dataset, batch_size=config_dict["batch_size"], shuffle=True, drop_last=True)
    # src_valid_dataloader = DataLoader(src_valid_dataset, batch_size=config_dict["batch_size"], shuffle=True, drop_last=True)
    # tar_dataloader = DataLoader(tar_dataset, batch_size=config_dict["batch_size"], shuffle=False)
    src_train_dataloader=lists_to_dataloader(src_train_feature_list,src_train_label_list,src_train_domain_list)
    src_valid_dataloader=lists_to_dataloader(src_valid_feature_list,src_valid_label_list,src_valid_domain_list)
    tar_dataloader=lists_to_dataloader(tar_feature_list,tar_label_list,tar_domain_list)

    return src_train_dataloader,src_valid_dataloader,tar_dataloader

def lists_to_dataloader(_feature_list,_label_list,_domain_list):
    dataset=My_Dataset(_feature_list,_label_list,_domain_list)
    dataloader=DataLoader(dataset,batch_size=config_dict["batch_size"], shuffle=True, drop_last=True)
    return dataloader

def zipped_dataset_to_dataloader(_zipped_dataset):
    feature_list=_zipped_dataset.feature_list
    label_list=_zipped_dataset.label_list
    domain_list=_zipped_dataset.domain_list
    dataset=My_Dataset(feature_list,label_list,domain_list)
    dataloader=DataLoader(dataset,batch_size=config_dict["batch_size"], shuffle=True, drop_last=True)
    return dataloader

if __name__=="__main__":
    # merge_raw_dataset()
    # split_dataset_by_subject_group()
    split_subject_group(6,0.5)

