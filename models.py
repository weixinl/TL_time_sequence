from my_env import *
from network import *
import savers

def my_standardize(_feature_prev):
    means = _feature_prev.mean(dim=-1, keepdim=True)
    stds = _feature_prev.std(dim=-1, keepdim=True)
    feature_new = (_feature_prev - means) / stds
    return feature_new

class Baseline(object):
    def __init__(self,_config,_saver,_device_id=0):
        super(Baseline, self).__init__()
        self.epoch_num=_config["epoch_num"]
        self.lr=_config["lr"]
        self.momentum=_config["momentum"]
        self.batch_size=_config["batch_size"]
        self.attr_num=_config["attr_num"]
        self.window_size=_config["window_size"]
        self.saver=_saver
        
        self.device=torch.device('cuda:'+str(_device_id) if torch.cuda.is_available() else 'cpu')

        self.loss_obj=nn.CrossEntropyLoss()
        self.model=Baseline_Network(_config,self.device).to(self.device)
        self.optimizer = optim.SGD(params=self.model.parameters(), lr=self.lr, momentum=self.momentum)
    
    def train(self,_src_train_loader,_src_valid_loader,_tar_loader):
        src_train_set_size=len(_src_train_loader.dataset)
        src_valid_set_size=len(_src_valid_loader.dataset)
        tar_set_size=len(_tar_loader.dataset)
        src_train_batch_num=src_train_set_size//self.batch_size
        src_valid_batch_num=src_valid_set_size//self.batch_size
        tar_batch_num=tar_set_size//self.batch_size

        for epoch_id in range(self.epoch_num):
            self.model.train()
            src_train_acc_cnt=0
            total_src_train_loss = 0
            for index, (feature, label, domain) in enumerate(_src_train_loader):
                # a batch
                feature= feature.to(self.device).float()
                label = label.to(self.device).long()
                # prepare for convolutional layer
                feature = feature.view(-1, self.attr_num, 1, self.window_size)
                one_hot_label_predict = self.model(feature)
                # print("one_hot_label_predict: ")
                # print(one_hot_label_predict)
                loss = self.loss_obj(one_hot_label_predict, label)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                total_src_train_loss += loss.item()
                _, label_predict = torch.max(one_hot_label_predict, 1)
                src_train_acc_cnt += (label_predict == label).sum()
       
            src_train_acc = float(src_train_acc_cnt)/(src_train_batch_num*self.batch_size)
            src_train_loss=float(total_src_train_loss)/src_train_batch_num
            print('Epoch_id: [{}/{}], src_train loss: {:.4f}, src_train acc: {:.4f}'.format(epoch_id, self.epoch_num, src_train_loss, src_train_acc))

            # src_valid and tar
            self.model.eval()

            src_valid_acc_cnt=0
            with torch.no_grad():
                for index,(feature, label, domain) in enumerate(_src_valid_loader):
                    feature= feature.to(self.device).float()
                    label = label.to(self.device).long()
                    feature = feature.view(-1, self.attr_num, 1, self.window_size)
                    one_hot_label_predict = self.model(feature)
                    _, label_predict = torch.max(one_hot_label_predict, 1)
                    src_valid_acc_cnt+=(label_predict == label).sum()
                    
            src_valid_acc = float(src_valid_acc_cnt)/(src_valid_batch_num*self.batch_size)
            print('Epoch_id: [{}/{}], src_valid acc: {:.4f}'.format(epoch_id, self.epoch_num, src_valid_acc))

            tar_acc_cnt=0
            with torch.no_grad():
                for index,(feature, label, domain) in enumerate(_tar_loader):
                    feature= feature.to(self.device).float()
                    label = label.to(self.device).long()
                    feature = feature.view(-1, self.attr_num, 1, self.window_size)
                    one_hot_label_predict = self.model(feature)
                    _, label_predict = torch.max(one_hot_label_predict, 1)
                    tar_acc_cnt+=(label_predict == label).sum()
                    
            tar_acc = float(tar_acc_cnt)/(tar_batch_num*self.batch_size)
            print('Epoch_id: [{}/{}], tar acc: {:.4f}'.format(epoch_id, self.epoch_num, tar_acc))

            self.saver.add_classify_log(epoch_id,src_train_loss,src_train_acc,src_valid_acc,tar_acc,self.model)
    
    def save_log(self):
        self.saver.save_log()

    def test(self,_model_path,_test_dataloader):
        test_set_size=len(_test_dataloader.dataset)
        test_batch_num=test_set_size//self.batch_size

        self.model.load_state_dict(torch.load(_model_path))
        self.model.eval()
        test_acc_cnt=0
        with torch.no_grad():
            for index,(feature, label, domain) in enumerate(_test_dataloader):
                feature= feature.to(self.device).float()
                label = label.to(self.device).long()
                feature = feature.view(-1, 9, 1, 128)
                one_hot_label_predict = self.model(feature)
                _, label_predict = torch.max(one_hot_label_predict, 1)
                test_acc_cnt+=(label_predict == label).sum()
                
        test_acc = float(test_acc_cnt)/(test_batch_num*self.batch_size)
        print('test acc: {:.4f}'.format(test_acc))





class Finetune(object):
    def __init__(self,_config_dict,_saver,_prev_model_path,_device_id=0):
        super(Finetune, self).__init__()
        self.epoch_num=_config_dict["epoch_num"]
        self.lr=_config_dict["lr"]
        self.momentum=_config_dict["momentum"]
        self.batch_size=_config_dict["batch_size"]
        self.saver=_saver
        self.prev_model_path=_prev_model_path
        
        self.device=torch.device('cuda:'+str(_device_id) if torch.cuda.is_available() else 'cpu')

        self.loss_obj=nn.CrossEntropyLoss()
        

        self.prev_model=Classify_Network().to(self.device)
        self.prev_model.load_state_dict(torch.load(self.prev_model_path))
        self.model=Classify_Network().to(self.device)
        self.model.base_extractor=self.prev_model.base_extractor
        self.model.label_classifier=self.prev_model.label_classifier

        param_dict_list=[]
        param_dict_list+=[{"params":self.model.base_extractor.parameters(),"lr":self.lr/10}]
        param_dict_list+=[{"params":self.model.label_classifier.parameters(),"lr":self.lr}]

        self.optimizer = optim.SGD(params=param_dict_list, lr=self.lr, momentum=self.momentum)

        
    def train(self,_train_dataloader,_valid_dataloader,_test_dataloader):
        train_set_size=len(_train_dataloader.dataset)
        valid_set_size=len(_valid_dataloader.dataset)
        test_set_size=len(_test_dataloader.dataset)
        train_batch_num=train_set_size//self.batch_size
        valid_batch_num=valid_set_size//self.batch_size
        test_batch_num=test_set_size//self.batch_size

        for epoch_id in range(self.epoch_num):
            self.model.train()
            train_acc_cnt=0
            total_train_loss = 0
            for index, (feature, label, domain) in enumerate(_train_dataloader):
                # a batch
                feature= feature.to(self.device).float()
                label = label.to(self.device).long()
                # prepare for convolutional layer
                feature = feature.view(-1, 9, 1, 128)
                one_hot_label_predict = self.model(feature)
                # print("one_hot_label_predict: ")
                # print(one_hot_label_predict)
                loss = self.loss_obj(one_hot_label_predict, label)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                total_train_loss += loss.item()
                _, label_predict = torch.max(one_hot_label_predict, 1)
                train_acc_cnt += (label_predict == label).sum()
       
            train_acc = float(train_acc_cnt)/(train_batch_num*self.batch_size)
            train_loss=float(total_train_loss)/train_batch_num
            print('Epoch_id: [{}/{}], train loss: {:.4f}, train acc: {:.4f}'.format(epoch_id, self.epoch_num, train_loss, train_acc))

            # src_valid and tar
            self.model.eval()

            valid_acc_cnt=0
            with torch.no_grad():
                for index,(feature, label, domain) in enumerate(_valid_dataloader):
                    feature= feature.to(self.device).float()
                    label = label.to(self.device).long()
                    feature = feature.view(-1, 9, 1, 128)
                    one_hot_label_predict = self.model(feature)
                    _, label_predict = torch.max(one_hot_label_predict, 1)
                    valid_acc_cnt+=(label_predict == label).sum()
                    
            valid_acc = float(valid_acc_cnt)/(valid_batch_num*self.batch_size)
            print('Epoch_id: [{}/{}], valid acc: {:.4f}'.format(epoch_id, self.epoch_num, valid_acc))

            test_acc_cnt=0
            with torch.no_grad():
                for index,(feature, label, domain) in enumerate(_test_dataloader):
                    feature= feature.to(self.device).float()
                    label = label.to(self.device).long()
                    feature = feature.view(-1, 9, 1, 128)
                    one_hot_label_predict = self.model(feature)
                    _, label_predict = torch.max(one_hot_label_predict, 1)
                    test_acc_cnt+=(label_predict == label).sum()
                    
            test_acc = float(test_acc_cnt)/(test_batch_num*self.batch_size)
            print('Epoch_id: [{}/{}], test acc: {:.4f}'.format(epoch_id, self.epoch_num, test_acc))

            self.saver.add_classify_log(epoch_id,train_loss,train_acc,valid_acc,test_acc,self.model)
    
    def save_results(self):
        self.saver.save_results()

    def test(self,_model_path,_test_dataloader):
        test_set_size=len(_test_dataloader.dataset)
        test_batch_num=test_set_size//self.batch_size

        self.model.load_state_dict(torch.load(_model_path))
        self.model.eval()
        test_acc_cnt=0
        with torch.no_grad():
            for index,(feature, label, domain) in enumerate(_test_dataloader):
                feature= feature.to(self.device).float()
                label = label.to(self.device).long()
                feature = feature.view(-1, 9, 1, 128)
                one_hot_label_predict = self.model(feature)
                _, label_predict = torch.max(one_hot_label_predict, 1)
                test_acc_cnt+=(label_predict == label).sum()
                
        test_acc = float(test_acc_cnt)/(test_batch_num*self.batch_size)
        print('test acc: {:.4f}'.format(test_acc))



class Transfer(nn.Module):
    def __init__(self,_config_dict,_saver,_domain_num,_label_num,_device_id=0):
        super(Transfer, self).__init__()
        self.epoch_num=_config_dict["epoch_num"]
        self.lr=_config_dict["lr"]
        self.momentum=_config_dict["momentum"]
        self.batch_size=_config_dict["batch_size"]
        self.saver=_saver
        self.domain_num=_domain_num
        self.label_num=_label_num

        self.device=torch.device('cuda:'+str(_device_id) if torch.cuda.is_available() else 'cpu')

        self.cross_entropy_loss_obj=nn.CrossEntropyLoss()
        
        self.model=Transfer_Network(_domain_num,_label_num,self.device).to(self.device)
        self.optimizer = optim.SGD(params=self.model.parameters(), lr=self.lr, momentum=self.momentum)

        
    def train(self,_train_dataloader,_valid_dataloader,_test_dataloader,_tar_group_id):
        train_set_size=len(_train_dataloader.dataset)
        valid_set_size=len(_valid_dataloader.dataset)
        test_set_size=len(_test_dataloader.dataset)
        train_batch_num=train_set_size//self.batch_size
        valid_batch_num=valid_set_size//self.batch_size
        test_batch_num=test_set_size//self.batch_size

        for epoch_id in range(self.epoch_num):
            self.model.train()
            train_acc_cnt=0
            total_train_loss = 0
            for index, (feature, label, domain) in enumerate(_train_dataloader):
                # a batch
                feature=my_standardize(feature)
                feature= feature.to(self.device).float()
                label = label.to(self.device).long()
                for entry_id in range(self.batch_size):
                    domain_item=domain[entry_id]
                    if(domain_item>_tar_group_id):
                        domain[entry_id]-=1
                domain = domain.to(self.device).long()
                # print("domain:")
                # print(domain)
                # prepare for convolutional layer
                feature = feature.view(-1, 9, 1, 128)
                one_hot_label_predict,one_hot_domain_predict,reconstruct_loss = self.model(feature)
                # print("one_hot_label_predict: ")
                # print(one_hot_label_predict)
                # print("one_hot_domain_predict: ")
                # print(one_hot_domain_predict)
                label_loss=self.cross_entropy_loss_obj(one_hot_label_predict,label)
                domain_loss=self.cross_entropy_loss_obj(one_hot_domain_predict,domain)
                loss=label_loss+ 0.05*domain_loss+0*reconstruct_loss

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_train_loss += loss.item()
                _, label_predict = torch.max(one_hot_label_predict, 1)
                train_acc_cnt += (label_predict == label).sum()
       
            train_acc = float(train_acc_cnt)/(train_batch_num*self.batch_size)
            train_loss=float(total_train_loss)/train_batch_num
            print("================")
            print("label_loss: ",label_loss)
            print("domain_loss: ",domain_loss)
            print("reconstruct_loss: ",reconstruct_loss)
            print("one_hot_label_predict: ")
            print(one_hot_label_predict)
            print("label:")
            print(label)
            print("one_hot_domain_predict: ")
            print(one_hot_domain_predict)
            print("domain:")
            print(domain)
            print('Epoch_id: [{}/{}], train loss: {:.4f}, train acc: {:.4f}'.format(epoch_id, self.epoch_num, train_loss, train_acc))

            # src_valid and tar

                    
            valid_acc = self.eval_model(_valid_dataloader,valid_batch_num)
            print('Epoch_id: [{}/{}], valid acc: {:.4f}'.format(epoch_id, self.epoch_num, valid_acc))

            test_acc=self.eval_model(_test_dataloader,test_batch_num)
            print('Epoch_id: [{}/{}], test acc: {:.4f}'.format(epoch_id, self.epoch_num, test_acc))

            self.saver.add_classify_log(epoch_id,train_loss,train_acc,valid_acc,test_acc,self.model)

    
    def save_results(self):
        self.saver.save_results()

    def eval_model(self,_dataloader,_data_batch_num):
        self.model.eval()
        acc_cnt=0
        with torch.no_grad():
            for index,(feature, label, domain) in enumerate(_dataloader):
                feature=my_standardize(feature)
                feature= feature.to(self.device).float()
                label = label.to(self.device).long()
                feature = feature.view(-1, 9, 1, 128)
                one_hot_label_predict, _, _ = self.model(feature)
                _, label_predict = torch.max(one_hot_label_predict, 1)
                acc_cnt+=(label_predict == label).sum()
                
        acc = float(acc_cnt)/(_data_batch_num*self.batch_size)
        return acc

    def test_model(self,_model_path,_test_dataloader):
        test_set_size=len(_test_dataloader.dataset)
        test_batch_num=test_set_size//self.batch_size

        self.model.load_state_dict(torch.load(_model_path))
        self.model.eval()
        test_acc_cnt=0
        with torch.no_grad():
            for index,(feature, label, domain) in enumerate(_test_dataloader):
                feature= feature.to(self.device).float()
                label = label.to(self.device).long()
                feature = feature.view(-1, 9, 1, 128)
                one_hot_label_predict = self.model(feature)
                _, label_predict = torch.max(one_hot_label_predict, 1)
                test_acc_cnt+=(label_predict == label).sum()
                
        test_acc = float(test_acc_cnt)/(test_batch_num*self.batch_size)
        print('test acc: {:.4f}'.format(test_acc))


class Two_Step_Transfer(nn.Module):
    def __init__(self,_config_dict,_domain_num,_label_num,_device_id=0):
        super(Two_Step_Transfer, self).__init__()
        self.epoch_num=_config_dict["epoch_num"]
        self.lr=_config_dict["lr"]
        self.momentum=_config_dict["momentum"]
        self.batch_size=_config_dict["batch_size"]
        # self.domain_branch_saver=_domain_branch_saver
        # self.label_branch_saver=_label_branch_saver
        # self.saver=_saver
        self.domain_num=_domain_num
        self.label_num=_label_num

        self.device=torch.device('cuda:'+str(_device_id) if torch.cuda.is_available() else 'cpu')

        self.cross_entropy_loss_obj=nn.CrossEntropyLoss()
        
        self.model=Transfer_Network(_domain_num,_label_num,self.device).to(self.device)


    def domain_branch_pretrain(self,_train_dataloader,_valid_dataloader,_tar_group_id,_domain_branch_saver):
        self.domain_branch_saver=_domain_branch_saver
        print("domain branch pretrain: ")
        train_set_size=len(_train_dataloader.dataset)
        valid_set_size=len(_valid_dataloader.dataset)
        train_batch_num=train_set_size//self.batch_size
        valid_batch_num=valid_set_size//self.batch_size

        # domain_branch=self.model.domain_branch
        domain_branch=Domain_Branch(self.domain_num,self.device).to(self.device)
        # print("domain branch parameters: ")
        # print(list(domain_branch.parameters()))
        # print("domain branch state dict: ")
        # print(domain_branch.state_dict())
        domain_branch_optimizer=optim.SGD(params=domain_branch.parameters(), lr=self.lr, momentum=self.momentum)
        
        for epoch_id in range(self.epoch_num):
            domain_branch.train()
            domain_acc_cnt=0

            for index, (feature, label, domain) in enumerate(_train_dataloader):
                # a batch
              
                feature= feature.to(self.device).float()
                
                # print("domain:")
                # print(domain)

                # print("label:")
                # print(label)

                for entry_id in range(self.batch_size):
                    domain_item=domain[entry_id]
                    if(domain_item>_tar_group_id):
                        domain[entry_id]-=1
                domain = domain.to(self.device).long()

                feature = feature.view(-1, 9, 1, 128)
                one_hot_domain_predict = domain_branch(feature)

                # print("one_hot_domain_predict: ")
                # print(one_hot_domain_predict)

                domain_loss=self.cross_entropy_loss_obj(one_hot_domain_predict,domain)

                _, domain_predict = torch.max(one_hot_domain_predict, 1)
                domain_acc_cnt += (domain_predict == domain).sum()
                
                domain_branch_optimizer.zero_grad()
                domain_loss.backward()
                domain_branch_optimizer.step()
            
            domain_acc=float(domain_acc_cnt)/(train_batch_num*self.batch_size)
            
            print("one_hot_domain_predict: ")
            print(one_hot_domain_predict)
            print("actual domain:")
            print(domain)
            print("domain loss:")
            print(domain_loss)
            print("domain acc:")
            print(domain_acc)

            # src_valid and tar

            domain_branch.eval() 
            valid_domain_acc_cnt=0
            with torch.no_grad():
                for index,(feature, label, domain) in enumerate(_valid_dataloader):
                    # feature=my_standardize(feature)
                    # print("batch feature shape:")
                    # print(feature.shape)
                    feature= feature.to(self.device).float()
                    for entry_id in range(self.batch_size):
                        domain_item=domain[entry_id]
                        if(domain_item>_tar_group_id):
                            domain[entry_id]-=1
                    domain = domain.to(self.device).long()
                    feature = feature.view(-1, 9, 1, 128)
                    domain_probs_predict = domain_branch(feature)

                    _, domain_predict = torch.max(domain_probs_predict, 1)
                    valid_domain_acc_cnt += (domain_predict == domain).sum()
            valid_domain_acc=float(valid_domain_acc_cnt)/(valid_batch_num*self.batch_size)

            print('Epoch_id: [{}/{}], valid_domain_acc: {:.4f}'.format(epoch_id, self.epoch_num, valid_domain_acc))
            self.domain_branch_saver.add_log(valid_domain_acc,domain_branch,epoch_id)
        self.domain_branch_saver.final_print()
    

    def domain_specific_label_subbranch_pretrain(self,_subbranch_id,_label_subbranch_saver,_train_dataloader,_valid_dataloader,_tar_group_id):
        print("label branch pretrain: ")
        train_set_size=len(_train_dataloader.dataset)
        valid_set_size=len(_valid_dataloader.dataset)
        train_batch_num=train_set_size//self.batch_size
        valid_batch_num=valid_set_size//self.batch_size
        self.label_subbranch_saver=_label_subbranch_saver

        label_subbranch=Label_Subbranch(self.label_num,self.device).to(self.device)

        label_subbranch_optimizer=optim.SGD(params=label_subbranch.parameters(), lr=self.lr, momentum=self.momentum)
        
        for epoch_id in range(self.epoch_num):
            label_subbranch.train()
            train_label_acc_cnt=0

            for index, (feature, label, domain) in enumerate(_train_dataloader):
                # a batch

                feature= feature.to(self.device).float()
                label = label.to(self.device).long()
                

                feature = feature.view(-1, 9, 1, 128)
                label_prob_vec_predict = label_subbranch(feature)
                # print("label_prob_vec_predict:")
                # print(label_prob_vec_predict)
                # print("label:")
                # print(label)

                # print("one_hot_domain_predict: ")
                # print(one_hot_domain_predict)

                label_loss=self.cross_entropy_loss_obj(label_prob_vec_predict,label)

                _, label_predict = torch.max(label_prob_vec_predict, 1)
                train_label_acc_cnt += (label_predict == label).sum()
                
                label_subbranch_optimizer.zero_grad()
                label_loss.backward()
                label_subbranch_optimizer.step()

            
            train_label_acc=float(train_label_acc_cnt)/(train_batch_num*self.batch_size)

            print("train_label_acc:")
            print(train_label_acc)

            # src_valid and tar

            label_subbranch.eval() 
            valid_label_acc_cnt=0
            with torch.no_grad():
                for index,(feature, label, domain) in enumerate(_valid_dataloader):

                    feature= feature.to(self.device).float()
                    label = label.to(self.device).long()
                    feature = feature.view(-1, 9, 1, 128)
                    label_prob_vec_predict = label_subbranch(feature)
                    
                    _, label_predict = torch.max(label_prob_vec_predict, 1)
                    valid_label_acc_cnt += (label_predict == label).sum()
                    
            valid_label_acc=float(valid_label_acc_cnt)/(valid_batch_num*self.batch_size)

            print('Epoch_id: [{}/{}], valid_label_acc: {:.4f}'.format(epoch_id, self.epoch_num, valid_label_acc))
            self.label_subbranch_saver.add_log(valid_label_acc,label_subbranch,epoch_id)


    def train_model(self,_pretrain_domain_branch_path,_pretrain_subbranches_dir,_train_dataloader,_valid_dataloader,_test_dataloader,_tar_group_id,_saver):
        self.model.domain_branch.load_state_dict(torch.load(_pretrain_domain_branch_path))
        self.saver=_saver
        for subbranch_id in range(self.domain_num):
            pretrain_subbranch_model_path=_pretrain_subbranches_dir+"/label_subbranch_"+str(subbranch_id)+".pkl"
            self.model.label_branch.label_subbranch_list[subbranch_id].load_state_dict(torch.load(pretrain_subbranch_model_path))
        # print("model parameters: ")
        # print(list(self.model.parameters()))
        # print("label branch parameters:")
        # print(list(self.model.label_branch.parameters()))
        parameters=self.model.get_parameters()
        optimizer=optim.SGD(params=parameters, lr=self.lr/10, momentum=self.momentum)

        train_set_size=len(_train_dataloader.dataset)
        valid_set_size=len(_valid_dataloader.dataset)
        test_set_size=len(_test_dataloader.dataset)
        train_batch_num=train_set_size//self.batch_size
        valid_batch_num=valid_set_size//self.batch_size
        test_batch_num=test_set_size//self.batch_size

        # acc without furthur training
        initial_valid_acc = self.eval_model(_valid_dataloader,valid_batch_num)
        initial_test_acc=self.eval_model(_test_dataloader,test_batch_num)
        self.saver.initial_valid_acc=initial_valid_acc
        self.saver.initial_test_acc=initial_test_acc
        print("initial valid acc: "+str(initial_valid_acc))
        print("initial test acc: "+str(initial_test_acc))
        return

        for epoch_id in range(self.epoch_num):
            self.model.train()
            train_acc_cnt=0
            total_train_loss = 0
            for index, (feature, label, domain) in enumerate(_train_dataloader):

                # a batch
                feature=my_standardize(feature)
                feature= feature.to(self.device).float()
                label = label.to(self.device).long()
                # print("label:")
                # print(label)
                for entry_id in range(self.batch_size):
                    domain_item=domain[entry_id]
                    if(domain_item>_tar_group_id):
                        domain[entry_id]-=1
                domain = domain.to(self.device).long()
                # print("domain:")
                # print(domain)
                # prepare for convolutional layer
                feature = feature.view(-1, 9, 1, 128)
                one_hot_label_predict,one_hot_domain_predict = self.model(feature)
                # print("one_hot_label_predict: ")
                # print(one_hot_label_predict)
                # print("one_hot_domain_predict: ")
                # print(one_hot_domain_predict)
                label_loss=self.cross_entropy_loss_obj(one_hot_label_predict,label)
                domain_loss=self.cross_entropy_loss_obj(one_hot_domain_predict,domain)

                loss=label_loss+0.1*domain_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_train_loss += loss.item()
                _, label_predict = torch.max(one_hot_label_predict, 1)
                train_acc_cnt += (label_predict == label).sum()
       
            train_acc = float(train_acc_cnt)/(train_batch_num*self.batch_size)
            train_loss=float(total_train_loss)/train_batch_num
            print("================")
            # print("label_loss: ",label_loss)
            # print("domain_loss: ",domain_loss)
            # print("reconstruct_loss: ",reconstruct_loss)
            # print("one_hot_label_predict: ")
            # print(one_hot_label_predict)
            # print("label:")
            # print(label)
            # print("one_hot_domain_predict: ")
            # print(one_hot_domain_predict)
            # print("domain:")
            # print(domain)
            print('Epoch_id: [{}/{}], train loss: {:.4f}, train acc: {:.4f}'.format(epoch_id, self.epoch_num, train_loss, train_acc))

            # src_valid and tar

                    
            valid_acc = self.eval_model(_valid_dataloader,valid_batch_num)
            print('Epoch_id: [{}/{}], valid acc: {:.4f}'.format(epoch_id, self.epoch_num, valid_acc))

            test_acc=self.eval_model(_test_dataloader,test_batch_num)
            print('Epoch_id: [{}/{}], test acc: {:.4f}'.format(epoch_id, self.epoch_num, test_acc))

            self.saver.add_classify_log(epoch_id,train_loss,train_acc,valid_acc,test_acc,self.model)
        self.save_results()


    def save_results(self):
        self.saver.save_results()

    def eval_model(self,_dataloader,_data_batch_num):
        self.model.eval()
        acc_cnt=0
        with torch.no_grad():
            for index,(feature, label, domain) in enumerate(_dataloader):
                feature= feature.to(self.device).float()
                label = label.to(self.device).long()
                feature = feature.view(-1, 9, 1, 128)
                one_hot_label_predict, _= self.model(feature)
                _, label_predict = torch.max(one_hot_label_predict, 1)
                acc_cnt+=(label_predict == label).sum()
                
        acc = float(acc_cnt)/(_data_batch_num*self.batch_size)
        return acc


class Transfer_Share_Encoder(nn.Module):
    def __init__(self,_config_dict,_domain_num,_label_num,_device_id=0):
        super(Two_Step_Transfer, self).__init__()
        self.epoch_num=_config_dict["epoch_num"]
        self.lr=_config_dict["lr"]
        self.momentum=_config_dict["momentum"]
        self.batch_size=_config_dict["batch_size"]
        # self.domain_branch_saver=_domain_branch_saver
        # self.label_branch_saver=_label_branch_saver
        # self.saver=_saver
        self.domain_num=_domain_num
        self.label_num=_label_num
        self.encoder=Transfer_Encoder()

        self.device=torch.device('cuda:'+str(_device_id) if torch.cuda.is_available() else 'cpu')

        self.cross_entropy_loss_obj=nn.CrossEntropyLoss()
        
        self.model=Transfer_Network(_domain_num,_label_num,self.device).to(self.device)
        self.pretrain_subbranch_list=[]




    def domain_branch_pretrain(self,_train_dataloader,_valid_dataloader,_tar_group_id,_domain_branch_saver):
        self.domain_branch_saver=_domain_branch_saver
        print("domain branch pretrain: ")
        train_set_size=len(_train_dataloader.dataset)
        valid_set_size=len(_valid_dataloader.dataset)
        train_batch_num=train_set_size//self.batch_size
        valid_batch_num=valid_set_size//self.batch_size

        # domain_branch=self.model.domain_branch
        domain_branch=Domain_Branch(self.domain_num,self.device).to(self.device)
        # print("domain branch parameters: ")
        # print(list(domain_branch.parameters()))
        # print("domain branch state dict: ")
        # print(domain_branch.state_dict())
        domain_branch_optimizer=optim.SGD(params=domain_branch.parameters(), lr=self.lr, momentum=self.momentum)
        
        for epoch_id in range(self.epoch_num):
            domain_branch.train()
            domain_acc_cnt=0

            for index, (feature, label, domain) in enumerate(_train_dataloader):
                # a batch
              
                feature= feature.to(self.device).float()
                
                # print("domain:")
                # print(domain)

                # print("label:")
                # print(label)

                for entry_id in range(self.batch_size):
                    domain_item=domain[entry_id]
                    if(domain_item>_tar_group_id):
                        domain[entry_id]-=1
                domain = domain.to(self.device).long()

                feature = feature.view(-1, 9, 1, 128)
                one_hot_domain_predict = domain_branch(feature)

                # print("one_hot_domain_predict: ")
                # print(one_hot_domain_predict)

                domain_loss=self.cross_entropy_loss_obj(one_hot_domain_predict,domain)

                _, domain_predict = torch.max(one_hot_domain_predict, 1)
                domain_acc_cnt += (domain_predict == domain).sum()
                
                domain_branch_optimizer.zero_grad()
                domain_loss.backward()
                domain_branch_optimizer.step()
            
            domain_acc=float(domain_acc_cnt)/(train_batch_num*self.batch_size)
            
            print("one_hot_domain_predict: ")
            print(one_hot_domain_predict)
            print("actual domain:")
            print(domain)
            print("domain loss:")
            print(domain_loss)
            print("domain acc:")
            print(domain_acc)

            # src_valid and tar

            domain_branch.eval() 
            valid_domain_acc_cnt=0
            with torch.no_grad():
                for index,(feature, label, domain) in enumerate(_valid_dataloader):
                    # feature=my_standardize(feature)
                    # print("batch feature shape:")
                    # print(feature.shape)
                    feature= feature.to(self.device).float()
                    for entry_id in range(self.batch_size):
                        domain_item=domain[entry_id]
                        if(domain_item>_tar_group_id):
                            domain[entry_id]-=1
                    domain = domain.to(self.device).long()
                    feature = feature.view(-1, 9, 1, 128)
                    domain_probs_predict = domain_branch(feature)

                    _, domain_predict = torch.max(domain_probs_predict, 1)
                    valid_domain_acc_cnt += (domain_predict == domain).sum()
            valid_domain_acc=float(valid_domain_acc_cnt)/(valid_batch_num*self.batch_size)

            print('Epoch_id: [{}/{}], valid_domain_acc: {:.4f}'.format(epoch_id, self.epoch_num, valid_domain_acc))
            self.domain_branch_saver.add_log(valid_domain_acc,domain_branch,epoch_id)
        self.domain_branch_saver.final_print()
    



    def train_model(self,_pretrain_domain_branch_path,_pretrain_subbranches_dir,_train_dataloader,_valid_dataloader,_test_dataloader,_tar_group_id,_saver):
        self.model.domain_branch.load_state_dict(torch.load(_pretrain_domain_branch_path))
        self.saver=_saver
        for subbranch_id in range(self.domain_num):
            pretrain_subbranch_model_path=_pretrain_subbranches_dir+"/label_subbranch_"+str(subbranch_id)+".pkl"
            self.model.label_branch.label_subbranch_list[subbranch_id].load_state_dict(torch.load(pretrain_subbranch_model_path))
        # print("model parameters: ")
        # print(list(self.model.parameters()))
        # print("label branch parameters:")
        # print(list(self.model.label_branch.parameters()))
        parameters=self.model.get_parameters()
        optimizer=optim.SGD(params=parameters, lr=self.lr/10, momentum=self.momentum)

        train_set_size=len(_train_dataloader.dataset)
        valid_set_size=len(_valid_dataloader.dataset)
        test_set_size=len(_test_dataloader.dataset)
        train_batch_num=train_set_size//self.batch_size
        valid_batch_num=valid_set_size//self.batch_size
        test_batch_num=test_set_size//self.batch_size

        # self.model.eval()

        for epoch_id in range(self.epoch_num):
            self.model.train()
            train_acc_cnt=0
            total_train_loss = 0
            for index, (feature, label, domain) in enumerate(_train_dataloader):

                # a batch
                feature=my_standardize(feature)
                feature= feature.to(self.device).float()
                label = label.to(self.device).long()
                # print("label:")
                # print(label)
                for entry_id in range(self.batch_size):
                    domain_item=domain[entry_id]
                    if(domain_item>_tar_group_id):
                        domain[entry_id]-=1
                domain = domain.to(self.device).long()
                # print("domain:")
                # print(domain)
                # prepare for convolutional layer
                feature = feature.view(-1, 9, 1, 128)
                one_hot_label_predict,one_hot_domain_predict = self.model(feature)
                # print("one_hot_label_predict: ")
                # print(one_hot_label_predict)
                # print("one_hot_domain_predict: ")
                # print(one_hot_domain_predict)
                label_loss=self.cross_entropy_loss_obj(one_hot_label_predict,label)
                domain_loss=self.cross_entropy_loss_obj(one_hot_domain_predict,domain)

                loss=label_loss+0.1*domain_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_train_loss += loss.item()
                _, label_predict = torch.max(one_hot_label_predict, 1)
                train_acc_cnt += (label_predict == label).sum()
       
            train_acc = float(train_acc_cnt)/(train_batch_num*self.batch_size)
            train_loss=float(total_train_loss)/train_batch_num
            print("================")
            # print("label_loss: ",label_loss)
            # print("domain_loss: ",domain_loss)
            # print("reconstruct_loss: ",reconstruct_loss)
            # print("one_hot_label_predict: ")
            # print(one_hot_label_predict)
            # print("label:")
            # print(label)
            # print("one_hot_domain_predict: ")
            # print(one_hot_domain_predict)
            # print("domain:")
            # print(domain)
            print('Epoch_id: [{}/{}], train loss: {:.4f}, train acc: {:.4f}'.format(epoch_id, self.epoch_num, train_loss, train_acc))

            # src_valid and tar

                    
            valid_acc = self.eval_model(_valid_dataloader,valid_batch_num)
            print('Epoch_id: [{}/{}], valid acc: {:.4f}'.format(epoch_id, self.epoch_num, valid_acc))

            test_acc=self.eval_model(_test_dataloader,test_batch_num)
            print('Epoch_id: [{}/{}], test acc: {:.4f}'.format(epoch_id, self.epoch_num, test_acc))

            self.saver.add_classify_log(epoch_id,train_loss,train_acc,valid_acc,test_acc,self.model)
        self.save_results()


    def save_results(self):
        self.saver.save_results()

    def eval_model(self,_dataloader,_data_batch_num):
        self.model.eval()
        acc_cnt=0
        with torch.no_grad():
            for index,(feature, label, domain) in enumerate(_dataloader):
                feature= feature.to(self.device).float()
                label = label.to(self.device).long()
                feature = feature.view(-1, 9, 1, 128)
                one_hot_label_predict, _= self.model(feature)
                _, label_predict = torch.max(one_hot_label_predict, 1)
                acc_cnt+=(label_predict == label).sum()
                
        acc = float(acc_cnt)/(_data_batch_num*self.batch_size)
        return acc



class Label_Branch_Pretrain_Share_Encoder(nn.Module):
    def __init__(self,_config_dict,_domain_num,_label_num,\
        _src_dataloader_a_list,_src_dataloader_b_list,\
        _encoder_model_dir,_label_subbranches_without_encoder_dir,_device_id=0):
        super(Label_Branch_Pretrain_Share_Encoder, self).__init__()
        self.epoch_num=_config_dict["epoch_num"]
        self.lr=_config_dict["lr"]
        self.momentum=_config_dict["momentum"]
        self.batch_size=_config_dict["batch_size"]
        # self.domain_branch_saver=_domain_branch_saver
        # self.label_branch_saver=_label_branch_saver
        # self.saver=_saver
        self.domain_num=_domain_num
        self.label_num=_label_num
        self.encoder=Transfer_Encoder()
        self.device=torch.device('cuda:'+str(_device_id) if torch.cuda.is_available() else 'cpu')
        self.label_subbranches_dir=_label_subbranches_dir

        self.cross_entropy_loss_obj=nn.CrossEntropyLoss()
        
        self.train_dataloader_list=_src_dataloader_a_list
        self.valid_dataloader_list=_src_dataloader_b_list
        self.pretrain_subbranch_list=[]
        self.subbranch_saver_list=[]
        self.subbranch_optimizer_list=[]

        
    # def initialize_pretrain_subbranches(self):

    #     for subbranch_id in range(self.domain_num):
    #         subbranch=Label_Subbranch_Share_Encoder(self.encoder,self.label_num,self.device).to(self.device)
    #         self.pretrain_subbranch_list.append(subbranch)
    #         subbranch_model_path=self.label_subbranches_dir+"/label_subbranch_"+str(subbranch_id)+".pkl"
    #         subbranch_saver=savers.Transfer_Label_Subbranch_Saver(subbranch_model_path)
    #         self.subbranch_saver_list.append(subbranch_saver)
    #         subbranch_params=subbranch.get_parameters_inclu_encoder()
    #         subbranch_optimizer = optim.SGD(params=subbranch_params, lr=self.lr, momentum=self.momentum)
    #         self.subbranch_optimizer_list.append(subbranch_optimizer)
    
    def subbranches_pretrain(self):
        for epoch_id in range(self.epoch_num):
            for subbranch_id in range(self.domain_num):
                train_dataloader=self.train_dataloader_list[subbranch_id]
                valid_dataloader=self.valid_dataloader_list[subbranch_id]
                subbranch=self.pretrain_subbranch_list[subbranch_id]
                subbranch_saver=self.subbranch_saver_list[subbranch_id]
                subbranch_optimizer=self.subbranch_optimizer_list[subbranch_id]
                self.subbranch_pretrain_one_step(_epoch_id=epoch_id,_subbranch_id=subbranch_id,_subbranch=subbranch,_subbranch_saver=subbranch_saver,\
                    _train_dataloader=train_dataloader,_valid_dataloader=valid_dataloader,\
                    _subbranch_optimizer=subbranch_optimizer)
        for subbranch_id in range(self.domain_num):
            subbranch_saver=self.subbranch_saver_list[subbranch_id]
            print("best model of subbranch "+str(subbranch_id))
            subbranch_saver.final_print()


    def subbranch_pretrain_one_step(self,_epoch_id,_subbranch_id,_subbranch,_subbranch_saver,_train_dataloader,_valid_dataloader,_subbranch_optimizer):
        print("label branch pretrain: ")
        train_set_size=len(_train_dataloader.dataset)
        valid_set_size=len(_valid_dataloader.dataset)
        train_batch_num=train_set_size//self.batch_size
        valid_batch_num=valid_set_size//self.batch_size
   
        _subbranch.train()
        train_label_acc_cnt=0

        for index, (feature, label, domain) in enumerate(_train_dataloader):
            # a batch

            feature= feature.to(self.device).float()
            label = label.to(self.device).long()
            
            feature = feature.view(-1, 9, 1, 128)
            label_prob_vec_predict,reconstruct_loss = _subbranch(feature)

            label_loss=self.cross_entropy_loss_obj(label_prob_vec_predict,label)
            subbranch_loss=label_loss+reconstruct_loss

            _subbranch_optimizer.zero_grad()
            subbranch_loss.backward()
            _subbranch_optimizer.step()

            _, label_predict = torch.max(label_prob_vec_predict, 1)
            train_label_acc_cnt += (label_predict == label).sum()
            
        
        train_label_acc=float(train_label_acc_cnt)/(train_batch_num*self.batch_size)

        print("train_label_acc:")
        print(train_label_acc)

        # src_valid and tar

        _subbranch.eval() 
        valid_label_acc_cnt=0
        with torch.no_grad():
            for index,(feature, label, domain) in enumerate(_valid_dataloader):

                feature= feature.to(self.device).float()
                label = label.to(self.device).long()
                feature = feature.view(-1, 9, 1, 128)
                label_prob_vec_predict,_ = _subbranch(feature)
                
                _, label_predict = torch.max(label_prob_vec_predict, 1)
                valid_label_acc_cnt += (label_predict == label).sum()
                
        valid_label_acc=float(valid_label_acc_cnt)/(valid_batch_num*self.batch_size)

        print('Epoch_id: [{}/{}], subbranch_id: {} valid_label_acc: {:.4f}'.format(_epoch_id, self.epoch_num,_subbranch_id, valid_label_acc))
        _subbranch_saver.add_log(valid_label_acc,_subbranch,_epoch_id)


    def train_model(self,_pretrain_domain_branch_path,_pretrain_subbranches_dir,_train_dataloader,_valid_dataloader,_test_dataloader,_tar_group_id,_saver):
        self.model.domain_branch.load_state_dict(torch.load(_pretrain_domain_branch_path))
        self.saver=_saver
        for subbranch_id in range(self.domain_num):
            pretrain_subbranch_model_path=_pretrain_subbranches_dir+"/label_subbranch_"+str(subbranch_id)+".pkl"
            self.model.label_branch.label_subbranch_list[subbranch_id].load_state_dict(torch.load(pretrain_subbranch_model_path))
        # print("model parameters: ")
        # print(list(self.model.parameters()))
        # print("label branch parameters:")
        # print(list(self.model.label_branch.parameters()))
        parameters=self.model.get_parameters()
        optimizer=optim.SGD(params=parameters, lr=self.lr/10, momentum=self.momentum)

        train_set_size=len(_train_dataloader.dataset)
        valid_set_size=len(_valid_dataloader.dataset)
        test_set_size=len(_test_dataloader.dataset)
        train_batch_num=train_set_size//self.batch_size
        valid_batch_num=valid_set_size//self.batch_size
        test_batch_num=test_set_size//self.batch_size

        # self.model.eval()

        for epoch_id in range(self.epoch_num):
            self.model.train()
            train_acc_cnt=0
            total_train_loss = 0
            for index, (feature, label, domain) in enumerate(_train_dataloader):

                # a batch
                feature=my_standardize(feature)
                feature= feature.to(self.device).float()
                label = label.to(self.device).long()
                # print("label:")
                # print(label)
                for entry_id in range(self.batch_size):
                    domain_item=domain[entry_id]
                    if(domain_item>_tar_group_id):
                        domain[entry_id]-=1
                domain = domain.to(self.device).long()
                # print("domain:")
                # print(domain)
                # prepare for convolutional layer
                feature = feature.view(-1, 9, 1, 128)
                one_hot_label_predict,one_hot_domain_predict = self.model(feature)
                # print("one_hot_label_predict: ")
                # print(one_hot_label_predict)
                # print("one_hot_domain_predict: ")
                # print(one_hot_domain_predict)
                label_loss=self.cross_entropy_loss_obj(one_hot_label_predict,label)
                domain_loss=self.cross_entropy_loss_obj(one_hot_domain_predict,domain)

                loss=label_loss+0.1*domain_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_train_loss += loss.item()
                _, label_predict = torch.max(one_hot_label_predict, 1)
                train_acc_cnt += (label_predict == label).sum()
       
            train_acc = float(train_acc_cnt)/(train_batch_num*self.batch_size)
            train_loss=float(total_train_loss)/train_batch_num
            print("================")
            # print("label_loss: ",label_loss)
            # print("domain_loss: ",domain_loss)
            # print("reconstruct_loss: ",reconstruct_loss)
            # print("one_hot_label_predict: ")
            # print(one_hot_label_predict)
            # print("label:")
            # print(label)
            # print("one_hot_domain_predict: ")
            # print(one_hot_domain_predict)
            # print("domain:")
            # print(domain)
            print('Epoch_id: [{}/{}], train loss: {:.4f}, train acc: {:.4f}'.format(epoch_id, self.epoch_num, train_loss, train_acc))

            # src_valid and tar

                    
            valid_acc = self.eval_model(_valid_dataloader,valid_batch_num)
            print('Epoch_id: [{}/{}], valid acc: {:.4f}'.format(epoch_id, self.epoch_num, valid_acc))

            test_acc=self.eval_model(_test_dataloader,test_batch_num)
            print('Epoch_id: [{}/{}], test acc: {:.4f}'.format(epoch_id, self.epoch_num, test_acc))

            self.saver.add_classify_log(epoch_id,train_loss,train_acc,valid_acc,test_acc,self.model)
        self.save_results()


    def save_results(self):
        self.saver.save_results()

    def eval_model(self,_dataloader,_data_batch_num):
        self.model.eval()
        acc_cnt=0
        with torch.no_grad():
            for index,(feature, label, domain) in enumerate(_dataloader):
                feature= feature.to(self.device).float()
                label = label.to(self.device).long()
                feature = feature.view(-1, 9, 1, 128)
                one_hot_label_predict, _= self.model(feature)
                _, label_predict = torch.max(one_hot_label_predict, 1)
                acc_cnt+=(label_predict == label).sum()
                
        acc = float(acc_cnt)/(_data_batch_num*self.batch_size)
        return acc

class Transfer_Domain_Branch_Pretrain(nn.Module):
    def __init__(self,_config,_device_id=0):
        super(Transfer_Domain_Branch_Pretrain, self).__init__()
        self.config=_config
        self.epoch_num=_config["domain_branch_epoch_num"]
        self.lr=_config["domain_branch_lr"]
        self.momentum=_config["domain_branch_momentum"]
        self.batch_size=_config["domain_branch_batch_size"]
        self.attr_num=_config["attr_num"]
        self.window_size=_config["window_size"]
        # self.domain_branch_saver=_domain_branch_saver
        # self.label_branch_saver=_label_branch_saver
        # self.saver=_saver
        self.domain_num=_config["domain_num"]
        # self.label_num=_label_num
        self.device=torch.device('cuda:'+str(_device_id) if torch.cuda.is_available() else 'cpu')
        self.cross_entropy_loss_obj=nn.CrossEntropyLoss()
    
    def domain_branch_pretrain(self,_train_dataloader,_valid_dataloader,_tar_group_id,_domain_branch_saver):
        self.domain_branch_saver=_domain_branch_saver
        print("domain branch pretrain: ")
        train_set_size=len(_train_dataloader.dataset)
        valid_set_size=len(_valid_dataloader.dataset)
        train_batch_num=train_set_size//self.batch_size
        valid_batch_num=valid_set_size//self.batch_size

        domain_branch=Domain_Branch(self.config,self.device).to(self.device)

        domain_branch_optimizer=optim.SGD(params=domain_branch.parameters(), lr=self.lr, momentum=self.momentum)
        
        for epoch_id in range(self.epoch_num):
            domain_branch.train()
            train_domain_acc_cnt=0

            for index, (feature, label, domain) in enumerate(_train_dataloader):
                # a batch
              
                feature= feature.to(self.device).float()

                for entry_id in range(self.batch_size):
                    domain_item=domain[entry_id]
                    if(domain_item>_tar_group_id):
                        domain[entry_id]-=1
                domain = domain.to(self.device).long()

                feature = feature.view(-1, self.attr_num, 1, self.window_size)
                domain_prob_vec_predict = domain_branch(feature)


                domain_loss=self.cross_entropy_loss_obj(domain_prob_vec_predict,domain)

                _, domain_predict = torch.max(domain_prob_vec_predict, 1)
                train_domain_acc_cnt += (domain_predict == domain).sum()
                
                domain_branch_optimizer.zero_grad()
                domain_loss.backward()
                domain_branch_optimizer.step()
            
            train_domain_acc=float(train_domain_acc_cnt)/(train_batch_num*self.batch_size)
            
            domain_branch.eval() 
            valid_domain_acc_cnt=0
            with torch.no_grad():
                for index,(feature, label, domain) in enumerate(_valid_dataloader):
                    feature= feature.to(self.device).float()
                    for entry_id in range(self.batch_size):
                        domain_item=domain[entry_id]
                        if(domain_item>_tar_group_id):
                            domain[entry_id]-=1
                    domain = domain.to(self.device).long()
                    feature = feature.view(-1, self.attr_num, 1,self.window_size)
                    domain_probs_predict = domain_branch(feature)
                    _, domain_predict = torch.max(domain_probs_predict, 1)
                    valid_domain_acc_cnt += (domain_predict == domain).sum()
            valid_domain_acc=float(valid_domain_acc_cnt)/(valid_batch_num*self.batch_size)

            print('Epoch_id: [{}/{}], valid_domain_acc: {:.4f}'.format(epoch_id, self.epoch_num, valid_domain_acc))
            self.domain_branch_saver.add_log(train_domain_acc,valid_domain_acc,domain_branch,epoch_id)

        self.domain_branch_saver.final_print()


class Transfer_Label_Branch_Pretrain(nn.Module):  
    def __init__(self,_config,_device_id=0):
        super(Transfer_Label_Branch_Pretrain, self).__init__()
        self.config=_config
        self.attr_num=_config["attr_num"]
        self.window_size=_config["window_size"]
        self.epoch_num=_config["label_branch_epoch_num"]
        self.lr=_config["label_branch_lr"]
        self.momentum=_config["label_branch_momentum"]
        self.batch_size=_config["label_branch_batch_size"]
        # self.domain_branch_saver=_domain_branch_saver
        # self.label_branch_saver=_label_branch_saver
        # self.saver=_saver
        self.domain_num=_config["domain_num"]
        self.label_num=_config["label_num"]
        self.device=torch.device('cuda:'+str(_device_id) if torch.cuda.is_available() else 'cpu')
        self.cross_entropy_loss_obj=nn.CrossEntropyLoss()   

    # def initialize_pretrain_subbranches(self):

    #     for subbranch_id in range(self.domain_num):
    #         subbranch=Label_Subbranch_Share_Encoder(self.encoder,self.label_num,self.device).to(self.device)
    #         self.pretrain_subbranch_list.append(subbranch)
    #         subbranch_model_path=self.label_subbranches_dir+"/label_subbranch_"+str(subbranch_id)+".pkl"
    #         subbranch_saver=savers.Transfer_Label_Subbranch_Saver(subbranch_model_path)
    #         self.subbranch_saver_list.append(subbranch_saver)
    #         subbranch_params=subbranch.get_parameters_inclu_encoder()
    #         subbranch_optimizer = optim.SGD(params=subbranch_params, lr=self.lr, momentum=self.momentum)
    #         self.subbranch_optimizer_list.append(subbranch_optimizer)
    
    def subbranches_pretrain(self,_train_dataloader_list,_valid_dataloader_list,_subbranches_dir):

        for subbranch_id in range(self.domain_num):
            train_dataloader=_train_dataloader_list[subbranch_id]
            valid_dataloader=_valid_dataloader_list[subbranch_id]
            subbranch=Label_Subbranch(self.config,self.device).to(self.device)
            subbranch_model_path=_subbranches_dir+"/label_subbranch_"+str(subbranch_id)+".pkl"
            subbranch_saver=savers.Transfer_Label_Subbranch_Saver(subbranch_model_path)
            subbranch_optimizer = optim.SGD(params=subbranch.parameters(), lr=self.lr, momentum=self.momentum)
            self.subbranch_pretrain(_subbranch_id=subbranch_id,_subbranch=subbranch,_subbranch_saver=subbranch_saver,\
                _train_dataloader=train_dataloader,_valid_dataloader=valid_dataloader,\
                _subbranch_optimizer=subbranch_optimizer)


    def subbranch_pretrain(self,_subbranch_id,_subbranch,_subbranch_saver,_train_dataloader,_valid_dataloader,_subbranch_optimizer):
        print("label subbranch "+str(_subbranch_id)+" pretrain: ")
        train_set_size=len(_train_dataloader.dataset)
        valid_set_size=len(_valid_dataloader.dataset)
        train_batch_num=train_set_size//self.batch_size
        valid_batch_num=valid_set_size//self.batch_size

        for epoch_id in range(self.epoch_num):
            _subbranch.train()
            train_label_acc_cnt=0

            for index, (feature, label, domain) in enumerate(_train_dataloader):
                # a batch

                feature= feature.to(self.device).float()
                label = label.to(self.device).long()
                
                feature = feature.view(-1, self.attr_num, 1, self.window_size)
                label_prob_vec_predict,_ = _subbranch(feature)

                label_loss=self.cross_entropy_loss_obj(label_prob_vec_predict,label)
                subbranch_loss=label_loss

                _subbranch_optimizer.zero_grad()
                subbranch_loss.backward()
                _subbranch_optimizer.step()

                _, label_predict = torch.max(label_prob_vec_predict, 1)
                train_label_acc_cnt += (label_predict == label).sum()
                
            
            train_label_acc=float(train_label_acc_cnt)/(train_batch_num*self.batch_size)

            print("train_label_acc:")
            print(train_label_acc)

            # src_valid and tar

            _subbranch.eval() 
            valid_label_acc_cnt=0
            with torch.no_grad():
                for index,(feature, label, domain) in enumerate(_valid_dataloader):

                    feature= feature.to(self.device).float()
                    label = label.to(self.device).long()
                    feature = feature.view(-1, self.attr_num, 1, self.window_size)
                    label_prob_vec_predict,_ = _subbranch(feature)
                    
                    _, label_predict = torch.max(label_prob_vec_predict, 1)
                    valid_label_acc_cnt += (label_predict == label).sum()
                    
            valid_label_acc=float(valid_label_acc_cnt)/(valid_batch_num*self.batch_size)

            print('Epoch_id: [{}/{}], subbranch_id: {} valid_label_acc: {:.4f}'.format(epoch_id, self.epoch_num,_subbranch_id, valid_label_acc))
            _subbranch_saver.add_log(valid_label_acc,_subbranch,epoch_id)

        print("best model of subbranch "+str(_subbranch_id))
        _subbranch_saver.final_print()


class Transfer_With_Reconstruct(nn.Module):
    def __init__(self,_config,_device_id=0):
        super(Transfer_With_Reconstruct, self).__init__()
        self.config=_config
        self.epoch_num=_config["epoch_num"]
        self.lr=_config["lr"]
        self.momentum=_config["momentum"]
        self.batch_size=_config["batch_size"]
        self.domain_num=_config["domain_num"]
        self.label_num=_config["label_num"]
        self.attr_num=_config["attr_num"]
        self.window_size=_config["window_size"]

        self.device=torch.device('cuda:'+str(_device_id) if torch.cuda.is_available() else 'cpu')

        self.cross_entropy_loss_obj=nn.CrossEntropyLoss()
        
        self.model=Transfer_Network_With_Reconstruct(self.config,self.device).to(self.device)

    def train_model(self,_pretrain_domain_branch_path,_pretrain_subbranches_dir,\
        _train_dataloader,_valid_dataloader,_test_dataloader,_tar_group_id,_saver):
        self.model.domain_branch.load_state_dict(torch.load(_pretrain_domain_branch_path))
        self.saver=_saver
        for subbranch_id in range(self.domain_num):
            pretrain_subbranch_model_path=_pretrain_subbranches_dir+"/label_subbranch_"+str(subbranch_id)+".pkl"
            self.model.label_branch.label_subbranch_list[subbranch_id].load_state_dict(torch.load(pretrain_subbranch_model_path))
        # print("model parameters: ")
        # print(list(self.model.parameters()))
        # print("label branch parameters:")
        # print(list(self.model.label_branch.parameters()))
        parameters=self.model.get_parameters()
        optimizer=optim.SGD(params=parameters, lr=self.lr, momentum=self.momentum)

        train_set_size=len(_train_dataloader.dataset)
        valid_set_size=len(_valid_dataloader.dataset)
        test_set_size=len(_test_dataloader.dataset)
        train_batch_num=train_set_size//self.batch_size
        valid_batch_num=valid_set_size//self.batch_size
        test_batch_num=test_set_size//self.batch_size

        # acc without furthur training
        initial_valid_acc = self.eval_model(_valid_dataloader,valid_batch_num)
        initial_test_acc=self.eval_model(_test_dataloader,test_batch_num)
        self.saver.initial_valid_acc=initial_valid_acc
        self.saver.initial_test_acc=initial_test_acc
        print("initial valid acc: "+str(initial_valid_acc))
        print("initial test acc: "+str(initial_test_acc))


        for epoch_id in range(self.epoch_num):
            self.model.train()
            train_acc_cnt=0
            total_train_loss = 0
            for index, (feature, label, domain) in enumerate(_train_dataloader):

                # a batch
                feature=my_standardize(feature)
                feature= feature.to(self.device).float()
                label = label.to(self.device).long()
                # print("label:")
                # print(label)
                for entry_id in range(self.batch_size):
                    domain_item=domain[entry_id]
                    if(domain_item>_tar_group_id):
                        domain[entry_id]-=1
                domain = domain.to(self.device).long()
                # print("domain:")
                # print(domain)
                # prepare for convolutional layer
                feature = feature.view(-1, self.attr_num, 1, self.window_size)
                label_vec_predict,domain_vec_predict,reconstruct_loss = self.model(feature)
                # print("one_hot_label_predict: ")
                # print(one_hot_label_predict)
                # print("one_hot_domain_predict: ")
                # print(one_hot_domain_predict)
                label_loss=self.cross_entropy_loss_obj(label_vec_predict,label)
                domain_loss=self.cross_entropy_loss_obj(domain_vec_predict,domain)

                # print("label_loss: "+str(label_loss.item())+",domain_loss: "+str(domain_loss.item())\
                #     +",reconstruct_loss: "+str(reconstruct_loss.item()))
                loss=label_loss+5*domain_loss+1*reconstruct_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_train_loss += loss.item()
                _, label_predict = torch.max(label_vec_predict, 1)
                train_acc_cnt += (label_predict == label).sum()
       
            train_acc = float(train_acc_cnt)/(train_batch_num*self.batch_size)
            train_loss=float(total_train_loss)/train_batch_num
            print("================")
            # print("label_loss: ",label_loss)
            # print("domain_loss: ",domain_loss)
            # print("reconstruct_loss: ",reconstruct_loss)
            # print("one_hot_label_predict: ")
            # print(one_hot_label_predict)
            # print("label:")
            # print(label)
            # print("one_hot_domain_predict: ")
            # print(one_hot_domain_predict)
            # print("domain:")
            # print(domain)
            print('Epoch_id: [{}/{}], train loss: {:.4f}, train acc: {:.4f}'.format(epoch_id, self.epoch_num, train_loss, train_acc))

            # src_valid and tar

                    
            valid_acc = self.eval_model(_valid_dataloader,valid_batch_num)
            print('Epoch_id: [{}/{}], valid acc: {:.4f}'.format(epoch_id, self.epoch_num, valid_acc))

            test_acc=self.eval_model(_test_dataloader,test_batch_num)
            print('Epoch_id: [{}/{}], test acc: {:.4f}'.format(epoch_id, self.epoch_num, test_acc))

            self.saver.add_classify_log(epoch_id,train_loss,train_acc,valid_acc,test_acc,self.model)
        self.save_results()


    def save_results(self):
        self.saver.save_results()

    def eval_model(self,_dataloader,_data_batch_num):
        self.model.eval()
        acc_cnt=0
        with torch.no_grad():
            for index,(feature, label, domain) in enumerate(_dataloader):
                feature= feature.to(self.device).float()
                label = label.to(self.device).long()
                feature = feature.view(-1, self.attr_num, 1, self.window_size)
                label_vec_predict,_,_= self.model(feature)
                _, label_predict = torch.max(label_vec_predict, 1)
                acc_cnt+=(label_predict == label).sum()
                
        acc = float(acc_cnt)/(_data_batch_num*self.batch_size)
        return acc



