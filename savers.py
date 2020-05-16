from my_env import *

class Baseline_Saver(object):
    def __init__(self,_res_save_path,_best_model_info_path,_best_model_path):
        self.res_save_path=_res_save_path
        self.best_model_info_path=_best_model_info_path
        self.best_model_path=_best_model_path
        self.res_log=[]
        self.best_model_info=[]
        self.max_src_valid_acc=0
    
    def add_classify_log(self,_epoch_id,_src_train_loss,_src_train_acc,_src_valid_acc,_tar_acc,_model):
        self.res_log.append({"epoch_id":_epoch_id,"src_train_loss":_src_train_loss,"src_train_acc":_src_train_acc,\
            "src_valid_acc":_src_valid_acc,"tar_acc":_tar_acc})
        if(_src_valid_acc>self.max_src_valid_acc):
            self.max_src_valid_acc=_src_valid_acc
            self.best_model_info=[{"epoch_id":_epoch_id,"src_train_loss":_src_train_loss,"src_train_acc":_src_train_acc,\
                "src_valid_acc":_src_valid_acc,"tar_acc":_tar_acc}]
            torch.save(_model.state_dict(),self.best_model_path)
    
    
    def save_log(self):
        df_log=pd.DataFrame(self.res_log)
        df_log.to_csv(self.res_save_path)
        print("log saved at "+str(self.res_save_path))
        df_best_model_info=pd.DataFrame(self.best_model_info)
        df_best_model_info.to_csv(self.best_model_info_path)

class Finetune_Saver(object):
    def __init__(self,_res_save_path,_best_model_info_path,_best_model_path):
        self.res_save_path=_res_save_path
        self.best_model_info_path=_best_model_info_path
        self.best_model_path=_best_model_path
        self.res_log=[]
        self.best_model_info=[]
        self.max_valid_acc=0
    
    def add_classify_log(self,_epoch_id,_train_loss,_train_acc,_valid_acc,_test_acc,_model):
        self.res_log.append({"epoch_id":_epoch_id,"train_loss":_train_loss,"train_acc":_train_acc,\
            "valid_acc":_valid_acc,"test_acc":_test_acc})
        if(_valid_acc>self.max_valid_acc):
            self.max_valid_acc=_valid_acc
            self.best_model_info=[{"epoch_id":_epoch_id,"train_loss":_train_loss,"train_acc":_train_acc,\
                "valid_acc":_valid_acc,"test_acc":_test_acc}]
            torch.save(_model.state_dict(),self.best_model_path)
    
    
    def save_results(self):
        df_log=pd.DataFrame(self.res_log)
        df_log.to_csv(self.res_save_path)
        print("log saved at "+str(self.res_save_path))
        df_best_model_info=pd.DataFrame(self.best_model_info)
        df_best_model_info.to_csv(self.best_model_info_path)


class Transfer_Saver(object):
    def __init__(self,_res_save_path,_best_model_info_path,_best_model_path):
        self.res_save_path=_res_save_path
        self.best_model_info_path=_best_model_info_path
        self.best_model_path=_best_model_path
        self.res_log=[]
        self.best_model_info=[]
        self.max_valid_acc=0
        self.initial_valid_acc=-1
        self.initial_test_acc=-1
    
    def add_classify_log(self,_epoch_id,_train_loss,_train_acc,_valid_acc,_test_acc,_model):
        self.res_log.append({"epoch_id":_epoch_id,"train_loss":_train_loss,"train_acc":_train_acc,\
            "valid_acc":_valid_acc,"test_acc":_test_acc})
        if(_valid_acc>self.max_valid_acc):
            self.max_valid_acc=_valid_acc
            self.best_model_info=[{"epoch_id":_epoch_id,"train_loss":_train_loss,"train_acc":_train_acc,\
                "initial_valid_acc":self.initial_valid_acc,"initial_test_acc":self.initial_test_acc,\
                "valid_acc":_valid_acc,"test_acc":_test_acc}]
            torch.save(_model.state_dict(),self.best_model_path)
    
    def save_results(self):
        df_log=pd.DataFrame(self.res_log)
        df_log.to_csv(self.res_save_path)
        print("log saved at "+str(self.res_save_path))
        df_best_model_info=pd.DataFrame(self.best_model_info)
        df_best_model_info.to_csv(self.best_model_info_path)

class Transfer_Domain_Branch_Saver(object):
    def __init__(self,_best_model_path,_best_model_info_path,_log_path):

        self.best_model_path=_best_model_path
        self.best_model_info_path=_best_model_info_path
        self.log_path=_log_path
        self.log=[]
        self.best_model_info=[]
        self.max_valid_domain_acc=0
        self.best_epoch_id=-1
    
    def add_log(self,_train_acc,_valid_acc,_model,_epoch_id):
        self.log.append({"epoch_id":_epoch_id,"train_acc":_train_acc,\
            "valid_acc":_valid_acc})
        if(_valid_acc>self.max_valid_domain_acc):
            self.max_valid_domain_acc=_valid_acc
            self.best_epoch_id=_epoch_id
            self.best_model_info=[{"epoch_id":_epoch_id,"train_acc":_train_acc,\
            "valid_acc":_valid_acc}]
            torch.save(_model.state_dict(),self.best_model_path)
    
    def final_print(self):
        print("best epoch id: "+str(self.best_epoch_id))
        print("max valid domain acc: "+str(self.max_valid_domain_acc))
        df_log=pd.DataFrame(self.log)
        df_log.to_csv(self.log_path)
        df_best_model_info=pd.DataFrame(self.best_model_info)
        df_best_model_info.to_csv(self.best_model_info_path)


class Transfer_Label_Subbranch_Saver(object):
    def __init__(self,_best_model_path):
        self.best_model_path=_best_model_path
        self.max_valid_label_acc=0
        self.best_epoch_id=-1
    
    def add_log(self,_valid_label_acc,_model,_epoch_id):

        if(_valid_label_acc>self.max_valid_label_acc):
            self.max_valid_label_acc=_valid_label_acc
            self.best_epoch_id=_epoch_id
            torch.save(_model.state_dict(),self.best_model_path)
    
    def final_print(self):
        print("best epoch id: "+str(self.best_epoch_id))
        print("max valid label acc: "+str(self.max_valid_label_acc))