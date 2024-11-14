"""
2017 ICLR Adversarial Training Methods for Semi-Supervised Text Classification
Ref: https://github.com/iBelieveCJM/Tricks-of-Semi-supervisedDeepLeanring-Pytorch/blob/master/trainer/VATv2.py
# https://github.com/WangJiuniu/adversarial_training/blob/master/at_pytorch/model.py
"""
import torch
import torch.nn as nn
from copy import deepcopy
import numpy as np
from sklearn.metrics import confusion_matrix

from Utils.all_loss import mse_with_softmax


# rampup策略
def exp_rampup(rampup_length=20):
    def warpper(epoch):
        if epoch < rampup_length:
            epoch = np.clip(epoch, 0.0, rampup_length)
            phase = 1.0 - epoch / rampup_length
            return float(np.exp(-5.0 * phase * phase))
        else:
            return 1.0
    return warpper


class Trainer:
    def __init__(self, model, log_set, device=None):
        self.log_set  = log_set
        self.log_set.info('--------------------AdvText 2017--------------------')
        self.optimizer = model.optimizer
        self.device = device
        
        self.ce_loss = nn.CrossEntropyLoss().to(self.device)
        self.cons_loss = mse_with_softmax
        self.model     = model.to(self.device)
        self.best_model = None
        
        self.usp_weight  = 30.0
        self.rampup      = exp_rampup(20)
        self.epoch       = 0
        
        self.p_mult      = 0.2


    def train_iteration(self, train_u_loader, train_x_loader):
        
        label_acc, unlabel_acc, all_loss = 0., 0., 0.
        num_step = 0
        
        for (label_x, label_y), (unlab_x, unlab_y) in zip(train_u_loader, train_x_loader):
            num_step += 1
            label_x, label_y = label_x.to(self.device), label_y.to(self.device)
            unlab_x, unlab_y = unlab_x.to(self.device), unlab_y.to(self.device)
            
            # === decode targets of unlabeled data ===
            lbs, ubs = label_x.size(0), unlab_x.size(0)

            # === forward ===
            _, outputs = self.model(label_x)
            supervised_loss = self.ce_loss(outputs, label_y)

            # === Semi-supervised Training ===
            unlab_emb, unlab_outputs = self.model(unlab_x, 'advtext')
            
            with torch.no_grad():
                vlogits = unlab_outputs.clone().detach()
                r_vadv  = self.gen_r_vadv(unlab_emb) 
                _, rlogits = self.model(unlab_x, r_vadv)
                lds  = self.cons_loss(rlogits, vlogits)
                unsupervised_loss =  lds

            total_loss = supervised_loss + unsupervised_loss
            all_loss += total_loss.item()
            
            ## backward
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()
            
            ##=== log info ===
            temp_acc = label_y.eq(outputs.max(1)[1]).float().sum().item()
            label_acc += temp_acc / lbs
            
            temp_acc = unlab_y.eq(unlab_outputs.max(1)[1]).float().sum().item()
            unlabel_acc += temp_acc / ubs
            
        self.log_set.info(">>>>>[train] label data's accuracy is {0}, and unlabel data's accuracy is {1}, and all training loss is {2}".format(label_acc/float(num_step), unlabel_acc/float(num_step), all_loss/float(num_step)))
    
    
    def train(self, label_loader, unlab_loader):
        self.model.train()

        with torch.enable_grad():
            self.train_iteration(label_loader, unlab_loader)
    
    def val_iteration(self, data_loader):
        acc, num_step = 0., 0
        for _, (data, targets) in enumerate(data_loader):
            num_step += 1
            data, targets = data.to(self.device), targets.to(self.device)

            # === forward ===
            _, outputs     = self.model(data)

            test_acc = targets.eq(outputs.max(1)[1]).float().sum().item()
            acc += test_acc / data.size(0)

        self.log_set.info(">>>>>[test] test data's accuracy is {0}".format(acc/float(num_step)))
        return acc/float(num_step)
    
    def validate(self, data_loader):
        self.model.eval()

        with torch.no_grad():
            return self.val_iteration(data_loader)
    
    def predict(self, model, data_loader):
        model.eval()
        pred_list, y_list = [], []
        for _, (data, targets) in enumerate(data_loader):
            data, targets = data.to(self.device), targets.to(self.device)

            # === forward ===
            _, outputs     = model(data)

            if torch.cuda.is_available():
                y_label = targets.cpu().detach().numpy().tolist()
                pred = outputs.cpu().detach().numpy().tolist()
            else:
                y_label = targets.detach().numpy().tolist()
                pred = outputs.detach().numpy().tolist()
            
            pred_list.extend(pred)
            y_list.extend(y_label)
        
        # print("pred_list shape is {0}, and y_list shape is {1}".format(np.array(pred_list).shape, np.array(y_list).shape))
        tn, fp, fn, tp = confusion_matrix(y_list, np.argmax(pred_list, axis=1)).ravel()
        acc = (tp + tn) / (tp + tn + fp + fn)
        
        recall, precision = tp / (tp + fn + 0.000001), tp / (tp + fp + 0.000001)
        F1 = (2 * precision * recall) / (precision + recall + 0.000001)
        
        return acc, recall, precision, F1
        
    # 主函数
    def loop(self, epochs, label_data, unlab_data, test_data):
        best_F1, best_epoch = 0., 0
        for ep in range(epochs):
            self.epoch = ep
            self.log_set.info("---------------------------- Epochs: {} ----------------------------".format(ep))
            self.train(label_data, unlab_data)
            
            acc, recall, precision, val_F1 = self.predict(self.model, test_data)
            self.log_set.info(
                "Epoch {0}, we get Accuracy: {1}, Recall(TPR): {2}, Precision: {3}, F1 score: {4}".format(ep,
                                                                                                                acc,
                                                                                                                recall,
                                                                                                                precision,
                                                                                                                val_F1))
            if val_F1 > best_F1:
                best_F1 = val_F1
                best_epoch = ep
                self.best_model = deepcopy(self.model).to(self.device)

        acc, recall, precision, F1 = self.predict(self.model, test_data)
        self.log_set.info("Final epoch {0}, we get Accuracy: {1}, Recall(TPR): {2}, Precision: {3}, F1 score: {4}".format(epochs, acc, recall, precision, F1))
        acc, recall, precision, F1 = self.predict(self.best_model, test_data)
        self.log_set.info("The best epoch {0}, we get Accuracy: {1}, Recall(TPR): {2}, Precision: {3}, F1 score: {4}".format(best_epoch, acc, recall, precision, F1))

    
    def gen_r_vadv(self, x):
        if torch.cuda.is_available():
            sample_numpy = x.cpu().detach().numpy()
        else:
            sample_numpy = x.detach().numpy()
            
        noise = sample_numpy / (np.sqrt(np.sum(sample_numpy ** 2, axis=(1, 2))).reshape((-1, 1, 1)) + 1e-16)
        # print("noise shape is {0}".format(noise.shape))
        noise = self.p_mult * noise
        return torch.from_numpy(noise).to(self.device)
        