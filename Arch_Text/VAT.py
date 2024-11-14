import torch
import contextlib
import torch.nn as nn
from copy import deepcopy
import numpy as np
from sklearn.metrics import confusion_matrix

from Utils.all_loss import mse_with_softmax
"""
2019 TPAMI Virtual Adversarial Training: A Regularization Method for Supervised and Semi-Supervised Learning
Ref: https://github.com/iBelieveCJM/Tricks-of-Semi-supervisedDeepLeanring-Pytorch/blob/master/trainer/VATv2.py
"""

@contextlib.contextmanager
def disable_tracking_bn_stats(model):
    def switch_attr(m):
        if hasattr(m, 'track_running_stats'):
            m.track_running_stats ^= True

    model.apply(switch_attr)
    yield
    model.apply(switch_attr)


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
    def __init__(self, model, log_set, embedding_dim=256, device=None):
        self.log_set = log_set
        self.log_set.info('--------------------VAT 2019--------------------')
        self.optimizer = model.optimizer
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=-1).to(device)
        self.cons_loss = mse_with_softmax
        self.model     = model.to(device)
        self.best_model = None
        self.device = device
        
        self.usp_weight  = 30.0
        self.print_freq  = 20
        self.rampup      = exp_rampup(20)
        self.epoch       = 0
        
        self.embedding_dim = embedding_dim
        self.xi          = 10.
        self.n_power     = 1
        self.eps         = 5e-4


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

            ##=== Semi-supervised Training ===
            _, unlab_outputs = self.model(unlab_x)
            
            with torch.no_grad():
                vlogits = unlab_outputs.clone().detach()
                
            with disable_tracking_bn_stats(self.model):
                r_vadv  = self.gen_r_vadv(unlab_x) 
                _, rlogits = self.model(unlab_x, r_vadv)
                lds  = self.cons_loss(rlogits, vlogits)
                unsupervised_loss = self.rampup(self.epoch) * self.usp_weight * lds

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

        print(">>>>>[test] test data's accuracy is {0}".format(acc/float(num_step)))
        return acc/float(num_step)
    
    def validate(self, data_loader):
        self.model.eval()

        with torch.no_grad():
            return self.val_iteration(data_loader)
    
    def predict(self, model, data_loader):
        model.eval()
        
        for _, (data, targets) in enumerate(data_loader):
            data, targets = data.to(self.device), targets.to(self.device)

            # === forward ===
            _, outputs     = model(data)
            
            test_acc = targets.eq(outputs.max(1)[1]).float().sum().item()
            acc = test_acc / float(data.size(0))

            if self.device:
                y_label = targets.cpu().detach().numpy()
                pred = outputs.cpu().detach().numpy()
            else:
                y_label = targets.detach().numpy()
                pred = outputs.detach().numpy()
            
        tn, fp, fn, tp = confusion_matrix(y_label, np.argmax(pred, axis=1)).ravel()
        
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

        # print(">>>[best] Epoch {0}, the best accuracy is {1}".format(best_epoch, best_acc))
        # self.predict(test_data)
        acc, recall, precision, F1 = self.predict(self.model, test_data)
        self.log_set.info(
            "Final epoch {0}, we get Accuracy: {1}, Recall(TPR): {2}, Precision: {3}, F1 score: {4}".format(epochs, acc,
                                                                                                            recall,
                                                                                                            precision,
                                                                                                            F1))
        acc, recall, precision, F1 = self.predict(self.best_model, test_data)
        self.log_set.info(
            "The best epoch {0}, we get Accuracy: {1}, Recall(TPR): {2}, Precision: {3}, F1 score: {4}".format(
                best_epoch, acc, recall, precision, F1))

    
    def gen_r_vadv(self, x, mean=0, std=1):
        batch_size, seq_len = x.size(0), x.size(1)

        noise = torch.randn((batch_size, seq_len, self.embedding_dim)) * std + mean
        return noise.to(self.device)
        