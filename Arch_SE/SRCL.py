"""
2022 ICPC SRCL: Self-Supervised Learning of Smart Contract Representations
Ref: https://github.com/iBelieveCJM/Tricks-of-Semi-supervisedDeepLeanring-Pytorch/blob/master/trainer/eMixPseudoLabelv2.py
https://github.com/SCRepslearner/SmartLearner
"""
import torch
import torch.nn as nn
from copy import deepcopy
import numpy as np
from sklearn.metrics import confusion_matrix

from Utils.mixup import mixup_ce_loss_with_softmax, mixup_two_targets


def one_hot(targets, nClass, device):
    logits = torch.zeros(targets.size(0), nClass).to(device)
    return logits.scatter_(1, targets.unsqueeze(1), 1)


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
        self.log_set = log_set
        self.log_set.info('------------------------2022 ICPC SRCL---------------------------')
        self.optimizer = model.optimizer
        self.device = device

        self.ce_loss = nn.CrossEntropyLoss(ignore_index=-1).to(self.device)
        self.mixup_loss = mixup_ce_loss_with_softmax  # mixup_mse_loss_with_softmax
        self.model = model.to(self.device)
        self.best_model = None

        self.usp_weight = 1.0
        self.log_set.info_freq = 20
        self.rampup = exp_rampup(80)
        self.epoch = 0
        self.num_classes = 2

        self.soft = True
        self.alpha = 1.0

    def train_iteration(self, train_u_loader, train_x_loader):
        label_acc, unlabel_acc, all_loss = 0., 0., 0.
        num_step = 0
        for (label_x, label_y, _), (unlab_x, unlab_y, udx) in zip(train_u_loader, train_x_loader):
            num_step += 1
            label_x, label_y = label_x.to(self.device), label_y.to(self.device)
            unlab_x, unlab_y = unlab_x.to(self.device), unlab_y.to(self.device)

            # === decode targets of unlabeled data ===
            lbs, ubs = label_x.size(0), unlab_x.size(0)

            # === forward ===
            _, lab_outputs = self.model(label_x)
            supervised_loss = self.ce_loss(lab_outputs, label_y)

            # === Semi-supervised Training ===
            ## mixup pslab loss
            iter_unlab_pslab = self.epoch_pslab[udx]
            mixed_ux, uy_a, uy_b, lam = mixup_two_targets(unlab_x, iter_unlab_pslab, self.alpha, is_bias=False)

            _, mixed_outputs_u = self.model(mixed_ux.long())
            mix_loss = self.mixup_loss(mixed_outputs_u, uy_a, uy_b, lam)
            unsupervised_loss = self.rampup(self.epoch) * self.usp_weight * mix_loss

            # 计算所有损失
            total_loss = supervised_loss + unsupervised_loss
            all_loss += total_loss.item()

            ## backward
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()

            ##=== log info ===
            temp_acc = label_y.eq(lab_outputs.max(1)[1]).float().sum().item()
            label_acc += temp_acc / lbs

            temp_acc = unlab_y.eq(mixed_outputs_u.max(1)[1]).float().sum().item()
            unlabel_acc += temp_acc / ubs

        self.log_set.info(
            ">>>>>[train] label data's accuracy is {0}, and unlabel data's accuracy is {1}, and all training loss is {2}".format(
                label_acc / float(num_step), unlabel_acc / float(num_step), all_loss / float(num_step)))

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
            _, outputs = self.model(data)

            test_acc = targets.eq(outputs.max(1)[1]).float().sum().item()
            acc += test_acc / data.size(0)

        self.log_set.info(">>>>>[test] test data's accuracy is {0}".format(acc / float(num_step)))
        return acc / float(num_step)

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
            _, outputs = model(data)

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
        # construct epoch pseudo labels
        init_pslab = self.create_soft_pslab if self.soft else self.create_pslab
        self.epoch_pslab = init_pslab(n_samples=len(unlab_data.dataset), n_classes=self.num_classes)

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
        self.log_set.info(
            "Final epoch {0}, we get Accuracy: {1}, Recall(TPR): {2}, Precision: {3}, F1 score: {4}".format(epochs, acc,
                                                                                                            recall,
                                                                                                            precision,
                                                                                                            F1))
        acc, recall, precision, F1 = self.predict(self.best_model, test_data)
        self.log_set.info(
            "The best epoch {0}, we get Accuracy: {1}, Recall(TPR): {2}, Precision: {3}, F1 score: {4}".format(
                best_epoch, acc, recall, precision, F1))

    def create_pslab(self, n_samples, n_classes, dtype='rand'):
        if dtype == 'rand':
            pslab = torch.randint(0, n_classes, (n_samples,))
        elif dtype == 'zero':
            pslab = torch.zeros(n_samples)
        else:
            raise ValueError('Unknown pslab dtype: {}'.format(dtype))
        return pslab.long().to(self.device)

    def create_soft_pslab(self, n_samples, n_classes, dtype='rand'):
        if dtype == 'rand':
            rlabel = torch.randint(0, n_classes, (n_samples,)).long().to(self.device)
            pslab = one_hot(rlabel, n_classes, self.device)
        elif dtype == 'zero':
            pslab = torch.zeros(n_samples, n_classes)
        else:
            raise ValueError('Unknown pslab dtype: {}'.format(dtype))
        return pslab.to(self.device)