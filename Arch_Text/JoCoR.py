"""
2020 CVPR JoCoR: Combating Noisy Labels by Agreement: A Joint Training Method with Co-Regularization
Ref: https://github.com/chengtan9907/Co-learning/blob/master/algorithms/JoCoR.py
"""
import torch
import torch.nn as nn
from copy import deepcopy
import numpy as np
from sklearn.metrics import confusion_matrix

from Utils.Teaching_Loss import loss_jocor


class Trainer:
    def __init__(self, model1, model2, log_set, lr, epochs, device=None):
        self.log_set  = log_set
        self.log_set.info('--------------------2020 CVPR JoCoR--------------------')
        # 设置model
        # 设置model
        self.device, self.co_lambda = device, 0.1
        self.model1, self.model2 = model1.to(self.device), model2.to(self.device)
        self.lr, self.epochs = lr, epochs

        forget_rate = 0.2
        epoch_decay_start, num_gradual, exponent = 20, 1, 1
        self.adjust_lr = 1.0

        # Adjust learning rate and betas for Adam Optimizer
        mom1, mom2 = 0.9, 0.1
        self.alpha_plan = [self.lr] * self.epochs
        self.beta1_plan = [mom1] * self.epochs

        for i in range(epoch_decay_start, self.epochs):
            self.alpha_plan[i] = float(self.epochs - i) / (self.epochs - epoch_decay_start) * self.lr
            self.beta1_plan[i] = mom2

        # define drop rate schedule
        self.rate_schedule = np.ones(self.epochs) * forget_rate
        self.rate_schedule[:num_gradual] = np.linspace(0, forget_rate ** exponent, num_gradual)

        self.optimizer = torch.optim.Adam(list(self.model1.parameters()) + list(self.model2.parameters()), lr=self.lr)
        self.ce_loss = nn.CrossEntropyLoss().to(self.device)
        self.loss_fn = loss_jocor

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
            _, output_1 = self.model1(label_x)
            supervised_loss_1 = self.ce_loss(output_1, label_y)

            _, output_2 = self.model2(label_x)
            supervised_loss_2 = self.ce_loss(output_2, label_y)

            # === Semi-supervised Training ===
            _, unlab_output_1 = self.model1(unlab_x)
            _, unlab_output_2 = self.model2(unlab_x)
            loss_1, loss_2 = self.loss_fn(unlab_output_1, unlab_output_2, unlab_y, self.rate_schedule[self.epoch], self.co_lambda)

            total_loss = supervised_loss_1 + supervised_loss_2 + loss_1 + loss_2
            all_loss += total_loss.item()

            ## backward
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()

            ##=== log info ===
            temp_acc = label_y.eq(output_1.max(1)[1]).float().sum().item()
            label_acc += temp_acc / lbs

            temp_acc = unlab_y.eq(unlab_output_1.max(1)[1]).float().sum().item()
            unlabel_acc += temp_acc / ubs

        self.log_set.info(
            ">>>>>[train] label data's accuracy is {0}, and unlabel data's accuracy is {1}, and all training loss is {2}".format(
                label_acc / float(num_step), unlabel_acc / float(num_step), all_loss / float(num_step)))

    def train(self, label_loader, unlab_loader):
        print('Start training ...')
        self.model1.train()
        self.model2.train()

        if self.adjust_lr == 1:
            self.adjust_learning_rate(self.optimizer, self.epoch)

        with torch.enable_grad():
            self.train_iteration(label_loader, unlab_loader)


    def predict(self, model, data_loader):
        model.eval()

        pred_list, y_list = [], []
        for data, targets in data_loader:
            data, targets = data.to(self.device), targets.to(self.device)

            # === forward ===
            _, logits = model(data)

            if torch.cuda.is_available():
                y_label = targets.cpu().detach().numpy().tolist()
                pred = logits.cpu().detach().numpy().tolist()
            else:
                y_label = targets.detach().numpy().tolist()
                pred = logits.detach().numpy().tolist()

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

            acc, recall, precision, val_F1 = self.predict(self.model1, test_data)
            self.log_set.info(
                "Epoch {0}, we get Accuracy: {1}, Recall(TPR): {2}, Precision: {3}, F1 score: {4}".format(ep,
                                                                                                          acc,
                                                                                                          recall,
                                                                                                          precision,
                                                                                                          val_F1))
            if val_F1 > best_F1:
                best_F1 = val_F1
                best_epoch = ep
                self.best_model = deepcopy(self.model1).to(self.device)

        acc, recall, precision, F1 = self.predict(self.model1, test_data)
        self.log_set.info(
            "Final epoch {0}, we get Accuracy: {1}, Recall(TPR): {2}, Precision: {3}, F1 score: {4}".format(epochs,
                                                                                                            acc,
                                                                                                            recall,
                                                                                                            precision,
                                                                                                            F1))
        acc, recall, precision, F1 = self.predict(self.best_model, test_data)
        self.log_set.info(
            "The best epoch {0}, we get Accuracy: {1}, Recall(TPR): {2}, Precision: {3}, F1 score: {4}".format(
                best_epoch, acc, recall, precision, F1))

    def adjust_learning_rate(self, optimizer, epoch):
        for param_group in optimizer.param_groups:
            param_group['lr'] = self.alpha_plan[epoch]
            param_group['betas'] = (self.beta1_plan[epoch], 0.999)  # Only change beta1