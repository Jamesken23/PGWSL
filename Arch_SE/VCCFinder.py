"""
2021 ESE VCCFinder: Revisiting the VCCFinder approach for the identification of vulnerability-contributing commits
Ref: https://github.com/chengtan9907/Co-learning/blob/master/algorithms/Decoupling.py
https://github.com/VulnCatcher/VulnCatcher
"""
import torch
import torch.nn as nn
from copy import deepcopy
import numpy as np
from sklearn.metrics import confusion_matrix



class Trainer:
    def __init__(self, model1, model2, log_set, lr, epochs, device=None):
        self.log_set  = log_set
        self.log_set.info('--------------------2021 ESE VCCFinder--------------------')
        # 设置model
        self.device = device
        self.model1, self.model2 = model1.to(self.device), model2.to(self.device)

        epoch_decay_start, num_gradual, exponent = 20, 1, 1
        self.adjust_lr = 1.0

        # Adjust learning rate and betas for Adam Optimizer
        mom1, mom2 = 0.9, 0.1
        self.alpha_plan = [lr] * epochs
        self.beta1_plan = [mom1] * epochs

        for i in range(epoch_decay_start, epochs):
            self.alpha_plan[i] = float(epochs - i) / (epochs - epoch_decay_start) * lr
            self.beta1_plan[i] = mom2

        self.optimizer = torch.optim.AdamW(list(self.model1.parameters()) + list(self.model2.parameters()), lr=lr)
        self.loss_fn = torch.nn.CrossEntropyLoss()


    def train_iteration(self, train_u_loader, train_x_loader):
        label_acc, unlabel_acc, all_loss = 0., 0., 0.
        num_step = 0
        for (label_x, label_y, _), (unlab_x, unlab_y, udx) in zip(train_u_loader, train_x_loader):
            num_step += 1

            iter_unlab_pslab = self.epoch_pslab[udx]
            label_x, label_y = label_x.to(self.device), label_y.to(self.device)
            unlab_x, unlab_y = unlab_x.to(self.device), unlab_y.to(self.device)

            # 对clean_x和noisy_x进行拼接
            l_u_x = torch.cat((label_x, unlab_x), 0)
            # 对clean_y和noisy_y进行拼接
            l_u_y = torch.cat((label_y, iter_unlab_pslab), 0)
            # === decode targets of unlabeled data ===
            lbs = l_u_x.size(0)

            # print("clean_x shape is {0}, noisy_x shape is {1}, c_n_x shape is {2}".format(clean_x.shape, noisy_x.shape, c_n_x.shape))
            l_u_x, l_u_y = l_u_x.to(self.device), l_u_y.to(self.device)

            # === forward ===
            _, logits1 = self.model1(l_u_x)
            _, logits2 = self.model2(l_u_x)

            _, pred1 = torch.max(logits1, dim=1)
            _, pred2 = torch.max(logits2, dim=1)

            # 选择两个模型预测结果不一样的数据，计算这些数据上的loss并用来更新模型参数
            # Update by Disagreement
            inds = torch.where(pred1 != pred2)
            loss_1 = self.loss_fn(logits1[inds[0]], l_u_y[inds[0]])
            loss_2 = self.loss_fn(logits2[inds[0]], l_u_y[inds[0]])

            ## update pseudo labels
            with torch.no_grad():
                pseudo_preds = logits1.max(1)[1]
                self.epoch_pslab[udx] = pseudo_preds[label_x.size(0):].detach()

            ## backward
            self.optimizer.zero_grad()
            loss_1.backward()
            loss_2.backward()
            self.optimizer.step()

            ##=== log info ===
            temp_acc = l_u_y.eq(logits1.max(1)[1]).float().sum().item()
            label_acc += temp_acc / lbs

            all_loss += loss_1.item() + loss_2.item()

        self.log_set.info(">>>>>[train] label data's accuracy is {0}, and all training loss is {1}".format(
            label_acc / float(num_step), all_loss / float(num_step)))


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
        self.epoch_pslab = self.create_pslab(n_samples=len(unlab_data.dataset), n_classes=2)

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


    def create_pslab(self, n_samples, n_classes, dtype='rand'):
        if dtype == 'rand':
            pslab = torch.randint(0, n_classes, (n_samples,))
        elif dtype == 'zero':
            pslab = torch.zeros(n_samples)
        else:
            raise ValueError('Unknown pslab dtype: {}'.format(dtype))

        return pslab.long().to(self.device)


    def adjust_learning_rate(self, optimizer, epoch):
        for param_group in optimizer.param_groups:
            param_group['lr'] = self.alpha_plan[epoch]
            param_group['betas'] = (self.beta1_plan[epoch], 0.999)  # Only change beta1