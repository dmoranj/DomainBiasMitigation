import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter
from models import basenet
from models import dataloader
import utils
import pandas as pd

RESULTS_FOLDER = './data/results/'
RESULTS_CSV = RESULTS_FOLDER + '/results.csv'


def save_results(domain, save_path, test_results, suffix=""):
    data = {key: [value] for key, value in test_results.items() if key in ['f1', 'precision', 'recall', 'accuracy', 'loss']}
    experiment, name = save_path.split("/")[-2:]
    data['domain'] = [domain]
    data['experiment'] = [experiment + suffix]
    data['name'] = [name]

    os.makedirs(RESULTS_FOLDER, exist_ok=True)
    data_df = pd.DataFrame(data)

    filename = RESULTS_CSV

    if os.path.exists(filename):
        data_df.to_csv(filename, mode='a', header=False)
    else:
        data_df.to_csv(filename, mode='w')


class CifarModel():
    def __init__(self, opt):
        super(CifarModel, self).__init__()
        self.epoch = 0
        self.device = opt['device']
        self.save_path = opt['save_folder']
        self.print_freq = opt['print_freq']
        self.init_lr = opt['optimizer_setting']['lr']
        self.log_writer = SummaryWriter(os.path.join(self.save_path, 'logfile'))
        
        self.set_network(opt)
        self.set_data(opt)
        self.set_optimizer(opt)

    def set_network(self, opt):
        """Define the network"""
        
        self.network = basenet.ResNet18(num_classes=opt['output_dim']).to(self.device)

    def forward(self, x):
        out, feature = self.network(x)
        return out, feature

    def set_data(self, opt):
        """Set up the dataloaders"""
        
        data_setting = opt['data_setting']

        with open(data_setting['train_data_path'], 'rb') as f:
            train_array = pickle.load(f)

        mean = tuple(np.mean(train_array / 255., axis=(0, 1, 2)))
        std = tuple(np.std(train_array / 255., axis=(0, 1, 2)))
        normalize = transforms.Normalize(mean=mean, std=std)

        if data_setting['augment']:
            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])
        else:
            transform_train = transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])

        train_data = dataloader.CifarDataset(data_setting['train_data_path'], 
                                             data_setting['train_label_path'],
                                             transform_train)
        test_color_data = dataloader.CifarDataset(data_setting['test_color_path'], 
                                                  data_setting['test_label_path'],
                                                  transform_test)
        test_gray_data = dataloader.CifarDataset(data_setting['test_gray_path'], 
                                                 data_setting['test_label_path'],
                                                 transform_test)

        self.train_loader = torch.utils.data.DataLoader(
                                 train_data, batch_size=opt['batch_size'],
                                 shuffle=True, num_workers=1)
        self.test_color_loader = torch.utils.data.DataLoader(
                                      test_color_data, batch_size=opt['batch_size'],
                                      shuffle=False, num_workers=1)
        self.test_gray_loader = torch.utils.data.DataLoader(
                                     test_gray_data, batch_size=opt['batch_size'],
                                     shuffle=False, num_workers=1)
    
    def set_optimizer(self, opt):
        optimizer_setting = opt['optimizer_setting']
        self.optimizer = optimizer_setting['optimizer']( 
                            params=self.network.parameters(), 
                            lr=optimizer_setting['lr'],
                            momentum=optimizer_setting['momentum'],
                            weight_decay=optimizer_setting['weight_decay']
                            )
        
    def _criterion(self, output, target):
        return F.cross_entropy(output, target)
        
    def state_dict(self):
        state_dict = {
            'model': self.network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epoch': self.epoch
        }  
        return state_dict
    
    def load_state_dict(self, state_dict):
        self.network.load_state_dict(state_dict['model'])
        self.optimizer.load_state_dict(state_dict['optimizer'])
        self.epoch = state_dict['epoch']

    def log_result(self, name, result, step):
        self.log_writer.add_scalars(name, result, step)

    def adjust_lr(self):
        lr = self.init_lr * (0.1 ** (self.epoch // 50))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def _train(self, loader):
        """Train the model for one epoch"""
        
        self.network.train()
        self.adjust_lr()
        
        train_loss = 0
        total = 0
        correct = 0
        for i, (images, targets) in enumerate(loader):
            images, targets = images.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()
            outputs, _ = self.forward(images)
            loss = self._criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            accuracy = correct*100. / total

            train_result = {
                'accuracy': correct*100. / total,
                'loss': loss.item(),
            }
            self.log_result('Train iteration', train_result,
                            len(loader)*self.epoch + i)

            if self.print_freq and (i % self.print_freq == 0):
                print('Training epoch {}: [{}|{}], loss:{}, accuracy:{}'.format(
                    self.epoch, i+1, len(loader), loss.item(), accuracy
                ))

        self._train_accuracy = accuracy
        self.epoch += 1

    def _test(self, loader):
        """Test the model performance"""
        
        self.network.eval()

        total = 0
        correct = 0
        test_loss = 0
        output_list = []
        feature_list = []
        predict_list = []
        precision_list = []
        recall_list = []

        with torch.no_grad():
            for i, (images, targets) in enumerate(loader):
                images, targets = images.to(self.device), targets.to(self.device)
                outputs, features = self.forward(images)
                loss = self._criterion(outputs, targets)
                test_loss += loss.item()

                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                predict_list.extend(predicted.tolist())
                output_list.append(outputs.cpu().numpy())
                feature_list.append(features.cpu().numpy())

                precision_per_class, recall_per_class = utils.compute_matrix_metrics(predicted, targets)
                precision_list.append(precision_per_class.mean())
                recall_list.append(recall_per_class.mean())

        recall = np.nanmean(recall_list)
        precision = np.nanmean(precision_list)

        test_result = {
            'accuracy': correct*100. / total,
            'predict_labels': predict_list,
            'outputs': np.vstack(output_list),
            'features': np.vstack(feature_list),
            'precision': precision,
            'recall': recall,
            'f1': 0.5 * (precision * recall) / (precision + recall),
            'loss': test_loss
        }

        return test_result

    def train(self):
        self._train(self.train_loader)
        utils.save_state_dict(self.state_dict(), os.path.join(self.save_path, 'ckpt.pth'))

    def test(self):
        # Test and save the result
        test_color_result = self._test(self.test_color_loader)
        test_gray_result = self._test(self.test_gray_loader)
        utils.save_pkl(test_color_result, os.path.join(self.save_path, 'test_color_result.pkl'))
        utils.save_pkl(test_gray_result, os.path.join(self.save_path, 'test_gray_result.pkl'))

        save_results("color", self.save_path, test_color_result)
        save_results("gray", self.save_path, test_gray_result)

        # Output the classification accuracy on test set
        info = ('Test on color images accuracy: {}\n' 
                'Test on gray images accuracy: {}'.format(test_color_result['accuracy'],
                                                          test_gray_result['accuracy']))
        utils.write_info(os.path.join(self.save_path, 'test_result.txt'), info)


    


            
