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
from models.cifar_core import CifarModel, save_results
import utils


def save_all_tests(results, domain, path, keys):
    for key, value in results.items():
        if key in keys:
            complete_results = {
                'f1': 0.5 * (value[1]*value[2]) / (value[1] + value[2]),
                'precision': value[1],
                'recall': value[2],
                'accuracy': value[0],
                'loss': results['test_loss']
            }

            save_results(domain, path, complete_results, suffix="_" + key[8:])


class CifarDomainDiscriminative(CifarModel):
    def __init__(self, opt):
        super(CifarDomainDiscriminative, self).__init__(opt)
        self.prior_shift_weight = np.array(opt['prior_shift_weight'])
            
    def _test(self, loader):
        """Test the model performance"""
        
        self.network.eval()

        total = 0
        correct = 0
        test_loss = 0
        output_list = []
        feature_list = []
        target_list = []
        with torch.no_grad():
            for i, (images, targets) in enumerate(loader):
                images, targets = images.to(self.device), targets.to(self.device)
                outputs, features = self.forward(images)
                loss = self._criterion(outputs, targets)
                test_loss += loss.item()

                output_list.append(outputs)
                feature_list.append(features)
                target_list.append(targets)
                
        outputs = torch.cat(output_list, dim=0)
        features = torch.cat(feature_list, dim=0)
        targets = torch.cat(target_list, dim=0)
        
        metrics_sum_prob_wo_prior_shift = self.compute_accuracy_sum_prob_wo_prior_shift(outputs, targets)
        metrics_sum_prob_w_prior_shift = self.compute_accuracy_sum_prob_w_prior_shift(outputs, targets)
        metrics_max_prob_w_prior_shift = self.compute_accuracy_max_prob_w_prior_shift(outputs, targets)

        test_result = {
            'accuracy_sum_prob_wo_prior_shift': metrics_sum_prob_wo_prior_shift[0],
            'accuracy_sum_prob_w_prior_shift': metrics_sum_prob_w_prior_shift[0],
            'accuracy_max_prob_w_prior_shift': metrics_max_prob_w_prior_shift[0],
            'metrics_sum_prob_wo_prior_shift': metrics_sum_prob_wo_prior_shift,
            'metrics_sum_prob_w_prior_shift': metrics_sum_prob_w_prior_shift,
            'metrics_max_prob_w_prior_shift':metrics_max_prob_w_prior_shift,
            'test_loss': test_loss,
            'outputs': outputs.cpu().numpy(),
            'features': features.cpu().numpy()
        }
        return test_result

    def compute_accuracy_sum_prob_wo_prior_shift(self, outputs, targets):
        probs = F.softmax(outputs, dim=1).cpu().numpy()
        targets = targets.cpu().numpy()
        predictions = np.argmax(probs[:, :10] + probs[:, 10:], axis=1)
        accuracy = (predictions == targets).mean() * 100.

        precision, recall = utils.compute_matrix_metrics(torch.tensor(predictions), torch.tensor(targets))

        return accuracy, np.nanmean(precision), np.nanmean(recall)

    def compute_accuracy_sum_prob_w_prior_shift(self, outputs, targets):
        probs = F.softmax(outputs, dim=1).cpu().numpy() * self.prior_shift_weight
        targets = targets.cpu().numpy()
        predictions = np.argmax(probs[:, :10] + probs[:, 10:], axis=1)
        accuracy = (predictions == targets).mean() * 100.

        precision, recall = utils.compute_matrix_metrics(torch.tensor(predictions), torch.tensor(targets))

        return accuracy, np.nanmean(precision), np.nanmean(recall)
    
    def compute_accuracy_max_prob_w_prior_shift(self, outputs, targets):
        probs = F.softmax(outputs, dim=1).cpu().numpy() * self.prior_shift_weight
        targets = targets.cpu().numpy()
        predictions = np.argmax(np.stack((probs[:, :10], probs[:, 10:])).max(axis=0), axis=1)
        accuracy = (predictions == targets).mean() * 100.

        precision, recall = utils.compute_matrix_metrics(torch.tensor(predictions), torch.tensor(targets))

        return accuracy, np.nanmean(precision), np.nanmean(recall)

    def test(self):
        # Test and save the result
        state_dict = torch.load(os.path.join(self.save_path, 'ckpt.pth'))
        self.load_state_dict(state_dict)
        test_color_result = self._test(self.test_color_loader)
        test_gray_result = self._test(self.test_gray_loader)
        utils.save_pkl(test_color_result, os.path.join(self.save_path, 'test_color_result.pkl'))
        utils.save_pkl(test_gray_result, os.path.join(self.save_path, 'test_gray_result.pkl'))

        save_all_tests(test_color_result, 'color', self.save_path, keys=['metrics_sum_prob_wo_prior_shift', 'metrics_sum_prob_w_prior_shift', 'metrics_max_prob_w_prior_shift'])


        # Output the classification accuracy on test set for different inference
        # methods
        info = ('Test on color images accuracy sum prob without prior shift: {}\n' 
                'Test on color images accuracy sum prob with prior shift: {}\n' 
                'Test on color images accuracy max prob with prior shift: {}\n' 
                'Test on gray images accuracy sum prob without prior shift: {}\n'
                'Test on gray images accuracy sum prob with prior shift: {}\n'
                'Test on gray images accuracy max prob with prior shift: {}\n'
                .format(test_color_result['accuracy_sum_prob_wo_prior_shift'],
                        test_color_result['accuracy_sum_prob_w_prior_shift'],
                        test_color_result['accuracy_max_prob_w_prior_shift'],
                        test_gray_result['accuracy_sum_prob_wo_prior_shift'],
                        test_gray_result['accuracy_sum_prob_w_prior_shift'],
                        test_gray_result['accuracy_max_prob_w_prior_shift']))
        utils.write_info(os.path.join(self.save_path, 'test_result.txt'), info)
    
    