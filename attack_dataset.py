"""
This module contains the Dataset class from Pytorch that will serve the purpose
of loading and preprocessing batches of data from the specified directory.
"""

import torch
from torch.utils.data import  Dataset
import h5py
import numpy as np

from attacks_adv.attack_utils import get_hd5_length


class AttackDataset(Dataset):
    """Dataset subclass for pytorch to load data from .h5 files"""

    def __init__(self,
                 data_path,
                 indexes,
                 n_classes,
                 x_original_key,
                 x_adv_key,
                 labels_key,
                 rescale=False,
                 return_original=True,
                 **args):
        """
        Initializer

        Args:
            data_path (str): path to .h5 file to load data from
            indexes (Union[List[int], np.ndarray]): indexes range to query from
                dataset.
            data_key (str): to define the key to the correct dataset in the .
                h5 file containing images or the x-values used for training
                an ML model
            labels_key (str): defines the key to the correct dataset in the .h5
                file containing the labels or the y-values used for training an
                 ML model
            masks_key (str): defines the key to the correct dataset in the .h5
                file containing the masks that are applied to the x-values.
            use_masks (bool): whether to apply masks to the x-values
            n_classes (int): number of classes in the dataset
            rescale (bool): whether to rescale dataset
            use_magnitude (bool): whether to apply magnitude function to the
                x-values.
            **args: optional
            
        Important Assumptions:
            - Based on the argument requirements for initialization, it is
            assumed that only one .h5 file contains several datasets: data,
            labels, masks, etc.

            Example:
                data.h5 file is an .h5 file which contents a layout
                structure as follows:
                    - data.h5:
                        - data
                        - labels
                        - masks

        """
        self.data_path = data_path
        self.x_adv_key = x_adv_key
        self.x_original_key = x_original_key
        self.labels_key = labels_key
        self.n_classes = n_classes
        self.rescale = rescale
        self.return_original=  return_original
        if indexes is not None:
            self.main_indexes = indexes

    def __len__(self):
        """Denotes the total number of samples"""
        if not hasattr(self, 'main_indices'):
            self.main_indexes = np.arange(
                get_hd5_length(self.data_path, self.x_original_key))
        return len(self.main_indexes)

    def __del__(self):
        """Closes .h5 file"""
        if hasattr(self, 'file_archive'):
            self.file_archive.close()

    def _open_h5_file(self):
        """
        Opens the necessary contents for data processing in the h5 file
        """
        self.file_archive = h5py.File(self.data_path, 'r')
        self.x_original = self.file_archive[self.x_original_key]
        self.x_adv = self.file_archive[self.x_adv_key]
        self.labels = self.file_archive[self.labels_key]

    def __getitem__(self, index):
        """Generates one sample of data"""
        if not hasattr(self, 'file_archive'):
            self._open_h5_file()
        x = self.x_original[index]
        x_adv = self.x_adv[index]

        if self.rescale:
            x = self._rescale(x)
            x_adv = self._rescale(x_adv)
        y = self.labels[index]
        if np.ndim(y) < 2:
            y = np.eye(self.n_classes)[y]
            
        x = torch.from_numpy(x).float()
        x_adv = torch.from_numpy(x_adv).float()
        y = torch.from_numpy(y).float()
        if self.return_original:
            return x, x_adv, y
        else:
            return x_adv, y

    def get_length_and_shape(self):
        """
        Method that returns the total length of the .h5 file
        """
        with h5py.File(self.data_path, 'r') as f:
            n = len(f[self.x_original_key])
            sample_shape = f[self.x_original_key][0].shape
        return n, sample_shape

    def get_n_classes(self):
        return self.n_classes
    
    @staticmethod
    def _rescale(x):
        return x / 255.



# class RandomBatchSampler(Sampler):
#     """
#     Sampling class to create random sequential batches from a given dataset
#     E.g. if data is [1,2,3,4] with bs=2. Then first batch, [[1,2], [3,4]] then
#     shuffle batches -> [[3,4],[1,2]]
#     This is useful for cases when you are interested in 'weak shuffling'
    
#     Args:
#         dataset (torch.utils.data.Dataset): dataset class that contains the
#             data that will be weakly shuffled and batched
#         batch_size (int): size of each batch
        
#     Returns:
#         generator object of the shuffled batched indexes
#     """

#     def __init__(self, dataset, batch_size):
#         self.batch_size = batch_size
#         self.dataset_length = len(dataset)
#         self.n_batches = self.dataset_length / self.batch_size
#         self.batch_ids = torch.randperm(int(self.n_batches))

#     def __len__(self):
#         return self.batch_size

#     def __iter__(self):
#         for id in self.batch_ids:
#             idx = torch.arange(id * self.batch_size,
#                                (id + 1) * self.batch_size)
#             for index in idx:
#                 yield int(index)
#         if int(self.n_batches) < self.n_batches:
#             idx = torch.arange(int(self.n_batches) * self.batch_size,
#                                self.dataset_length)
#             for index in idx:
#                 yield int(index)


# def build_test_dataset(data_path,
#                        indexes,
#                        test_batch_size,
#                        test_data_key,
#                        test_labels_key,
#                        test_masks_key,
#                        use_masks,
#                        n_classes,
#                        rescale,
#                        use_magnitude,
#                        num_workers=0,
#                        drop_last=False,
#                        weak_shuffling=False,
#                        **args):
#     """
#     Function to build test dataloader that will be used for the testing stage
#     of the neural network

#     Args:
#         data_path (str): name to .h5 file path
#         indexes (Union[list, np.ndarray]): indexes for testing to query
#         from. If set to None, it
#             will use every observation that is contained in the .h5 file.
#         test_batch_size (int): batch size
#         test_data_key (str): key to dataset in .h5 file
#         test_labels_key (str): key to labels in .h5 file
#         test_masks_key (str): key to masks in .h5 file
#         use_masks (bool): whether to use masks
#         n_classes (int): number of classes in dataset
#         rescale (bool): whether to rescale data
#         use_magnitude (bool): whether to apply magnitude to data
#         num_workers (int): number of processes to run. Defaults to 0
#         drop_last (bool, optional): whether to drop last batch. Defaults to
#             False.
#         weak_shuffling (bool, optional): whether to apply weak shuffling for 
#             fast loading. Defaults to False.

#     Returns:
#         torch.utils.data.DataLoader: test dataloader
#     """

#     test_set = AttackDataset(data_path=data_path,
#                                indexes=indexes,
#                                data_key=test_data_key,
#                                labels_key=test_labels_key,
#                                masks_key=test_masks_key,
#                                use_masks=use_masks,
#                                n_classes=n_classes,
#                                rescale=rescale,
#                                use_magnitude=use_magnitude,
#                                is_train=False
#                                )
#     if weak_shuffling:
#         test_sampler = BatchSampler(
#             RandomBatchSampler(test_set,
#                                test_batch_size),
#             batch_size=test_batch_size,
#             drop_last=drop_last)
#         test_generator = torch.utils.data.DataLoader(test_set,
#                                                      sampler=test_sampler,
#                                                      batch_size=None,
#                                                      num_workers=num_workers)

#     else:
#         test_generator = torch.utils.data.DataLoader(test_set,
#                                                      shuffle=False,
#                                                      batch_size=test_batch_size,
#                                                      num_workers=num_workers)
#     return test_generator


# def _build_fast_loader(dataset,
#                        batch_size=32,
#                        drop_last=False,
#                        num_workers=0,
#                        transforms=None):
#     """
#     Implements fast loading by taking advantage of .h5 dataset
#     The .h5 dataset has a speed bottleneck that scales (roughly) linearly with
#     the number of calls made to it. This is because when queries are made to
#     it, a search is made to find the data item at that index. However,
#     once the start index has been found, taking the next items
#     does not require any more significant computation. So indexing
#     data[start_index: start_index+batch_size] is almost the same as just
#     data[start_index]. The fast loading scheme takes advantage of this.
#     However,because the goal is NOT to load the entirety of the data in memory
#     at once, weak shuffling is used instead of strong shuffling.
    
#     Args:
#         dataset (torch.utils.data.Dataset): a dataset that loads data from .h5
#             files
#         batch_size (int): size of the data batch
#         drop_last (bool): flag to indicate if the last batch will be dropped
#             (if size < batch_size)'
#         num_workers (int): how many subprocess to use for data loading.
#             0 means data will be loaded in the main process (default: 0)
#         transform (Callable function): to apply transformations on dataset

#     Returns:
#         (torch.utils.data.DataLoader): data loading that queries from data
#             using shuffled batches

#     """

#     sampler = BatchSampler(
#         RandomBatchSampler(dataset, batch_size),
#         batch_size=batch_size,
#         drop_last=drop_last)

#     # In dataLoader, argument batch_size must be None
#     data_loader = torch.utils.data.DataLoader(dataset,
#                                               batch_size=None,
#                                               num_workers=num_workers,
#                                               sampler=sampler)

#     return data_loader
