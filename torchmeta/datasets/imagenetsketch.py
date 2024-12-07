import os
import pickle
from PIL import Image
import h5py
import json
import numpy as np

from torchmeta.utils.data import Dataset, ClassDataset, CombinationMetaDataset

class ImagenetSketch(CombinationMetaDataset):
    def __init__(self, root, num_classes_per_task=None, meta_train=False,
                 meta_val=False, meta_test=False, meta_split=None,
                 transform=None, target_transform=None, dataset_transform=None,
                 class_augmentations=None, download=False):
        dataset = ImagenetSketchClassDataset(root, meta_train=meta_train,
            meta_val=meta_val, meta_test=meta_test, meta_split=meta_split,
            transform=transform, class_augmentations=class_augmentations,
            download=download)
        super(ImagenetSketch, self).__init__(dataset, num_classes_per_task,
            target_transform=target_transform, dataset_transform=dataset_transform)

class ImagenetSketchClassDataset(ClassDataset):
    folder = 'imagenetsketch'
    filename = '{0}_data.hdf5'
    filename_labels = '{0}_labels.json'

    def __init__(self, root, meta_train=False, meta_val=False, meta_test=False,
                 meta_split=None, transform=None, class_augmentations=None,
                 download=False):
        super(ImagenetSketchClassDataset, self).__init__(meta_train=meta_train,
            meta_val=meta_val, meta_test=meta_test, meta_split=meta_split,
            class_augmentations=class_augmentations)
        
        self.root = os.path.join(os.path.expanduser(root), self.folder)
        self.transform = transform

        # Try multiple potential file paths
        potential_splits = [meta_split] if meta_split else ['train', 'val', 'test']
        
        for potential_split in potential_splits:
            self.split_filename = self._find_file(self.filename.format(potential_split))
            self.split_filename_labels = self._find_file(self.filename_labels.format(potential_split))
            
            if self.split_filename and self.split_filename_labels:
                self.meta_split = potential_split
                break
        
        self._data = None
        self._labels = None
        self._data_file = None

        # if not self._check_integrity():
        #     raise RuntimeError('ImagenetSketch integrity check failed')
        self._num_classes = len(self.labels)

    def _find_file(self, filename):
        """
        Search for the file in multiple potential locations
        """
        search_paths = [
            os.path.join(self.root, filename),  # Original location
            os.path.join(self.root, '..', filename),  # Parent directory
            os.path.join(os.path.expanduser('~'), 'datasets', 'imagenetsketch', filename),  # Home directory
            os.path.join(os.getcwd(), filename),  # Current working directory
        ]
        
        for path in search_paths:
            if os.path.isfile(path):
                return path
        return None

    def __getitem__(self, index):
        class_name = self.labels[index % self.num_classes]
        data = self.data[class_name]
        transform = self.get_transform(index, self.transform)
        target_transform = self.get_target_transform(index)

        return ImagenetSketchDataset(index, data, class_name,
            transform=transform, target_transform=target_transform)

    @property
    def num_classes(self):
        return self._num_classes

    @property
    def data(self):
        if self._data is None:
            # Support both HDF5 and numpy/pickle formats
            if self.split_filename.endswith('.hdf5'):
                self._data_file = h5py.File(self.split_filename, 'r')
                self._data = self._data_file['datasets']
            else:
                # Fallback to pickle or numpy loading
                try:
                    self._data = np.load(self.split_filename)
                except Exception:
                    with open(self.split_filename, 'rb') as f:
                        self._data = pickle.load(f)
        return self._data

    @property
    def labels(self):
        if self._labels is None:
            # Support both JSON and pickle label formats
            if self.split_filename_labels.endswith('.json'):
                with open(self.split_filename_labels, 'r') as f:
                    self._labels = json.load(f)
            else:
                # Fallback to pickle
                with open(self.split_filename_labels, 'rb') as f:
                    self._labels = pickle.load(f)
        return self._labels

    def _check_integrity(self):
        return (self.split_filename is not None and 
                self.split_filename_labels is not None)

    def close(self):
        if self._data_file is not None:
            self._data_file.close()
            self._data_file = None
            self._data = None

class ImagenetSketchDataset(Dataset):
    def __init__(self, index, data, class_name,
                 transform=None, target_transform=None):
        super(ImagenetSketchDataset, self).__init__(index, transform=transform,
                                                  target_transform=target_transform)
        self.data = data
        self.class_name = class_name

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        # Handle different data types (numpy array, list of images)
        if isinstance(self.data, h5py.Dataset):
            image = Image.fromarray(self.data[index])
        elif isinstance(self.data, np.ndarray):
            image = Image.fromarray(self.data[index])
        elif isinstance(self.data, list):
            image = Image.fromarray(np.array(self.data[index]))
        else:
            raise ValueError(f"Unsupported data type: {type(self.data)}")

        target = self.class_name

        if self.transform is not None:
            image = self.transform(image)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return (image, target)