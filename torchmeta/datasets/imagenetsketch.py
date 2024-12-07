import os
import h5py
import json
import numpy as np
from PIL import Image

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
    def __init__(self, root, meta_train=False, meta_val=False, meta_test=False,
                 meta_split=None, transform=None, class_augmentations=None,
                 download=False):
        super(ImagenetSketchClassDataset, self).__init__(meta_train=meta_train,
            meta_val=meta_val, meta_test=meta_test, meta_split=meta_split,
            class_augmentations=class_augmentations)
        
        self.root = root
        self.transform = transform

        # Potential filenames
        potential_filenames = [
            'imagenet_sketch_resized.hdf5',  # HuggingFace upload filename
            'train_data.hdf5',  # Original torchmeta filename
        ]

        # Potential label filenames
        potential_label_filenames = [
            'imagenet_sketch_labels.json',  # HuggingFace upload filename
            'train_labels.json',  # Original torchmeta filename
        ]

        # Find HDF5 file
        self.hdf5_path = self._find_file(potential_filenames)
        if not self.hdf5_path:
            raise FileNotFoundError("Could not find HDF5 dataset file")

        # Find labels file
        self.labels_path = self._find_file(potential_label_filenames)
        if not self.labels_path:
            raise FileNotFoundError("Could not find labels file")

        self._data_file = None
        self._data = None
        self._labels = None

        # Populate labels and prepare dataset
        self._num_classes = len(self.labels)

    def _find_file(self, possible_filenames):
        """
        Search for the file in multiple potential locations
        """
        search_paths = [
            self.root,
            os.path.join(self.root, 'imagenetsketch'),
            os.path.join(os.path.expanduser('~'), 'datasets', 'imagenetsketch'),
            os.getcwd()
        ]
        
        for path in search_paths:
            for filename in possible_filenames:
                full_path = os.path.join(path, filename)
                if os.path.isfile(full_path):
                    return full_path
        return None

    def __getitem__(self, index):
        class_name = self.labels[index % self.num_classes]
        
        # Find indices for this class
        class_indices = [i for i, label in enumerate(self.labels) if label == class_name]
        
        # Select a random index for this class
        class_index = class_indices[index % len(class_indices)]
        
        transform = self.get_transform(index, self.transform)
        target_transform = self.get_target_transform(index)

        return ImagenetSketchDataset(index, self.data, class_name, class_index,
            transform=transform, target_transform=target_transform)

    @property
    def num_classes(self):
        return self._num_classes

    @property
    def data(self):
        if self._data is None:
            # Open HDF5 file
            self._data_file = h5py.File(self.hdf5_path, 'r')
            
            # Access the 'datasets' group
            if 'datasets' in self._data_file:
                self._data = self._data_file['datasets']
            else:
                # Fallback to root-level dataset
                self._data = self._data_file
        return self._data

    @property
    def labels(self):
        if self._labels is None:
            # Load labels from JSON
            with open(self.labels_path, 'r') as f:
                self._labels = json.load(f)
        return self._labels

    def close(self):
        if self._data_file is not None:
            self._data_file.close()
            self._data_file = None
            self._data = None

class ImagenetSketchDataset(Dataset):
    def __init__(self, index, data, class_name, class_index,
                 transform=None, target_transform=None):
        super(ImagenetSketchDataset, self).__init__(index, transform=transform,
                                                  target_transform=target_transform)
        self.data = data
        self.class_name = class_name
        self.class_index = class_index

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # Get the image from the HDF5 dataset
        image = Image.fromarray(self.data[self.class_index])
        target = self.class_name

        if self.transform is not None:
            image = self.transform(image)

        if self.target_transform is not None:
            ta