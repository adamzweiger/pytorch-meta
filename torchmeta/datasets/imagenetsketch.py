import os
import h5py
import json
import numpy as np
import random
from PIL import Image
from tqdm import tqdm
from datasets import load_dataset

from torchmeta.utils.data import Dataset, ClassDataset, CombinationMetaDataset

class ImagenetSketch(CombinationMetaDataset):
    """
    Imagenet-Sketch dataset
    GitHub: https://github.com/HaohanWang/ImageNet-Sketch
    Paper: https://arxiv.org/abs/1905.13549

    """
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
    dataset_name = 'songweig/imagenet_sketch'
    filename = '{0}_data.hdf5'
    filename_labels = '{0}_labels.json'

    def __init__(self, root, meta_train=False, meta_val=False, meta_test=False,
                 meta_split=None, transform=None, class_augmentations=None,
                 download=False, seed=42):
        super(ImagenetSketchClassDataset, self).__init__(meta_train=meta_train,
            meta_val=meta_val, meta_test=meta_test, meta_split=meta_split,
            class_augmentations=class_augmentations)
        
        self.root = os.path.join(os.path.expanduser(root), self.folder)
        self.transform = transform

        self.split_filename = os.path.join(self.root,
            self.filename.format(self.meta_split))
        self.split_filename_labels = os.path.join(self.root,
            self.filename_labels.format(self.meta_split))

        self._data = None
        self._labels = None

        if download:
            self.download(seed)

        if not self._check_integrity():
            raise RuntimeError('ImagenetSketch integrity check failed')
        self._num_classes = len(self.labels)

    def __getitem__(self, index):
        image = self.data[str(index)][:]
        label = self.labels[index]

        if self.transform:
            image = self.transform(Image.fromarray(image))

        return image, label

    @property
    def num_classes(self):
        return self._num_classes

    @property
    def data(self):
        if self._data is None:
            self._data_file = h5py.File(self.split_filename, 'r')
            self._data = self._data_file['datasets']
        return self._data

    @property
    def labels(self):
        if self._labels is None:
            with open(self.split_filename_labels, 'r') as f:
                self._labels = json.load(f)
        return self._labels

    def _check_integrity(self):
        return os.path.isfile(self.split_filename) and os.path.isfile(self.split_filename_labels)

    def close(self):
        if self._data_file is not None:
            self._data_file.close()
            self._data_file = None
            self._data = None

    def download(self, seed):
        if self._check_integrity():
            return

        print("Downloading Imagenet-Sketch dataset from Hugging Face...")
        dataset = load_dataset(self.dataset_name, split="train", trust_remote_code=True)

        # Prepare output directories
        os.makedirs(self.root, exist_ok=True)

        # Set random seed for reproducibility
        random.seed(seed)

        # Select 100 random classes
        all_classes = sorted(set(sample['label'] for sample in dataset))
        class_indices = random.sample(all_classes, 100)
        filtered_data = [sample for sample in dataset if sample['label'] in class_indices]

        # Split into train (64), val (16), test (20)
        train_classes = class_indices[:64]
        val_classes = class_indices[64:80]
        test_classes = class_indices[80:]

        splits = {
            'train': [sample for sample in filtered_data if sample['label'] in train_classes],
            'val': [sample for sample in filtered_data if sample['label'] in val_classes],
            'test': [sample for sample in filtered_data if sample['label'] in test_classes],
        }

        # Save dataset in HDF5 and JSON format
        for split_name, split_data in splits.items():
            split_filename = os.path.join(self.root, self.filename.format(split_name))
            split_labels_filename = os.path.join(self.root, self.filename_labels.format(split_name))

            with h5py.File(split_filename, 'w') as hdf5_file:
                group = hdf5_file.create_group('datasets')
                labels = []

                for i, sample in enumerate(tqdm(split_data, desc=f"Processing {split_name} split")):
                    image = np.array(sample['image'])
                    label = sample['label']
                    group.create_dataset(str(i), data=image, compression="gzip")
                    labels.append(label)

            with open(split_labels_filename, 'w') as f:
                json.dump(labels, f)

        print("Dataset download and processing complete.")


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
        image = Image.fromarray(self.data[index])
        target = self.class_name

        if self.transform is not None:
            image = self.transform(image)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return (image, target)
