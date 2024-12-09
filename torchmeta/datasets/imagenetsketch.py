import os
import h5py
import json
import numpy as np
from PIL import Image
import requests
from tqdm import tqdm

from torchmeta.utils.data import Dataset, ClassDataset, CombinationMetaDataset

class ImagenetSketch(CombinationMetaDataset):
    def __init__(self, root, num_classes_per_task=None, meta_train=False,
                 meta_val=False, meta_test=False, meta_split=None,
                 transform=None, target_transform=None, dataset_transform=None,
                 class_augmentations=None, download=True):  # Changed default download to True
        dataset = ImagenetSketchClassDataset(root, meta_train=meta_train,
            meta_val=meta_val, meta_test=meta_test, meta_split=meta_split,
            transform=transform, class_augmentations=class_augmentations,
            download=download)
        super(ImagenetSketch, self).__init__(dataset, num_classes_per_task,
            target_transform=target_transform, dataset_transform=dataset_transform)

class ImagenetSketchClassDataset(ClassDataset):
    def __init__(self, root, meta_train=False, meta_val=False, meta_test=False,
                 meta_split=None, transform=None, class_augmentations=None,
                 download=True):  # Changed default download to True
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

        # Ensure root directory exists
        os.makedirs(self.root, exist_ok=True)

        # Download dataset if requested or files are missing
        if download:
            self._download_dataset()

        # Find HDF5 file
        self.hdf5_path = self._find_file(potential_filenames)
        if not self.hdf5_path:
            raise FileNotFoundError("Could not find HDF5 dataset file. Try downloading.")

        # Find labels file
        self.labels_path = self._find_file(potential_label_filenames)
        if not self.labels_path:
            raise FileNotFoundError("Could not find labels file. Try downloading.")

        self._data_file = None
        self._data = None
        self._labels = None

        # Populate labels and prepare dataset
        self._num_classes = len(self.labels)

    def _download_dataset(self):
        """
        Download ImageNet Sketch dataset from Hugging Face
        """
        # HuggingFace repository details
        repo = 'janellecai/imagenet_sketch_resized'
        base_url = f'https://huggingface.co/{repo}/resolve/main/'
        
        # Files to download
        files_to_download = {
            'imagenet_sketch_resized.hdf5': 'train_data.hdf5',
            'imagenet_sketch_labels.json': 'train_labels.json'
        }

        for remote_filename, local_filename in files_to_download.items():
            local_path = os.path.join(self.root, local_filename)
            remote_url = base_url + remote_filename

            # Skip if file already exists
            if os.path.exists(local_path):
                print(f"{local_filename} already exists. Skipping download.")
                continue

            print(f"Downloading {remote_filename}...")
            
            # Streaming download with progress bar
            response = requests.get(remote_url, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            block_size = 1024  # 1 Kibibyte
            
            with open(local_path, 'wb') as file, tqdm(
                desc=local_filename,
                total=total_size,
                unit='iB',
                unit_scale=True,
                unit_divisor=1024,
            ) as progress_bar:
                for data in response.iter_content(block_size):
                    size = file.write(data)
                    progress_bar.update(size)

            print(f"Download complete: {local_filename}")

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

    # Rest of the code remains the same as in the original file
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
            target = self.target_transform(target)

        return image, target