import os
import json
import h5py
import numpy as np
from PIL import Image
from tqdm import tqdm

from torchmeta.utils.data import Dataset, ClassDataset, CombinationMetaDataset
from huggingface_hub import hf_hub_download

class ImagenetSketch(CombinationMetaDataset):
    """
    Imagenet-Sketch dataset
    GitHub: https://github.com/HaohanWang/ImageNet-Sketch
    Paper: https://arxiv.org/abs/1905.13549

    This class follows a similar structure to the MiniImagenet class, but the dataset
    is provided as a single HDF5 file and a single labels JSON file from a Hugging Face
    repository. There are no predefined splits for train/val/test. If meta_train, meta_val,
    or meta_test is specified, we currently treat the entire dataset as one split.
    """
    def __init__(self, root, num_classes_per_task=None, meta_train=False,
                 meta_val=False, meta_test=False, meta_split=None,
                 transform=None, target_transform=None, dataset_transform=None,
                 class_augmentations=None, download=False):

        # If no meta_split given, default to 'train'
        # The dataset does not have separate splits, so we ignore differences.
        if meta_split is None:
            meta_split = 'train'
        if sum([meta_train, meta_val, meta_test]) > 1:
            raise ValueError("Only one of meta_train, meta_val, meta_test can be True.")
        
        # We do not have different splits, so we just treat the entire dataset as a single split
        # The chosen meta_split will be 'train' for consistency.
        meta_split = 'train'

        dataset = ImagenetSketchClassDataset(root, meta_train=(meta_split=='train'),
            meta_val=(meta_split=='val'), meta_test=(meta_split=='test'),
            meta_split=meta_split, transform=transform, class_augmentations=class_augmentations,
            download=download)
        super(ImagenetSketch, self).__init__(dataset, num_classes_per_task,
            target_transform=target_transform, dataset_transform=dataset_transform)


class ImagenetSketchClassDataset(ClassDataset):
    folder = 'imagenetsketch'
    # Names of the files on the huggingface repository
    hdf5_filename = 'imagenet_sketch_resized.hdf5'
    labels_filename = 'imagenet_sketch_labels.json'

    # Unlike miniimagenet, we have a single file for the entire dataset.
    def __init__(self, root, meta_train=False, meta_val=False, meta_test=False,
                 meta_split=None, transform=None, class_augmentations=None,
                 download=False):
        super(ImagenetSketchClassDataset, self).__init__(meta_train=meta_train,
            meta_val=meta_val, meta_test=meta_test, meta_split=meta_split,
            class_augmentations=class_augmentations)

        self.root = os.path.join(os.path.expanduser(root), self.folder)
        self.transform = transform

        self.split_filename = os.path.join(self.root, self.hdf5_filename)
        self.split_filename_labels = os.path.join(self.root, self.labels_filename)

        self._data_file = None
        self._data = None
        self._labels = None
        self.class_names = None
        self.class_dict = None

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('ImagenetSketch integrity check failed')

        # Once integrity is confirmed, we load labels and form class groups
        image_labels = self._load_image_labels()
        self._initialize_class_info(image_labels)

        self._num_classes = len(self.class_names)

    def _initialize_class_info(self, image_labels):
        # image_labels: list of int labels for each image index
        unique_labels = sorted(set(image_labels))
        # Convert each unique label into a string class name, similar to MiniImagenet
        self.class_names = [str(l) for l in unique_labels]

        # Build a dictionary of class_name -> list of image indices
        class_dict = {cn: [] for cn in self.class_names}
        for i, lbl in enumerate(image_labels):
            class_dict[str(lbl)].append(i)

        self.class_dict = class_dict

    def _load_image_labels(self):
        with open(self.split_filename_labels, 'r') as f:
            labels = json.load(f)
        return labels

    @property
    def num_classes(self):
        return self._num_classes

    @property
    def data(self):
        if self._data is None:
            self._data_file = h5py.File(self.split_filename, 'r')
            self._data = self._data_file['datasets']
        return self._data

    def __getitem__(self, index):
        # index here refers to class index in meta-learning scenario
        class_name = self.class_names[index % self.num_classes]
        # Get all image indices belonging to this class
        indices = self.class_dict[class_name]

        transform = self.get_transform(index, self.transform)
        target_transform = self.get_target_transform(index)

        return ImagenetSketchDataset(index, self.data, class_name, indices,
                                     transform=transform, target_transform=target_transform)

    def _check_integrity(self):
        return os.path.isfile(self.split_filename) and os.path.isfile(self.split_filename_labels)

    def close(self):
        if self._data_file is not None:
            self._data_file.close()
            self._data_file = None
            self._data = None

    def download(self):
        if self._check_integrity():
            return

        print("Downloading Imagenet-Sketch dataset from Hugging Face...")

        # Download files from the Hugging Face repo "janellecai/imagenet_sketch_resized"
        os.makedirs(self.root, exist_ok=True)
        hf_hub_download(repo_id="janellecai/imagenet_sketch_resized", filename=self.hdf5_filename, cache_dir=self.root, force_filename=self.hdf5_filename)
        hf_hub_download(repo_id="janellecai/imagenet_sketch_resized", filename=self.labels_filename, cache_dir=self.root, force_filename=self.labels_filename)

        print("Dataset download complete.")


class ImagenetSketchDataset(Dataset):
    def __init__(self, index, data, class_name, indices,
                 transform=None, target_transform=None):
        super(ImagenetSketchDataset, self).__init__(index, transform=transform,
                                                    target_transform=target_transform)
        self.data = data
        self.class_name = class_name
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        # index here is the index within the class-specific subset
        image_index = self.indices[index]
        image = self.data[str(image_index)][:]  # Load the image array
        image = Image.fromarray(image)
        target = self.class_name

        if self.transform is not None:
            image = self.transform(image)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return (image, target)
