import os
import pickle
from PIL import Image
import h5py
import json

from torchmeta.utils.data import Dataset, ClassDataset, CombinationMetaDataset
# QKFIX: See torchmeta.datasets.utils for more informations
from torchmeta.datasets.utils import download_file_from_gdrive_gdown


class ImagenetSketch(CombinationMetaDataset):
    """
    imagenet-sketch dataset
    github: https://github.com/HaohanWang/ImageNet-Sketch
    paper: https://arxiv.org/abs/1905.13549

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
    # Google Drive ID from https://github.com/renmengye/few-shot-ssl-public
    gdrive_id = '16V_ZlkW4SsnNDtnGmaBRq2OoPmUOc5mY'
    zip_filename = 'ImageNet-Sketch.zip'
    gz_md5 = 'b38f1eb4251fb9459ecc8e7febf9b2eb'
    pkl_filename = 'imagenet-sketch-cache-{0}.pkl'

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

        self.split_filename = os.path.join(self.root,
            self.filename.format(self.meta_split))
        self.split_filename_labels = os.path.join(self.root,
            self.filename_labels.format(self.meta_split))

        self._data = None
        self._labels = None

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('ImagenetSketch integrity check failed')
        self._num_classes = len(self.labels)

    def __getitem__(self, index):
        class_name = self.labels[index % self.num_classes]
        data = self.data[class_name]
        transform = self.get_transform(index, self.transform)
        target_transform = self.get_target_transform(index)

        return ImagenetSketch(index, data, class_name,
            transform=transform, target_transform=target_transform)

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
        return (os.path.isfile(self.split_filename)
            and os.path.isfile(self.split_filename_labels))

    def close(self):
        if self._data_file is not None:
            self._data_file.close()
            self._data_file = None
            self._data = None

    def download(self):
        import zipfile

        if self._check_integrity():
            return
        

        google_drive_link = "https://drive.google.com/file/d/16V_ZlkW4SsnNDtnGmaBRq2OoPmUOc5mY/view"
        download_file_from_gdrive_gdown(google_drive_link, self.root, self.zip_filename)

        filename = os.path.join(self.root, self.zip_filename)

        with zipfile.open(filename, 'r') as f:
            f.extractall(self.root)

        for split in ['train', 'val', 'test']:
            filename = os.path.join(self.root, self.filename.format(split))
            if os.path.isfile(filename):
                continue

            pkl_filename = os.path.join(self.root, self.pkl_filename.format(split))
            if not os.path.isfile(pkl_filename):
                raise IOError()
            with open(pkl_filename, 'rb') as f:
                data = pickle.load(f)
                images, classes = data['image'], data['label']

            with h5py.File(filename, 'w') as f:
                group = f.create_group('datasets')
                for name, indices in classes.items():
                    group.create_dataset(name, data=images[indices])

            labels_filename = os.path.join(self.root, self.filename_labels.format(split))
            with open(labels_filename, 'w') as f:
                labels = sorted(list(classes.keys()))
                json.dump(labels, f)

            if os.path.isfile(pkl_filename):
                os.remove(pkl_filename)


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
