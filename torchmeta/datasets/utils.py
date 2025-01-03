import os
import json
import gdown


def get_asset_path(*args):
    basedir = os.path.dirname(__file__)
    return os.path.join(basedir, 'assets', *args)


def get_asset(*args, dtype=None):
    filename = get_asset_path(*args)
    if not os.path.isfile(filename):
        raise IOError('{} not found'.format(filename))

    if dtype is None:
        _, dtype = os.path.splitext(filename)
        dtype = dtype[1:]

    if dtype == 'json':
        with open(filename, 'r') as f:
            data = json.load(f)
    else:
        raise NotImplementedError()
    return data

# QKFIX: The current version of `download_file_from_google_drive` (as of torchvision==0.8.1)
# is inconsistent, and a temporary fix has been added to the bleeding-edge version of
# Torchvision. The temporary fix removes the behaviour of `_quota_exceeded`, whenever the
# quota has exceeded for the file to be downloaded. As a consequence, this means that there
# is currently no protection against exceeded quotas. If you get an integrity error in Torchmeta
# (e.g. "MiniImagenet integrity check failed" for MiniImagenet), then this means that the quota
# has exceeded for this dataset. See also: https://github.com/tristandeleu/pytorch-meta/issues/54
# 
# See also: https://github.com/pytorch/vision/issues/2992
# 
# The following functions are our own implementation since they were removed from torchvision

def _get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value
    return None

def _save_response_content(response, destination, chunk_size=32768):
    with open(destination, "wb") as f:
        for chunk in response.iter_content(chunk_size):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)

def _quota_exceeded(response):
    return False

def download_file_from_gdrive_gdown(shareable_link, root, filename):
    """
    Downloads a file from Google Drive using its shareable link.

    Args:
        shareable_link (str): The Google Drive shareable link.
        output_file (str): Name of the file to save the downloaded content.

    Returns:
        str: Path of the downloaded file.
    """
    # Extract the file ID from the shareable link
    try:
        file_id = shareable_link.split("/d/")[1].split("/")[0]
        download_url = f"https://drive.google.com/uc?id={file_id}"
        
        os.makedirs(root, exist_ok=True)
        output_file = os.path.join(root, filename)
        # Download the file
        gdown.download(download_url, output_file, quiet=False)
        print(f"File downloaded successfully as {output_file}")
        return output_file
    except IndexError:
        raise ValueError("Invalid Google Drive shareable link format.")


def download_file_from_google_drive(file_id, root, filename=None, md5=None):
    """Download a Google Drive file from  and place it in root.

    Args:
        file_id (str): id of file to be downloaded
        root (str): Directory to place downloaded file in
        filename (str, optional): Name to save the file under. If None, use the id of the file.
        md5 (str, optional): MD5 checksum of the download. If None, do not check
    """
    # Based on https://stackoverflow.com/questions/38511444/python-download-files-from-google-drive-using-url
    import requests
    url = "https://docs.google.com/uc?export=download"

    root = os.path.expanduser(root)
    if not filename:
        filename = file_id
    fpath = os.path.join(root, filename)

    os.makedirs(root, exist_ok=True)

    if os.path.isfile(fpath):
        print('Using downloaded and verified file: ' + fpath)
    else:
        session = requests.Session()

        response = session.get(url, params={'id': file_id}, stream=True)
        token = _get_confirm_token(response)

        if token:
            params = {'id': file_id, 'confirm': token}
            response = session.get(url, params=params, stream=True)

        if _quota_exceeded(response):
            msg = (
                f"The daily quota of the file {filename} is exceeded and it "
                f"can't be downloaded. This is a limitation of Google Drive "
                f"and can only be overcome by trying again later."
            )
            raise RuntimeError(msg)

        _save_response_content(response, fpath)
