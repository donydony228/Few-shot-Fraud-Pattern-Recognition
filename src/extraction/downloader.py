import gdown
import os
import shutil
import zipfile

from tqdm import tqdm

def download_dataset(dir='dataset'):
    zip_file = f'{dir}.zip'
    if os.path.exists(dir) and os.path.isdir(dir):
        shutil.rmtree(dir)

    print('Downloading dataset...')
    file_id = os.getenv('DATASET_ID')
    gdown.download(id=file_id, output=zip_file)

    print('Extracting zip file...')
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        # If it contains multiple files/directories at the root, this might need adjustment
        first_entry_name = zip_ref.namelist()[0]
        original_parent_name = os.path.normpath(first_entry_name).split(os.sep)[0]
        # zip_ref.extractall()

        members = zip_ref.infolist()
        for member in tqdm(members, desc='Extracting', unit='file'):
                try:
                    zip_ref.extract(member)
                except zipfile.error as e:
                    print(f"Error extracting {member.filename}: {e}")

    os.rename(original_parent_name, dir)
    print(f'Dataset has been downloaded to {dir} directory.')