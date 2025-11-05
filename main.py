from dotenv import load_dotenv

from src.extraction.data_loader import load_data
from src.extraction.downloader import download_dataset

if __name__ == '__main__':
    load_dotenv()
    download_dataset(dir='dataset')

    features_dir = 'dataset/features'
    split_csv_path = 'dataset/label_split.csv'
    # Datasets: {'seen, 'unseen'}
    # DataLoaders: {'train', 'val', 'test_seen', 'test_unseen'}
    datasets, dataloaders = load_data(features_dir, split_csv_path, one_shot=True)