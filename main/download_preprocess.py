from src.data.downloader import download_tinystories
from src.data.preprocessor import tokenize_save_binary

def download_preprocess_tinystories(dataset:str='tinystories'):
    if dataset == 'tinystories':
        # Download TinyStories and save as json
        savedir_path="data/tinystories"
        json_paths = download_tinystories(savedir_path, max_limit=None)
        # Tokenize and save as binary
        tokenize_save_binary(loadfile_path=json_paths['train'], savefile_name='train.bin', chunk_size=10000)
        tokenize_save_binary(loadfile_path=json_paths['val'], savefile_name='val.bin', chunk_size=10000)
    else:
        raise ValueError(f"Dataset {dataset} not supported")


if __name__=='__main__':
    
    # Parse cmdline arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='tinystories', help='Dataset to download and preprocess')
    args = parser.parse_args()

    # Download and preprocess dataset
    download_preprocess_tinystories(args.dataset)