import h5py
import os 
from urllib.request import urlretrieve
from pathlib import Path

def download(src, dst):
    if not os.path.exists(dst):
        os.makedirs(Path(dst).parent, exist_ok=True)
        print('downloading %s -> %s...' % (src, dst))
        urlretrieve(src, dst)

def get_fn(dataset, task):
    return os.path.join("data", dataset, task, f"{dataset}.h5"), os.path.join('data', dataset, task, 'gt', f'gt_{dataset}.h5')

def prepare(dataset, task):
    url = DATASETS[dataset][task]['url']
    gt_url = DATASETS[dataset][task]['gt_url']
    fn, gt_fn = get_fn(dataset, task)

    download(url, fn)
    download(gt_url, gt_fn)


DATASETS = {
    'gooaq': {
        'task2': {
            'url': 'https://huggingface.co/datasets/sadit/SISAP2025/resolve/main/benchmark-dev-gooaq.h5?download=true',
            'queries': lambda x: x['train'],#[:5000],
            'data': lambda x: x['train'],#[:5000],
            'gt_url': 'https://huggingface.co/datasets/sadit/SISAP2025/resolve/main/allknn-benchmark-dev-gooaq.h5?download=true',
            'gt_I': lambda x: x['knns'],
            'c': [1, 2, 3, 4],
            'k': [1, 5, 10, 20, 30, 40, 50],
            't': [1, 3, 5],
            #'c': list(range(1, 3)),
            #'k': list(range(1, 11)),
            #'t': [1, 3, 5],
        }
    },
    'ccnews': {
        'task2': {
            'url': 'https://huggingface.co/datasets/sadit/SISAP2025/resolve/main/benchmark-dev-ccnews.h5?download=true',
            'queries': lambda x: x['train'],#[:5000],
            'data': lambda x: x['train'],#[:5000],
            'gt_url': 'https://huggingface.co/datasets/sadit/SISAP2025/resolve/main/allknn-benchmark-dev-ccnews.h5?download=true',
            'gt_I': lambda x: x['knns'],
            'c': [1, 2, 3, 4],
            'k': [1, 5, 10, 20, 30, 40, 50],
            't': [1, 3, 5],
            #'c': list(range(1, 3)),
            #'k': list(range(1, 11)),
            #'t': [1, 3, 5],
        }
    },
    'yahooaq': {
        'task2': {
            'url': 'https://huggingface.co/datasets/sadit/SISAP2025/resolve/main/benchmark-dev-yahooaq.h5?download=true',
            'queries': lambda x: x['train'],#[:5000],
            'data': lambda x: x['train'],#[:5000],
            'gt_url': 'https://huggingface.co/datasets/sadit/SISAP2025/resolve/main/allknn-benchmark-dev-yahooaq.h5?download=true',
            'gt_I': lambda x: x['knns'],
            'c': [1, 2, 3, 4],
            'k': [1, 5, 10, 20, 30, 40, 50],
            't': [1, 3, 5],
            #'c': list(range(1, 3)),
            #'k': list(range(1, 11)),
            #'t': [1, 3, 5],
        }
    },
    
    'pubmed23': {
        'task2': {
            'url': 'https://huggingface.co/datasets/sadit/SISAP2025/resolve/main/benchmark-dev-pubmed23.h5?download=true',
            'queries': lambda x: x['train'],#[:5000],
            'data': lambda x: x['train'],#[:5000],
            'gt_url': 'https://huggingface.co/datasets/sadit/SISAP2025/resolve/main/allknn-benchmark-dev-pubmed23.h5?download=true',
            'gt_I': lambda x: x['knns'],
            'c': [2], #[1, 2, 3, 4],
            'k': [1, 10, 30, 50], #[1, 5, 10, 20, 30, 40, 50],
            't': [1, 3, 5],
            #'c': list(range(1, 3)),
            #'k': list(range(1, 11)),
            #'t': [1, 3, 5],
        }
    }
}