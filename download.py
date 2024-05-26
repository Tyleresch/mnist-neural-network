import urllib.request
import os
import gzip
import shutil

# Alternative URLs for the MNIST dataset
urls = {
    "train-images-idx3-ubyte.gz": "https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz",
    "train-labels-idx1-ubyte.gz": "https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz",
    "t10k-images-idx3-ubyte.gz": "https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz",
    "t10k-labels-idx1-ubyte.gz": "https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz"
}

# Directory to save the files
data_dir = 'samples'
os.makedirs(data_dir, exist_ok=True)

for filename, url in urls.items():
    print(f"Downloading {filename} from {url}...")
    file_path = os.path.join(data_dir, filename)
    urllib.request.urlretrieve(url, file_path)
    print(f"Saved {filename} to {file_path}")
    
    # Extract the files
    print(f"Extracting {filename}...")
    with gzip.open(file_path, 'rb') as f_in:
        with open(file_path[:-3], 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    os.remove(file_path)
    print(f"Extracted {filename[:-3]}")

print("All files downloaded and extracted.")
