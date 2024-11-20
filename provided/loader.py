import time
from multiprocessing import Pool, cpu_count
import torch
from torch.utils.data import Dataset
import pandas as pd


class SingleProcessDataset(Dataset):
    def __init__(self, csv_file):
        start_time = time.time()
        print("Loading data using single process...")

        self.data = pd.read_csv(csv_file)
        self.features = torch.FloatTensor(self.data[["x1", "x2", "x3"]].values)
        self.labels = torch.LongTensor(self.data["label"].values)

        # Calculate total load time
        self.load_time = time.time() - start_time
        print(f"Dataset loading completed in {self.load_time:.2f} seconds")

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


class MultiProcessDataset(SingleProcessDataset):
    def __init__(self, csv_file):
        start_time = time.time()
        print("Loading data using multi process...")

        ########### YOUR CODE HERE ############

        # Number of processes to use
        num_processes = cpu_count()

        chunk_size = 1000  # based on dataset

        # Split the CSV into chunks
        chunks = pd.read_csv(csv_file, chunksize=chunk_size)

        # Create a pool of worker processes
        with Pool(processes=num_processes) as pool:
            results = pool.map(self._load_chunk, chunks)

        # Concatenate the results
        self.features = torch.cat([result[0] for result in results])
        self.labels = torch.cat([result[1] for result in results])

        ########### END YOUR CODE  ############

        # Calculate total load time
        self.load_time = time.time() - start_time
        print(f"Dataset loading completed in {self.load_time:.2f} seconds")

    def _load_chunk(self, chunk):
        chunk_features = torch.FloatTensor(chunk[["x1", "x2", "x3"]].values)
        chunk_labels = torch.LongTensor(chunk["label"].values)
        return chunk_features, chunk_labels
