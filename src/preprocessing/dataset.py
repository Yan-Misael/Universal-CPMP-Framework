import torch
import h5py
from torch.utils.data import Dataset
from settings import DATA_FOLDER
import os
import numpy as np

class H5Dataset(Dataset):
    def __init__(self, filepath):
        self.filepath = filepath
        self.name = os.path.basename(filepath)
        self.file = None
        
        with h5py.File(self.filepath, "r") as f:
            self.dataset_len = len(f["Y"])

    def _open_file(self):
        self.file = h5py.File(self.filepath, "r")
        self.G = self.file["G"]
        self.P = self.file["P"]
        self.I = self.file["I"]
        self.S = self.file["S"]
        self.H = self.file["H"]
        self.Y = self.file["Y"]
        self.C = self.file["C"]
        
    def close(self):
        if self.file is not None:
            self.file.close()
            self.file = None

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, idx):
        if self.file is None:
            self._open_file()
            
        return (
            torch.from_numpy(self.G[idx]),
            torch.from_numpy(self.P[idx]),
            torch.from_numpy(self.I[idx]),
            self.S[idx],
            self.H[idx],
            torch.from_numpy(self.Y[idx].astype(float))
        )

def load_dataset(filepath):
    dataset = H5Dataset(DATA_FOLDER / filepath)
    print(f"Dataset {dataset.name} cargado con {len(dataset)} muestras.")
    return dataset

def load_data(filepath):
    dataset = H5Dataset(filepath)
    dataset._open_file() 
    
    data = {
        'G': dataset.G[:],
        'P': dataset.P[:],
        'I': dataset.I[:],
        'S': dataset.S[:],
        'H': dataset.H[:],
        'Y': dataset.Y[:],
        'C': dataset.C[:]
    }
    
    dataset.close()
    return data

def generate_dataset(data_files, output_name, min_cost, max_cost, max_size):
    output_path = DATA_FOLDER / f"{output_name}.data"
    
    all_data = {key: [] for key in ['G', 'P', 'I', 'S', 'H', 'Y', 'C']}
    
    for data_file in data_files:
        path = str(DATA_FOLDER / data_file) + ".data"
        if os.path.exists(path):
            data = load_data(path)
            for key in all_data:
                all_data[key].append(data[key])
        else:
            print(f"Archivo no encontrado: {path}")

    if not all_data['G']:
        return

    with h5py.File(output_path, "w") as f:
        # Concatenamos primero para poder filtrar sobre el total
        combined_data = {key: np.concatenate(all_data[key], axis=0) for key in all_data}
        
        # Filtro por costo: min_cost <= data['C'] <= max_cost
        costs = combined_data['C']
        mask = (costs >= min_cost) & (costs <= max_cost)
        
        # Aplicar máscara y limitar por max_size
        for key in combined_data:
            filtered_data = combined_data[key][mask]
            final_data = filtered_data[:max_size]
            f.create_dataset(key, data=final_data)
            
    print(f"Dataset generado exitosamente en: {output_path} (Tamaño {len(final_data)})")