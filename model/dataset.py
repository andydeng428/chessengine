# dataset.py

import torch
from torch.utils.data import Dataset
import h5py

class ChessDataset(Dataset):
    def __init__(self, h5_file_path):
        self.file = h5py.File(h5_file_path, 'r')
        self.piece_masks = self.file['piece_masks']
        self.scalar_features = self.file['scalar_features']
        self.policy_targets = self.file['policy_target']
        self.value_targets = self.file['value_target'] 
        self.length = self.piece_masks.shape[0]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        piece_mask = torch.tensor(self.piece_masks[idx], dtype=torch.float32)
        scalar_feature = torch.tensor(self.scalar_features[idx], dtype=torch.float32)
        scalar_planes = scalar_feature.view(5, 1, 1).expand(-1, 8, 8)
        input_tensor = torch.cat((piece_mask, scalar_planes), dim=0)

        # Policy target
        policy_target = torch.tensor(self.policy_targets[idx], dtype=torch.float32)
        policy_index = torch.argmax(policy_target).item()

        # Value target
        value_target = torch.tensor(self.value_targets[idx], dtype=torch.float32)

        return input_tensor, policy_index, value_target
