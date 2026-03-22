
from torch.utils.data import Dataset

class NextTokenDataset(Dataset):
    def __init__(self, data_path):
        self.tokenized_texts = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    tokens = list(map(int, line.split(',')))
                    self.tokenized_texts.append(tokens)

    def __len__(self):
        return len(self.tokenized_texts)

    def __getitem__(self, idx):
        return {'text': self.tokenized_texts[idx]}