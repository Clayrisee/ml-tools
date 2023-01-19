import pandas as pd
from PIL import Image
import pandas as pd
from torch.utils.data import Dataset
import os

class CSVDataset(Dataset):
    def __init__(self, root_dir, csv_file, transform=None):
        super(CSVDataset, self).__init__()
        self.root_dir = root_dir
        self.data = pd.read_csv(os.path.join(self.root_dir, csv_file))
        self.transform = transform
            
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index) :

        img_name = self.data.iloc[index, 0]
        format = self.data.iloc[index, 1].lower()
        folder = self.data.iloc[index, 2]
        img_name = f"{img_name}.{format}"
        img_path = os.path.join(self.root_dir, folder, 'images', img_name)

        img = Image.open(img_path).convert('RGB')
        label = float(self.data.iloc[index, -1])
        
        if self.transform:
            img = self.transform(img)
        
        return img, label