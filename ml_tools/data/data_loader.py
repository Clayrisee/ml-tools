from data.base import DataModuleBase
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder


class ImageFolderDataModule(DataModuleBase):
    def __init__(self, cfg, train_val_transforms, test_transforms: transforms.Compose):
        
        super().__init__(cfg)

        self.train_bs = cfg['train']['batch_size']
        self.val_bs = cfg['val']['batch_size']
        self.test_bs = cfg['test']['batch_size']

        self.train_val_transforms = train_val_transforms

        self.test_transforms = test_transforms

        self.prepare_dataset(cfg=cfg)
    
    def prepare_dataset(self, cfg):
        self.train_set = ImageFolder(cfg['dataset']['train_dir'],  transform=self.train_val_transforms)
        self.val_set = ImageFolder(cfg['dataset']['val_dir'],  transform=self.train_vall_transforms)
        self.test_set = ImageFolder(cfg['dataset']['test_dir'], transform=self.test_transforms)

    def train_dataloader(self):
        kwargs = dict(
            batch_size=self.train_bs,
            shuffle=True,
            num_workers=2
        )
        self.train_dl = DataLoader(self.train_set, pin_memory=True,**kwargs)
        return self.train_dl

    def val_dataloader(self):
        kwargs = dict(
            batch_size=self.train_bs,
            shuffle=True,
            num_workers=2
        )
        self.val_dl = DataLoader(self.val_set,  pin_memory=True, **kwargs)
        return self.val_dl

    def test_dataloader(self):
        kwargs = dict(
            batch_size=self.test_bs,
            shuffle=True,
            num_workers=2
        )
        self.test_dl = DataLoader(self.test_set,  pin_memory=True, **kwargs)
        return self.test_dl