import torch.utils.data
from data.base_data_loader import BaseDataLoader


def CreateDataset(opt):
    dataset = None
    if opt.dataset_mode == 'aligned_deblur':
        from data.aligned_dataset_deblur import AlignedDataset
        dataset = AlignedDataset()
    elif opt.dataset_mode == 'aligned_deblur2':
        from data.aligned_dataset_deblur2 import AlignedDataset
        dataset = AlignedDataset()
    elif opt.dataset_mode == 'aligned_deblur_test':
        from data.aligned_dataset_deblur_test import AlignedDataset
        dataset = AlignedDataset()
    elif opt.dataset_mode == 'aligned_sr':
        from data.aligned_dataset_sr import AlignedDataset
        dataset = AlignedDataset()
    elif opt.dataset_mode == 'aligned_dehaze':
        from data.aligned_dataset_dehaze import AlignedDataset
        dataset = AlignedDataset()
    elif opt.dataset_mode == 'aligned_dehaze_test':
        from data.aligned_dataset_dehaze_test import AlignedDataset
        dataset = AlignedDataset()
    elif opt.dataset_mode == 'unaligned':
        from data.unaligned_dataset import UnalignedDataset
        dataset = UnalignedDataset()
    elif opt.dataset_mode == 'single':
        from data.single_dataset import SingleDataset
        dataset = SingleDataset()
    else:
        raise ValueError("Dataset [%s] not recognized." % opt.dataset_mode)

    print("dataset [%s] was created" % (dataset.name()))
    dataset.initialize(opt)
    return dataset


class CustomDatasetDataLoader(BaseDataLoader):
    def name(self):
        return 'CustomDatasetDataLoader'

    def initialize(self, opt):
        BaseDataLoader.initialize(self, opt)
        self.dataset = CreateDataset(opt)
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=opt.batchSize,
            shuffle=not opt.serial_batches,
            num_workers=int(opt.nThreads))

    def load_data(self):
        return self

    def __len__(self):
        return min(len(self.dataset), self.opt.max_dataset_size)

    def __iter__(self):
        for i, data in enumerate(self.dataloader):
            if i >= self.opt.max_dataset_size:
                break
            yield data
