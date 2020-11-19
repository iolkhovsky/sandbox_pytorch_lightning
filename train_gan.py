import pytorch_lightning as pl

from mnist_data_module import MNISTDataModule
from mnist_gan import GAN


dm = MNISTDataModule()
model = GAN(*dm.size())
trainer = pl.Trainer(gpus=0, max_epochs=10, progress_bar_refresh_rate=20)
trainer.fit(model, dm)
