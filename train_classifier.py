import pytorch_lightning as pl

from mnist_classifier import LitMNIST


model = LitMNIST()
trainer = pl.Trainer(gpus=0, max_epochs=3, progress_bar_refresh_rate=20)
trainer.fit(model)

trainer.test()
