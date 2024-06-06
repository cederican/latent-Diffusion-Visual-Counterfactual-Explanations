class CLSConfig():
    run_name: str = 'DiffAECLS-06.02-RESNET50'
    architecture: str = 'res50' # or 'linear' or vgg16
    batch_size: int = 64
    epochs: int = 2
    lr: float = 1e-4
    weight_decay: float = 0.01
    devices: list = [0]
    