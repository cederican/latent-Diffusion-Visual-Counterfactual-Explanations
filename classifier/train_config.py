class CLSConfig():

    run_name: str = 'DiffAECLS-06.08-Linear'
    architecture: str = 'linear' # or 'linear' or 'res50
    batch_size: int = 128
    epochs: int = 4
    lr: float = 1e-4
    weight_decay: float = 0.01
    devices: list = [0]
