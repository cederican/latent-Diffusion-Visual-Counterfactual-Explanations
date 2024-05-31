class CLSConfig():
    run_name: str = 'DiffAECLS-05.28-Linear'
    batch_size: int = 128
    epochs: int = 2
    lr: float = 1e-4
    weight_decay: float = 0.01
    devices: list = [0]
    #input_dim: tuple = (256,256,256)