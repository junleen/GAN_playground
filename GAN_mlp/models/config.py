
configurations = {
    'G': dict(
        input_dim=128,
        output_dim=28*28,
        hidden_layers=2,
        hidden_units=256
    ),
    'D': dict(
        input_dim=28*28,
        hidden_layers=2,
        hidden_units=512,
    ),
    'SEED': 1024,
    'lr_G': 0.001,
    'lr_D': 0.001,
    'batch_size': 64,
}
