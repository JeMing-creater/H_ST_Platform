from models.MIL import AttentionMIL

def get_model(config, **kwargs):
    if config.trainer.model == "MIL":
        model = AttentionMIL(input_dim=1024, hidden_dim=256)
    else:
        raise ValueError(f"Model {config.trainer.model} is not supported.")
    
    return model