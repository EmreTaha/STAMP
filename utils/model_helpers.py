def convert_model_to_fp32(model):
    # Converts a mixed-precision model to fp32 for evaluation
    for p in model.parameters():
        p.data = p.data.float()
        if p.grad:
            p.grad.data = p.grad.data.float()