from utils import device

def unwrap_input(outputs, inputs):
    return outputs, inputs['class_id'].to(device())

def accuracy(inputs, targets):
    "Compute accuracy with `target` when `inputs` is bs * n_classes."
    n = targets.shape[0]
    inputs = inputs.argmax(dim=-1).view(n,-1)
    targets = targets.view(n,-1)
    return (inputs==targets).float().mean()
