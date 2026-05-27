

import torch


def interpret_values(unit_values_collection, device = None, dtype = None):
    
    if not isinstance(unit_values_collection, torch.Tensor):        
        return torch.stack(unit_values_collection, dim=0).to(device)  # Stack tensors along a new dimension (dimension 0)

    else:
        return unit_values_collection.view(-1).to(device) 
        

def interpret_values(unit_values_collection, device):
    
    if not isinstance(unit_values_collection, torch.Tensor):        
        return torch.stack(unit_values_collection, dim=0).to(device)  # Stack tensors along a new dimension (dimension 0)

    else:
        return unit_values_collection.to(device)
