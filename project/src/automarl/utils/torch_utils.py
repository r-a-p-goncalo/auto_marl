

import gc

import torch


def get_find_all_tensors():
    
    gc.collect()
    
    # Find all live tensors
    all_tensors = []
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj):
                all_tensors.append(obj)
        except Exception:
            pass
        
    return all_tensors
        

def print_all_tensors_info():
    
    all_tensors = get_find_all_tensors()

    for tensor in all_tensors:
        tensor : torch.Tensor = tensor

        print(f"Tensor with shape {tensor.shape} has version {tensor._version}")
        

def print_all_tensors_info_with_version():
    
    all_tensors = get_find_all_tensors()

    for tensor in all_tensors:
        tensor : torch.Tensor = tensor
        if tensor._version > 0:
            print(f"Tensor with shape {tensor.shape} has version {tensor._version}")
            
            
def print_graph(tensor, indent=0):
    """Recursively prints the autograd graph."""
    if tensor is None:
        return
    print("  " * indent + f"{type(tensor).__name__}")
    if hasattr(tensor, 'next_functions'):
        for child, _ in tensor.next_functions:
            print_graph(child, indent + 1)
            
            
def print_graph_with_saved_tensors(fn, indent=0, visited=None):
    if fn is None:
        return

    if visited is None:
        visited = set()
    if fn in visited:
        return
    visited.add(fn)

    prefix = "    " * indent
    print(f"{prefix}{type(fn).__name__}")

    if hasattr(fn, "saved_tensors"):
        for t in fn.saved_tensors:
            if t is None:
                continue
            print(f"{prefix}  saved tensor: shape={tuple(t.shape)}, dtype={t.dtype}, device={t.device}, requires_grad={t.requires_grad}")
            # If you want, print tensor data
            # print(t)

    if hasattr(fn, "next_functions"):
        for u, _ in fn.next_functions:
            if u is not None:
                print_graph_with_saved_tensors(u, indent + 1, visited)
            

def render_computational_graph(tensor):
    
    from torchviz import make_dot

    make_dot(tensor).render("ppo_loss_graph", format="pdf")