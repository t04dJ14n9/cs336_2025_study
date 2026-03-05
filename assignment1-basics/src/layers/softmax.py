import torch

def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Compute the softmax function along a specified dimension.
    
    The softmax function transforms input values into a probability distribution
    where all values are in the range (0, 1) and sum to 1 along the specified dimension.
    
    Formula: softmax(x_i) = exp(x_i) / sum(exp(x_j)) for all j
    
    Args:
        x: Input tensor of any shape
        dim: The dimension along which to compute softmax
        
    Returns:
        Tensor of the same shape as input with softmax applied along the specified dimension
    """
    # Step 1: Find the maximum value along the specified dimension
    # keepdim=True preserves the dimension for broadcasting
    # [0] extracts the values (torch.max returns both values and indices)
    # This max will be used for numerical stability in the next step
    x_max = torch.max(x, dim=dim, keepdim=True)[0]
    
    # Step 2: Subtract the max from all elements for numerical stability
    # This prevents overflow when computing exp() of large numbers
    # Mathematically: softmax(x) = softmax(x - c) for any constant c
    # By choosing c = max(x), we ensure all values are <= 0 before exp()
    x_shifted = x - x_max
    
    # Step 3: Compute the exponential of the shifted values
    # Now all exp values are in range (0, 1] since x_shifted <= 0
    # This avoids numerical overflow that could occur with large positive values
    exp_x = torch.exp(x_shifted)
    
    # Step 4: Normalize by dividing by the sum of exponentials
    # This ensures the output sums to 1 along the specified dimension
    # keepdim=True ensures the sum has the same number of dimensions for broadcasting
    return exp_x / torch.sum(exp_x, dim=dim, keepdim=True)
