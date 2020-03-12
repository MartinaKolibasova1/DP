from __future__ import print_function
import torch

x = torch.tensor([5.5, 3])
print(x)

x = x.new_ones(5, 3, dtype=torch.double)  # new_* methods take in sizes
print(x)

x = torch.randn_like(x, dtype=torch.float)  # override dtype!
print(x)

print(x.size())

y = torch.rand(5, 3)
print(x + y)

print(torch.add(x, y))

result = torch.empty(5, 3)
torch.add(x, y, out=result)
print(result)

# adds x to y
y.add_(x)
print(y)

## Any operation that mutates a tensor in-place is post-fixed with an _. For example: x.copy_(y), x.t_(), will change x.

print(x[:, 1])

# Resizing:
# If you want to resize/reshape tensor, you can use torch.view:

x = torch.randn(4, 4)
y = x.view(16)
z = x.view(-1, 8)  # the size -1 is inferred from other dimensions
print(x.size(), y.size(), z.size())

# If you have a one element tensor, use .item() to get the value as a Python number

x = torch.randn(1)
print(x)
print(x.item())

# Converting a Torch Tensor to a NumPy array and vice versa is a breeze.

# The Torch Tensor and NumPy array will share their underlying memory locations (if the Torch Tensor is on CPU), and changing one will change the other

# Converting a Torch Tensor to a NumPy Array

a = torch.ones(5)
print(a)
b = a.numpy()
print(b)

a.add_(1)
print(a)
print(b)

# Converting NumPy Array to Torch Tensor
import numpy as np

a = np.ones(5)
b = torch.from_numpy(a)
np.add(a, 1, out=a)
print(a)
print(b)

if torch.cuda.is_available():
    device = torch.device("cuda")  # a CUDA device object
    y = torch.ones_like(x, device=device)  # directly create a tensor on GPU
    x = x.to(device)  # or just use strings ``.to("cuda")``
    z = x + y
    print(z)
    print(z.to("cpu", torch.double))
else:
    print("cuda is not available")

# Tensor
# torch.Tensor is the central class of the package. If you set its attribute .requires_grad as True,
# it starts to track all operations on it. When you finish your computation you can call .backward()
# and have all the gradients computed automatically. The gradient for this tensor will be accumulated into
# .grad attribute.

# To stop a tensor from tracking history, you can call .detach() to detach it from the computation history, and to prevent future computation from being tracked.

x = torch.ones(2, 2, requires_grad=True)
print(x)