import torch

from src.model import SimpleCNN


def test_output_shape(): #The model should emit one logit per class.
    m = SimpleCNN(10)
    assert m(torch.randn(4, 3, 32, 32)).shape == (4, 10)


def test_gradients(): #Backprop should reach every parameter.
    m = SimpleCNN(10)
    out = m(torch.randn(1, 3, 32, 32))
    out.sum().backward()
    assert all(p.grad is not None for p in m.parameters())