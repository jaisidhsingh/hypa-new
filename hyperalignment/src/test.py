import torch


x = torch.randn((3, 256))
y = torch.randn((3, 256))

z = torch.randn(3, 256)
u = torch.randn(3, 256)

X = torch.cat((x, z), dim=0)
Y = torch.cat((y, u), dim=0)

s1 = x @ y.T
s2 = X @ Y.T

print(s1)
print(s2)