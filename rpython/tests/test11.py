import torch
import torch.nn as nn
import torch.nn.functional as F

a = torch.tensor(3)
b = torch.tensor(4)
c = a.add(b)
print(c)  

d = a.mul(b)
print(d)  

e = d.sub(c)
print(e) 

z = torch.zeros(5)
print(z)

o = torch.ones(5)
print(o)

r = torch.arange(6)
print(r)

s = torch.ones(10).sum()
print(s)

m = torch.arange(4).float().mean()
print(m)

p = torch.tensor(2).pow(10)
print(p)

chain = torch.ones(4).mul(torch.ones(4)).sum()
print(chain)

mat = torch.arange(6).float().reshape(2, 3)
print(mat)

m1 = torch.ones(2, 2)
m2 = torch.ones(2, 2).mul(2)
mm = torch.matmul(m1, m2)
print(mm)

rng = torch.arange(1, 6)
print(rng.max())
print(rng.min())

w = torch.tensor(3.0, requires_grad=True)
loss = w.mul(w)
loss.backward()
print(w.grad)  # tensor(6.)

detached = loss.detach().item()
print(detached)

acc = torch.tensor(0)
i = 0
while i < 5:
    acc = acc.add(torch.tensor(i))
    i = i + 1
print(acc)
print(acc.item())

big = torch.zeros(3, 4, 5)
print(big.numel())
print(big.size())

linear = nn.Linear(4, 2)
inp = torch.ones(1, 4)
out = linear(inp) 
print(out.shape)

neg = torch.tensor(-5.0)
print(F.relu(neg))

pos = torch.tensor(3.0)
print(F.relu(pos))

wx = torch.tensor(1.0, requires_grad=True)
target = torch.tensor(2.0)
pred = wx.mul(torch.tensor(1.0))
diff = pred.sub(target)
mse = diff.mul(diff)
mse.backward()
print(wx.grad)