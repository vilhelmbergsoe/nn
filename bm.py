import torch
# import torch.nn as nn

# class XORNet(nn.Module):
#     def __init__(self):
#         super(XORNet, self).__init__()
#         self.fl1 = nn.Linear(2, 1)
#         self.fl2 = nn.Linear(1, 2)

#         self.bl1 = nn.Linear(2, 1)
#         self.bl2 = nn.Linear(1, 2)

#         self.hl1 = nn.Linear(4, 1)

#     def forward(self, x):
#         flx = torch.relu(self.fl1(x))
#         blx = torch.relu(self.bl1(x))
#         flx = torch.sigmoid(self.fl2(flx))
#         blx = torch.sigmoid(self.bl2(blx))

#         merged_tensor = torch.cat((flx, blx), dim=1)

#         x = torch.tanh(self.hl1(merged_tensor))

#         return x

def main():
    # nn = XORNet()

    # for i in range(10000):
    #     x = torch.tensor([[1., 0.]], dtype=torch.float32)
    #     output = nn(x)
    # for i in range(100000):
    # x = torch.tensor([[2.0,1.0], [3.0, 1.0]], requires_grad=True)
    # y = torch.tensor([[-2.0,1.5], [3.5, 1.5]], requires_grad=True)
    # z = x+y
    # g = z * torch.tensor(2.0, requires_grad=True);
    # g = g.sum()
    g = torch.tensor(4.0, requires_grad=True).square()
    g.backward()
    print(g)

if __name__ == "__main__":
    main()
