torch.unsqueeze was used to reshape the tensor. 

x = torch.tensor([1, 2, 3, 4])
torch.unsqueeze(x, 0)
tensor([[ 1,  2,  3,  4]])
torch.unsqueeze(x, 1)

The common practice was has input into encoder, than we obtain context, use decoder to read context, then deliver the output.

<img width="780" alt="Screen Shot 2023-12-17 at 10 15 00 AM" src="https://github.com/daichaoyi/DeepLearning_CS7643/assets/50822172/d5aae7df-c469-4253-9f6f-0c11eadab7bb">





