torch.unsqueeze was used to reshape the tensor. 

x = torch.tensor([1, 2, 3, 4])
torch.unsqueeze(x, 0)
tensor([[ 1,  2,  3,  4]])
torch.unsqueeze(x, 1)

The common practice was has input into encoder, than we obtain context, use decoder to read context, then deliver the output.

<img width="780" alt="Screen Shot 2023-12-17 at 10 15 00 AM" src="https://github.com/daichaoyi/DeepLearning_CS7643/assets/50822172/d5aae7df-c469-4253-9f6f-0c11eadab7bb">

The architecture of RNN, for encoder, the input was the original text and hidden state, then it pass the hidden state and output to the decoder.  

Each word input that goes to encoder will generate a context, this context will be passed to decoder. But each word has the same weight to the context. This does not make sense, therefore, we have the motivation for the attention.

<img width="759" alt="Screen Shot 2023-12-17 at 1 45 12 PM" src="https://github.com/daichaoyi/DeepLearning_CS7643/assets/50822172/dc1c0c84-43fa-4ecd-82d0-3899e94ae8fe">






