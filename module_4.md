torch.unsqueeze was used to reshape the tensor. 

x = torch.tensor([1, 2, 3, 4])
torch.unsqueeze(x, 0)
tensor([[ 1,  2,  3,  4]])
torch.unsqueeze(x, 1)

The common practice was has input into encoder, than we obtain context, use decoder to read context, then deliver the output.

<img width="780" alt="Screen Shot 2023-12-17 at 10 15 00 AM" src="https://github.com/daichaoyi/DeepLearning_CS7643/assets/50822172/d5aae7df-c469-4253-9f6f-0c11eadab7bb">

The architecture of RNN, for encoder, the input was the original text and hidden state, then it pass the hidden state and output to the decoder.  

Each word input that goes to encoder will generate a context, this context will be passed to decoder. But each word has the same contribution to the context. This does not make sense, therefore, we have the motivation for the attention.
With attention, the hidden state was passed to decoder. In the decoding process, the attention will find out the most relevant input hidden state, then use the softmax to generate output. 

<img width="759" alt="Screen Shot 2023-12-17 at 1 45 12 PM" src="https://github.com/daichaoyi/DeepLearning_CS7643/assets/50822172/dc1c0c84-43fa-4ecd-82d0-3899e94ae8fe">

In the framework of attention, the encoder translate the source word into hidden state. In the framework of the simple Encoder-Decoder, where only the last hidden state was passed to decoder as the context vector. In the attention model, all the hidden state was passed to decoder. In the decoding process, attention has an additional step: Looking for the most relevant input hidden state at the current time. (Usually, the relavance was computed through softmax).  
In the encoder-decoder structure, the context vector is:
$y_i=g(C, y_1, y_2, ..., y_{i-1})$

In the attention framework, the context vector is:
$y_i=g(C_i, y_1, y_2, ..., y_{i-1})$  $C_i$ are the hidden states. 
$C_i=\sum_{j=1}^{T_x} a_{ij} h_j$  $a_{ij}$ is the attention weight,

$e_{ij}=\alpha(s_{i-1}, h_j)$

$\`a_{ij}=[\frac{exp(e_{ij})}{\sum^{T_x}_{k=1}] exp(e_{ik})}`$

$C_{i}= \sum_{j=1}{T_x}a_{ij}h_j$

$S_i=f(s_{i-1}, y_{i-1}, c_i)$

$y_i= g(y_{i-1}, s_i, c_i)$

<img width="701" alt="Screen Shot 2023-12-23 at 10 07 44 PM" src="https://github.com/daichaoyi/DeepLearning_CS7643/assets/50822172/b9ff8113-4bfd-46fd-85a7-3955d23a29e6">

The workflow of attention:

1. The decoder RNN takes the in the embebdding of the <END> token, and an initial decoder hidden state.
2. RNN take the input, generate output(this part was discarded) and a new hidden state vector(h4).
3. We use the encoder hidden states and the h4 vector to calculate context vector(C4).
4. concatenate h4 and C4 into one vector.
5. pass this vector through a feedforward neural network
6. The output of the feedforward neural networks indicates the output word of this time step.
7. Repeat for the next time steps.

打分过程在decoder的每个timestep完成。一起来看下attention如何工作的：

<END>和初始化的decoder hidden state输入到attention decoder RNN。
RNN产生新的hidden state (h4)。
Attention 工作：使用encoder的hidden state和h4计算当前时刻的context vector (C4)。
将h4和C4拼接。
将第4步产生的向量输入的前向神经网络（这个网络和模型一起训练）
前向神经网络的输出就是当前时刻的输出word。
重复以上6步。
第2步中输出的h4不能直接使用，而是需要通过attention机制，与encoder中的各个hidden states计算后，再通过FNN输出当前timestep的output。
<img width="730" alt="Screen Shot 2023-12-23 at 10 08 52 PM" src="https://github.com/daichaoyi/DeepLearning_CS7643/assets/50822172/c187a735-d76b-461b-bf8c-45d9ef6075fd">








