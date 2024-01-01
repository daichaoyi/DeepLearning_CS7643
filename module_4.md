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

<img width="730" alt="Screen Shot 2023-12-23 at 10 08 52 PM" src="https://github.com/daichaoyi/DeepLearning_CS7643/assets/50822172/c187a735-d76b-461b-bf8c-45d9ef6075fd">

Why transformer?
1. RNN, CNN, the sequence to compute is left to right, or right to left. (1) The time t result is contingent on the t-1 result, it will restrict the parallel computation (2) Transformer was constituted by encoding and decoding. For the encoding part, it was consists of 6 encoders, for the decoding part, it was consisted by 6 decoders. The output of encoder will be the input of decoder. 

<img width="706" alt="Screen Shot 2024-01-01 at 9 55 00 AM" src="https://github.com/daichaoyi/DeepLearning_CS7643/assets/50822172/4c84c1fe-5c62-4119-928d-317c9705e60e">

Transformer was formed by encoding and decoding. In the encoding part, it was formed by 6 encoders, in the decoding part, it was formed by 6 decoders. The output of encoders works as the input of decoder. 


<img width="714" alt="Screen Shot 2024-01-01 at 10 18 20 AM" src="https://github.com/daichaoyi/DeepLearning_CS7643/assets/50822172/a6573901-8aa6-4e07-827d-1dad81b1dd1d">

Encoder was formed by self-attention layer and feedforward Neural Network. FFNN process each output of the self-attention layer independently. 

<img width="694" alt="Screen Shot 2024-01-01 at 10 25 55 AM" src="https://github.com/daichaoyi/DeepLearning_CS7643/assets/50822172/af454791-3f1d-455e-a426-ac8eceeade80">

Decoder was formed by three part: self-attention, encoder-decoder attention, feed forward.
Decoder added a layer of Encoder-Decoder attention between self-attention and FFNN, this layer can let decoder focus on part of the input sentence. (which is similar to soft attention)

Self-Attention
Self-attention will help encoder assimulate the information of other words when processing the current input word. For example, when dealing with the "The animal didn't cross the street because it was too tired", the encoder will associate 'it' with the 'animal', it will let the encoder acquire more information. 

Self-attention是如何计算的？
对于每个输入的embedding vector（即每个word），会生成3个向量，分别是Query Vector，Key Vector, Value Vector。这三个向量是通过训练中产生的三个权重矩阵（需要初始化）与embedding vector相乘得到。值得注意的是，这三个权重矩阵的维度（一般为64，关于投影后维度为什么要压缩，可以参考transformer中multi-head attention中每个head为什么要进行降维？）比embedding和encoder input/output vector要小很多。这是为了让multi-head attention的计算保持稳定。

How to compute self-attention?
1. For each input embedding vector, it will generate 3 vectors, query vector, key vector and value vector. It was generated through multiplying three weighting matrix with the embedding vectors.
<img width="690" alt="Screen Shot 2024-01-01 at 4 18 32 PM" src="https://github.com/daichaoyi/DeepLearning_CS7643/assets/50822172/67dcd3c2-b717-4e95-bb81-6f9b0804f79d">

2. Calculate score,assume when we calculate the first word: 'Thinking', we need to compute the score for the input sentence and this word. This score represents how much focus was imposed on each word during the encoding. We use the dot product between key vector and remaining word's query vector.
   
<img width="721" alt="Screen Shot 2024-01-01 at 4 18 45 PM" src="https://github.com/daichaoyi/DeepLearning_CS7643/assets/50822172/9c951988-9d54-460b-a55d-16dea5687092">

3. score divided by n, $n=\sqrt{dim of key vector}$, it helps to obtain more stable gradients. Feed this result into softmax function, we get unified results. The most relevant word gets the highest score. 

 <img width="705" alt="Screen Shot 2024-01-01 at 6 12 02 PM" src="https://github.com/daichaoyi/DeepLearning_CS7643/assets/50822172/373576d9-8f29-45e7-8472-71cd4dd41f98">

4. Each value vector times the softmax score, then get the weighted value vector. It was for amplifying the more relavant vector, and squeeze the less relavant vector. 

5. Adds up the weighted value vectors from step 4, we get the self-attention layer for the current word.

<img width="770" alt="Screen Shot 2024-01-01 at 6 16 42 PM" src="https://github.com/daichaoyi/DeepLearning_CS7643/assets/50822172/1d032099-e852-4477-b3f5-932f7eff8c25">
























