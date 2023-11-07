
<img width="596" alt="Screen Shot 2023-10-29 at 10 47 24 AM" src="https://github.com/daichaoyi/DeepLearning_CS7643/assets/50822172/8812f12b-ba08-43c8-b04c-69f4ef201586">
<img width="600" alt="Screen Shot 2023-10-29 at 10 47 59 AM" src="https://github.com/daichaoyi/DeepLearning_CS7643/assets/50822172/61865761-8a7e-44c1-8033-acc41fceb8b5">

The LLM has the application in predictive typing; speech recognition, grammar correction.

<img width="699" alt="Screen Shot 2023-10-29 at 10 58 35 AM" src="https://github.com/daichaoyi/DeepLearning_CS7643/assets/50822172/ddb7b5c3-085a-4105-a63f-df2e23d88d13">


<img width="715" alt="Screen Shot 2023-10-29 at 12 01 08 PM" src="https://github.com/daichaoyi/DeepLearning_CS7643/assets/50822172/e1b90986-b7cb-4230-bf04-8f9cbad2018d">

In RNN, h_t is a state variable, h_t is a function of the previous state, and the input variable x_t.

<img width="705" alt="Screen Shot 2023-10-29 at 12 07 16 PM" src="https://github.com/daichaoyi/DeepLearning_CS7643/assets/50822172/0ee7c343-a793-49de-9ccb-6b827d5e6b5b">

<img width="706" alt="Screen Shot 2023-10-29 at 12 13 12 PM" src="https://github.com/daichaoyi/DeepLearning_CS7643/assets/50822172/753663b2-f3e3-4f7f-adc4-aa418524e9ed">

'vanilla' RNN is difficult due to vanishing gradients.
$|W_{\theta}|<1$   LSTM was created to alleviate vanishing gradients.

<img width="734" alt="Screen Shot 2023-10-29 at 12 31 18 PM" src="https://github.com/daichaoyi/DeepLearning_CS7643/assets/50822172/7a0e7d90-ee60-4eed-903b-8765d75eba40">


What is embedding? A mapping between objects to vector, Generally, we want that functio to create a map such that similar objects are grouped together. 

How is graph embedding learned? We optimize the objective that connected nodes have more similar embeddings than the unconnected node via gradient descent.

The perplexity measures the amount of “randomness” in our model. the lower is better. 
P(“a red fox.”) = P(“a”) * P(“red” | “a”) * P(“fox” | “a red”) * P(“.” | “a red fox”)

What are language models fundamentally used for?  To estimate the probability of a sequences of words given all the preceding words.

What are four problems that arise if you try to use MLPs/FC networks for modeling sequential data? 1. cannot support variable sized sequences as inputs or outputs  2. no inherent temporal structure  3. No practical way of holding state  4. the size of the network grows with the maximum alloed input or output sequence

what is masked language modeling? It is auxiliary task, but it can help us achieve better performance by finding good initial parameter of the model.

What are the main four component of LSTM? input gate, forget gate, cell state, output gate.

What is the cell state in LSTM?  You can think of it as the “memory” of the network. The cell state, in theory, can carry relevant information throughout the processing of the sequence.

What are the major four component of RNN? 1. Input 2. Hidden state  3. Weight/Parameters  4. Output

Why is the use of fully connected layers/MLPs problematic for sequential/time-series data? in fully connected layer, the biases and weights are independent. prone to overfit.

What roles did hidden state play in RNN? it's a contextual vector at time t that acts as the 'memory' of the past states, it is calculated as the function of the current input(x_t) and the past hidden state h(t-1).

RNN update rule: input and hidden are fed through 1, linear layer, 2 followed by non-linear tanh, 3 multiplied by output linear layer to get logits, 4 softmax to get probabilities. 

RNN how is loss calculated？ loss is summed over time

RNN: hidden weights are shared throughout the sequence. 

RNN cons: 1 Vanishing gradients 2 Exploding gradients 3 Limited “memory” - Can handle short-term dependencies but can’t handle long term
4 Lack control over memory - unlike LSTM, does not have mechanism to control what information should be kept.

RNN pros: smaller parameter size

LSTM update rules: FICO 1 forget gate 2 input gate 3 cell gate 4 output gate

LSTM what component is passed through the layers:cell (which represents memory)
forget gate, decide what information from the cell should be discard.
input gate, update the inforamtion with the cell gate
output gate, decides the next hidden state based on the cell state

GRU main difference to LSTM: single gate controls forget and cell state update

gradient clipping:control gradiant explode

Why does MLP not working for modeling sequence? 1. cannot support variable sized inputs 2. no inherent temporal structure 3. cannot maintain 'state' of sequence

application of languate model: predictive typing  ASR  grammar correction

Teacher forcing: During training, the model is also fed with the true target sequence, not its own generated output.

knowledge distillation: Smaller model (student) learns to mimic the predictions of a larger, more complex model (teacher) by transferring its knowledge

Knowledge distillation: Teacher student loss CE loss between teacher and student 

GloVe is an unsupervised learning algorithm for obtaining vector representations for words. Is was quite similar to Word2Vec method.

The skip-gram model is a way of teaching a computer to understand the meaning of words based on the context they are used in.
predict the context based on the middle word.
In contrast, the word of bag(CBOW) predict the middle word based on the context.

skip-gram identity activation --> softmax activation --> cross-entropy loss function


Intrinsic evaluation:
• Evaluation on a specific, intermediate task
• Fast to compute performance
• Helps understand subsystem
• Needs positive correlation with real task to determine usefulness


Extrinsic evaluation:
• Is the evaluation on a real task
• Can be slow to compute performance
• Unclear if subsystem is the problem, other subsystems, or internal
interactions
• If replacing subsystem improves performance, the change is likely good











