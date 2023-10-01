# LLaMA2 from Scratch

This repository contains an implementation of the LLaMA 2 (Large Language Model Meta AI) model, a Generative Pretrained Transformer (GPT) variant. The implementation focuses on the model architecture and the inference process. The code is restructured and heavily commented to facilitate easy understanding of the key parts of the architecture.

## Model Features

- **RMS-Normalization:** RMSNorm is a simplification of the original layer normalization (LayerNorm). LayerNorm is a regularization technique that might handle the internal covariate shift issue so as to stabilize the layer activations and improve model convergence. It has been proved quite successful in LLaMA 2.

- **Activation Function:** LLaMA 2 uses the SwiGLU activation function instead of ReLU, leading to improved training performance.

- **Rotary Positional Embeddings (RoPE):** Inspired by the GPT-Neo-X project, LLaMA 2 incorporates rotary positional embeddings at each layer, enhancing the model's positional understanding.

- **Increased Context Length and Grouped-Query Attention (GQA):** LLaMA 2 model has a doubled context window (from 2048 to 4096 tokens) and employs grouped-query attention. This allows for better processing of long documents, chat histories, and summarization tasks.

## Implementation Highlights

### KV-Caching for Efficient Inference*

*KV-caching is a crucial optimization technique employed in this implementation to accelerate the inference process for Language Model (LM) decoding. During autoregressive decoding, where each token is predicted based on prior tokens, self-attention within the model is causal. This implies that a token's representation is computed based only on itself and the prior tokens, not the future ones.*

*In self-attention, the input sequence is projected using key, value, and query projections. The KV-cache efficiently stores the results of the key and value projections, eliminating the need for redundant computations in future decoding iterations. As a result, the representations of tokens that remain fixed during autoregressive decoding can be retrieved from the cache, significantly enhancing the inference speed.*

*This KV-caching technique is a key architectural feature that enhances the efficiency and speed of the LLaMA model during decoding.*

![](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/_images/kv-cache-optimization.png)


### Grouped-Query Attention (GQA) for Enhanced Efficiency

*The LLaMA 2 model incorporates a variation of the concept of Multi-Query Attention (MQA) proposed by Shazeer (2019), a refinement of the Multi-Head Attention (MHA) algorithm. MQA enhances the efficiency of attention mechanisms while maintaining minimal accuracy degradation.*

*In traditional multi-head attention, the entire attention computation is replicated h times, where h is the number of attention heads. However, GQA reduces computational redundancy by removing or significantly reducing the heads dimension (h) from the K and V values. In MQA, each "head" of the query value (Q) undergoes the same K and V transformation, optimizing the attention computation.*

*This refinement results in similar computational performance to MHA but significantly reduces the amount of data read/written from memory. As a consequence, GQA improves both performance (via an increase in arithmetic intensity) and memory space efficiency (via a decrease in the amount of KV-cache data stored), making it a valuable addition to the LLaMA architecture.*

![](https://pbs.twimg.com/media/FzjhZk5X0AYAs_-?format=jpg&name=4096x4096)

### Rotary Positional Embeddings for Enhanced Attention

*In the LLaMA 2 model, Rotary Positional Embeddings (RoPE) play a crucial role in enhancing attention mechanisms by incorporating positional information into the token representations. The concept of "attention" is powerful, but to ensure that the attention calculated is meaningful, tokens need to have a notion of position.*

*Position embeddings come in two main types: absolute and relative. Absolute position embeddings encode the absolute position of a word in the input phrase, while relative position embeddings encode the relative position between two words. These embeddings provide vital positional information that helps tokens understand their context in a sequence.*

*Rotary Positional Embeddings take a unique approach by leveraging rotation matrices to embed positional information. The goal is to ensure that the inner product of vectors q and k, at positions m and n, depends only on q, k, and their relative distance (m — n). The rotation matrix, where the angle is the vector’s position, is embedded into the original vector through matrix multiplication, aligning with this criterion.*

*This innovative approach to incorporating positional information enhances the model's ability to understand token relationships and context, contributing to improved attention mechanisms.*

![](https://pbs.twimg.com/media/FrqjrsmXoAQhr2R.jpg)

## Code Structure

- `model.py`: Contains the implementation of the LLaMA transformer model with detailed comments explaining each component and functionality.

- `inference.py`: Demonstrates how to use the trained LLaMA model for inference, providing insights into the input and output processing.

Feel free to explore the code, correct if any mistakes, and experiment with the LLaMA 2 model!
