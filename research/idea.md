This document is a list of the ideas I have when first starting this project.

# Working session 1
### Embedding SAE
*General idea:* Can we use a SAE to figure out what each value in a Large Language model's latent space means?
- Each dimension in a model's latent space represents some abstract concept
- These concepts aren't necessarily traditional "human" concepts
- Is there a way to establish a relationship between a single (or multiple) latent feature vectors and a more human-understandable meaning?
- If so, do different versions of the same model (GPT-4.1 vs GPT-4.1 mini) represent concepts in similar ways?
- Do different models by the same company represent concepts in similar ways?
- What about models from other companies? 
- Other languages?
- Other types of models? 
- **How can this concept be used to detect hallucinations, lies, etc**

*Methods:*
- Capture a model's embeddings for various input tokens
	- *This may not work, as words like 'explain' are made of two tokens and may not be accurately represented in the embeddings*
	- *However, small words like 'hello' are common enough to warrant their own embedding*
	- **Fix:** Use the model's final context vector instead of the first embedding layer, will capture semantic data (eg. turn left vs they left)
- Train a sparse auto encoder on these embeddings
- See how the model interprets the token embeddings and see if it recognizes patterns/groupings
	- *I believe this is done by looking at the model's sparse hidden activations, need to do more research to ensure*
	- *Other option is looking at the decoder weights*
- Apply the model to different LLMs within the same family/outside the family/outside the company and see how it responds



### After running this through ChatGPT 5:
$\star$ - A word/acronym I don't know/line where I'm confused
$\checkmark$ - A word/acronym I now know
Notes:
- Experiment on different layers (earlier layers are more syntactic, later layers are more semantic)$\star$$\checkmark$
- Use reconstruction loss and sparsity regularization (L1 or KL divergence on activations)$\star$$\checkmark$
- Try constraining weights to be nonnegative for more interpretable factors (NMF)$\star$$\checkmark$
- Compare SAE codes to PCA/ICA/NMV baselines to prove added value$\star$
- Probe sparse features by:
	- Activation maximization
	- Nearest neighbors
	- Human labeling
	- Auto concept discovery
- Feature alignment between models:
	- Canonical Correlation Analysis$\star$
	- Procrustes alignment$\star$
	- Centered Kernel Alignment$\star$
	- *Are concepts represented similarly?*

Challenges:
1. SAE features can be hard to name
	- Look to quantify stability and distinctiveness
2. "Does this neuron mean X" can be fuzzy
	- Combine human judgement with quantitative alignment metrics

 --- 
# Working session 2:
*I'm realizing this has just become a general project log. Not a bad thing!*

After re-reading last session's notes, I don't think looking at the model's embedding space will work
- As data model passes through the model, it changes the "meaning" of token embeddings significantly
- The embedding of a token *does not* represent the entire meaning of the word
	- Sorta more "what it could be"
- Hidden activation layers further down the network will more accurately represent the sum total meaning a model has assigned to a token
*TLDR: Move to the end of the model for a first try, then see if earlier layers retain the same "token understanding"*

Syntactic features: Word order, parts of speech, capitalization
- Grammar and "proper english"
Semantic features: flying bat vs baseball bat
Reconstruction loss: How good the SAE is at rebuilding its input (mean squared error)
Sparsity regularization: 
- Normal autoencoders distribute information across the network
	- Effects: No one or two unit will be a "meaning detector"
	- Fix: Use a regularizer to push the model into using less units
- L1 penalty: push activations to zero <- encourages sparsity
	- Question: What if the model learns to make all features go through one neuron?
	- Answer: *This was immediately answered by my next term 
- KL divergence penalty: Pushes the model to have each neuron only activate part of the time
	- This means the model can't just learn to push all values through a couple units and cheat the evaluation function
Nonnegative constraints (NMF-style)
- Nonnegative features can't cancel each other out
	- $dog \neq -cat$
- NMF: Nonnegative Matrix Factorization
	- Method to push representations into a parts-based design instead of something weird
	- $E(face)=E(mouth)+E(eyes)+E(nose)$ instead of $E(hitler)+E(italy)-E(germany)=E(mussolini)$
	- Question: Will this effect things that *require* contrast?
		- eg. yes vs no
Baselines:
- PCA
	- Principal Component Analysis
	- Looks for directions of maximum variance
- ICA
	- Independent Component Analysis
	- Looks for directions that are statistically independent
- NMF
	- Nonnegative Matrix Factorization (covered this already)
- COME BACK TO THIS [[MATS-9.0/research/todo|todo]]
Activation Maximization
- Find an input that maximizes a given feature
- This is what will help see what a feature looks for
CCA
- Canonical Correlation Analysis
- Compares two sets of vectors
- *This might be for later*
	- Once we start looking at multi-model features
Procrustes Alignment
- Aligns two vector spaces
- $E(dog) \approx E(chien)$
- *Also maybe for later*
	- Looking at aligning features across models
CKA
- Centered Kernel Alignment
- Similarity measure for comparing representation spaces
- Good for comparing models
