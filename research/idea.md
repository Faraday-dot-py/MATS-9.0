This document is a list of the ideas I have when first starting this project.

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