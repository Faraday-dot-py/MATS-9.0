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

# Working session 3:
*I tested myself on the code from last session and it all still makes sense, which is a good sign!*

*It's cool seeing concepts of different algorithms playing out in ML*
- *Example: minimax branch pruning $\approx$ topk masks*

**COOL THING FOUND:**
Now that I have neuron activations, I can plot them with matplotlib, and I've noticed something interesting: The same neurons seem to be firing a lot! 

This is a screenshot of a graph of all the activations for 10 passes through the network
![[plottedGPT2Activations.png]]
What's interesting is there seems to be a collection of activations between neurons 300 and 500, as well as a couple spikes around neuron 50 and a smaller one around neuron 700. This happens no matter what slice of the dataset I use!

My theories as to why this is happening:
1. Neurons at the center of the network have the most potential for data to pass through them *\*likely not, as the residual stream isn't geometric (it has no "middle")*
2. Since the dataset is mainly Shakespeare, is there something about that style of writing that activates certain neurons more than others?
	- Has the model captured the "feel" of certain writing styles?
	- Method to test this:
		- Get another dataset (more diverse) and see how the model reacts to it
		- Get a multi-lingual model and see how it reacts to the same text but in two different languages
	- Possible that the token being grabbed (last token in the last hidden layer) is punctuation/"the concept of the end of a line"
		- Solution: Grab different tokens (second to last, third to last, (middle???))
Other interesting things to look at later:
- mean and std of dataset
	- Look for anisotropy (some dimensions with higher levels of "energy")
	- Especially with similar datasets (look at The Bee Movie script vs the Shrek script)
		- Different biased datasets would have certain anisotropic dimensions
		- Unbiased dataset should be isotropic
			- Idea: Uniformity (isotropic vs anisotropic) of a model is a good measure of its bias
			- Look at the anisotropic features and see what they represent
	- A sign that a concept is represented in that dimension
- Grab a random token instead of the last one
- Dataset whitening
- Look at post-LN activations instead of the last hidden layer pre-LN activations
	- Normalized representations might help with scale artifacts
	- Better representation of what the unembedding matrix gets
	- "Pre-thought layer vs thought layer vs speech layer"

# Working session 4:
*It's training time!!!*

*This session will be interesting as it's the first time I've training a large model like this. I've done smaller models but mostly pure unsupervised learning*

*I feel like work has progressed slowly due to me constantly taking breaks but looking back at my notes the lack of progress in amount of code is more than made up for by the amount of research and learning I've done*

*I also need a better dataset, that'll be my first task for today*

Dataset needs to be:
Large: 
- Not sure how large, would prefer a couple gigabytes at least
Broad:
- Dataset must be broad in two senses of the word:
	- Broad in topic:
		- Covers a wide range of datasets
		- SigmaLaw (law dataset) not allowed
	- Broad in type:
		- Not only question-answer datasets
		- Not only code datasets
		- Q/A datasets might be good for a first pass if they're broad enough in topic
			- Need to check how the activations of answering questions compares to the activations of problem solving
	- English:
		- I don't know other languages
		- Languages have semantic information that google translate won't tell me
			- Swear words are a great example of this

https://www.kaggle.com/datasets/wikimedia-foundation/wikipedia-structured-contents/data
Pros
- Dataset has a lot of data
- Chunked nicely
- Separated nicely
- Has links to other articles which might be nice for comparing activation maps
Cons
- Lots of metadata in files
	-Fix: requires preprocessing


# Working session 5:
*I thought I would be training last session, turns out it was just more data wrangling*

*Data wrangling and making evaluation functions (and learning how they work) are the main focuses for today*

##### Evaluation function notes
"Good" values (as per ChatGPT):
**Density:**
Good range: ~0.02-0.1
- Good enough for good reconstruction and interpretable codes
Too high: > 0.2-0.3
- Features are firing too often
- Codes aren't as sparse as they need to be
- Harder to interpret
- Fix: Increase `l1Strength` or smaller `topk`
Too low: < 0.01
- Less features firing
- Leads to weak reconstruction
- Too many dead features
- Fix: reduce `l1Strength` or shrink `codeDimensions`

**Dead Rate**
Good range: ~0.05-0.2
- Small amounts of dead features is fine
Too high: >0.3
- Too many neurons aren't firing
- Too much of the dictionary is wasted
- Fix: Shrink `codeDims`, reduce `l1Strength`, add more training data
Too low: ~0
- Every feature is firing at least once
- Can be good (means the model utilizes the whole dictionary)
- Check if density is too high

**Gini coefficient**
Good range: ~0.2-0.6
- Mix of broadly useful features and niche specialists
- Could imply a good mix of broad concepts (good vs bad) and more specific concepts (black lotus)
Too high: ~1
- Only a few features dominate
- Unstable
- Not much diversity
- Fix: decrease `l1Strength` slightly
	- Watch that density doesn't shoot up
- Fix: Dictionary resizing
	- high dead features: shrink dictionary by reducing `codeDimsMultiplier` 
	- dead rate low & density tiny: lower `l1Strength` first, then increase width slowly
Too low: ~0
- Every feature fires equally often
- Can mean features are blurred or un-specialized
- Fix: raise `l1Strength` slightly
	- Again, watch that density doesn't shoot up
Can also decreasing top-k sparsity instead of pure L1


Note: `l1Strength` is like a focusing lens
 - Too sharp: features are too specialized
	 - Start to loose big-picture concepts
 - Too blurry: no specialist features
	 - Start to loose finer concepts

**Decoder Cosine Summary**
TLDR: How similar decoder features are to each other

Mean: *overall redundancy level*
Good: $\leq$ 0.2-0.3
- Most features are fairly distinct
Too high: $\geq$ 0.3-0.5
- Features overlap
- Dictionary isn't diverse enough to cover the concepts covered

p95: *how bad the worst of the worst are*
Good: $\leq$ 0.5-0.6
- Feature directions are distinct enough
Too high:
- Clear feature duplication
- Top 95% of pairs are nigh on the same direction

max: *the absolute worst overlap*
- ~1.0 is duplicate columns
- Okay if p95 is okay, but can be a hint to prune/regularize

Need to find a better computer to run stuff on
- My laptop is too slow
- Google Colab shuts down after a couple hours
	- Could write a workaround, would rather not
	- DS takes ~30-45 mins to download and parse
		- Could use smaller chunks 
			- DS is provided as 2GB chunks
		- Save model after every chunk
		- Would let me have several states of the progression of the model
- Sent a message to (unnamed CPP professor) about getting access to the CPP HPC
- Want to restrict money spent on this as much as possible

# Working Session 6:
*Realized I never explained why I'm exploring this on my own*
- Part of the goal of this project was to learn enough to ask a question about the inner workings of LLMs, and then answer that question myself
- This technology *does exist*, it *does work*. I found several papers doing it
	- Even the MATS application doc has neuronpedia linked (which is *so* cool)
- I want to explore the idea on my own, and see if I can take it a little bit further as well
- It's also a great excuse to get access to the CPP HPC

*First task for this session: dataset chunking and parallel download/processing*
- Why not have one download worker and one processing worker?
- Process the info while chunks are downloaded
- Idea: run a training worker in parallel as well?

It also may be time to switch to a new model now that I have GPT2 working

##### Training notes
`seed`: 4738
`printTopN`: 5
`showFeatures`: 5

| `codeDimsMultiplier` | `l1Strength` | `epochs` | `batchSize` | `topkFeatures` | `learningRate` | `modelDims` | `codeDims` | `Reconstruction MSE` | `Reconstruction R^2` | `Activaiton density` | `Dead feature rate` | `Gini`              | `Cos mean`             | `Cos p95`           | `Cos max`           | `Mean Active` |
| -------------------- | ------------ | -------- | ----------- | -------------- | -------------- | ----------- | ---------- | -------------------- | -------------------- | -------------------- | ------------------- | ------------------- | ---------------------- | ------------------- | ------------------- | ------------- |
| 4                    | 0.01         | 100      | 256         | None           | 0.001          | 640         | 2560       | 1.216102066040039    | 0.8707082630691537   | 0.509960949420929    | 0.0                 | 0.13721787929534912 | 2.140082506230101e-05  | 0.07624021917581558 | 0.1416170299053192  | 1305.5        |
| 4                    | 0.01         | 100      | 256         | 50             | 0.001          | 640         | 2560       | 3.162237548828125    | 0.6638019154039747   | 0.01953125           | 0.5582031011581421  | 0.7219640016555786  | 0.0011569368652999401  | 0.09361152350902557 | 0.246487557888031   | 50.0          |
| 4                    | 0.01         | 100      | 256         | 100            | 0.001          | 640         | 2560       | 2.7461373901367185   | 0.708040235325258    | 0.0390625            | 0.2972656190395355  | 0.5582460761070251  | 6.872609083075076e-05  | 0.09423691779375076 | 0.23151257634162903 | 100.0         |
| 4                    | 0.01         | 100      | 256         | 150            | 0.001          | 640         | 2560       | 2.502765655517578    | 0.733914670676925    | 0.05859375           | 0.15781250596046448 | 0.46436819434165955 | -0.0012068153591826558 | 0.08843746036291122 | 0.20416176319122314 | 150.0         |
| 4                    | 0            | 100      | 256         | 150            | 0.001          | 640         | 2560       | 2.504317169189453    | 0.7337497191460376   | 0.05859375           | 0.15976563096046448 | 0.46610578894615173 | -0.0012865083990618587 | 0.09159372746944427 | 0.22213371098041534 | 150.0         |
| 4                    | 0.0001       | 100      | 256         | 150            | 0.001          | 640         | 2560       | 2.4856434631347657   | 0.7357350425478868   | 0.05859375           | 0.15625             | 0.46557626128196716 | 0.0006806570454500616  | 0.08926211297512054 | 0.19671639800071716 | 150.0         |
| 4                    | 0.0003       | 100      | 256         | 150            | 0.001          | 640         | 2560       | 2.5052859497070314   | 0.733646721774912    | 0.05859375           | 0.15742187201976776 | 0.4603228271007538  | -0.0016386461211368442 | 0.08727496862411499 | 0.20150408148765564 | 150.0         |
| 4                    | 0.001        | 100      | 256         | 150            | 0.001          | 640         | 2560       | 2.487263641357422    | 0.7355627908410651   | 0.05859375           | 0.15976563096046448 | 0.4597325325012207  | -0.000974904338363558  | 0.08555655181407928 | 0.2157110720872879  | 150.0         |

Seem to have found a good baseline, need to experiment with steering and cross-model compatability
- Can I use LLMs to figure out what the top features mean?
- I'd like to increase the model's dimensions and increase l1 score and topk features to try and get more, distinct features

# Working Session 7
*Would be beneficial to save the DS as tensor activations*
- Pros:
	- Don't need to re-process the tensors every time my notebook shuts down
- Cons:
	- More difficult to load a new model
- Solution:
	- Create a folder named `{modelName}_activations`
	- Autogenerated config file at the top
		- Model name
		- Model version number
		- Last updated timestamp
		- File names processed

# Working Session 8
*Realized that the ability to show the top strings in the DS comes from the fact that it's a text-based DS*
- Would rather keep it as text for now as it makes testing and steering easier

Instead, let's confirm that the SAE actually is learning features

On my streaming dataset:
- Works really nicely for loading data quickly, as not everything is loaded at once
- Sucks because it makes looking back at what the text was much more difficult

I've started using my school google drive as storage for the wikipedia dataset, as it quickly connects to the colab notebook (14gb is under the 15gb limit!)
- This gives me access to millions of training examples (~7m examples with ~1/3 of the DS uploaded) (ended up being 33,516,409 lines of text, split across 3,373 files)
- Unfortunately means I have to re-tune network
	- Started re-tuning and I'm getting way more promising results!
	- Before, text outputs before didn't seem to have a cohesive topic
	- Now they seem to differ more (although topics still seem to be random-ish)
	- More data seems to help refine the SAE, I'll keep uploading the DS to my drive and see how that changes the model's performance

Now that I have a large dataset and a computer that can run large training jobs (I signed up for colab education), I can start steering and probing the SAE to see how it reacts
- Hopefully I can find specific features that react well
- Need to research how to successfully and objectively steer the SAE
- Would like to train it with a different seed and compare cosine scores to see how similar the model is
	- Might show if the SAE learns to interpret features in the same way
- Also would like to do this with a different model and see how the SAE reacts
	- Might show if two different LLMs "think" the same

# Working Session 9
*It's time to start trying to see if my SAE "understands"*
- Start working on steering
	- How is this done?
	- Can I engineer prompts to activate certain neurons?
- Make a list of the activations and start labeling them

*The model seems to pick up really well on sports-based concepts and competitions*
- Is this because there's so much data about that or is there another reason?

I'm having some issues getting features to define specific concepts
Applying a layernorm to the activations seems to have help separate the feature activations
- Theory: There were so many sports examples that the model is overfitting, which is why all features seem to relate to a sport in one way or another

Some good-old-fashioned keyword analysis might be good to figure out what these words mean
- Tuning the network and increasing the size of the DS are both really good ways to decrease cosine similarity scores between feature outputs
- Also good at decreasing top output scores

# Working session 10
*I'd like to wrap everything up in this session or the next*

Some advice from ChatGPT:
- Last token is often PAD/EOS
	- Select last non-pading token (with attention_mask)
	- Could also sample random token
- Exemplar text alignment can be wrong if shuffle=True
	- Has this been messing with my examples???
		- IT HAS BEEN (this has been fixed)
	- Don't shuffle

***CHATGPT'S ADVICE ON SHUFFLING HAS REVEALED THE MODEL WORKS***
*like, really well*
Feature 28 is the best example of that:
- 2,4,5-Trichlorophenol (TCP) is an organochloride with the molecular formula C H Cl O. It has been used as a fungicide and herbicide. Precursor chemical used in the production of 2,4,5-Trichlorophenoxy 
- Acyl azides have also been synthesized from various carboxylic acids and sodium azide in presence of triphenylphosphine and trichloroacetonitrile catalysts in excellent yields at mild conditions. Anot 
- It is also possible to synthesize a target DNA strand for a DNA construct. Short strands of DNA known as oligonucleotides can be developed using column-based synthesis, in which bases are added one at 
- Sliding clamps are loaded onto their associated DNA template strands by specialized proteins known as " sliding clamp loaders ", which also disassemble the clamps after replication has completed. The 

The feature very clearly represents chemistry, potentially specifically biochem

The model does have some less clear features, like Feature 1336
- Looks like some taxonomic/morphological species decriptions
- There's some odd random stuff about Charles-Louis-Joseph-Xavier de la Vallée-Poussin (???) and a type of Frigate

Playing with the L1 strength or topk might give more clear features, but I want to move on for now

*It's cool seeing the model pick up on both human-intuitive concepts (organic chemistry) and "AI-intuitive" concepts (acronyms)*

Now that the model is *actually* interpretable, it's time to get probing
- Start with different seed
- Compare cosines of the same feature trained with a different seed
- Compare it to an older version of Gemma