# Do different AIs dream of the same electric sheep?
Do Large Language Models learn to represent high-level concepts in similar ways?

## Introduction
### Initial interest
The idea that LLMs use vectors to represent ideas was originally introduced to me by 3Blue1Brown in his video on how LLMs work. What really interested me was the (I'm sure now quite famous) example `Hitler+Italy-Germany=Mussolini`. My first thought was "how can we use AI to detect vectors with certain meanings?" After a quick Google search, I realized that this has already been done quite a bit, but what I couldn't seem to find much about was whether different LLMs represent concepts in the same way.

A good way to describe my idea is by comparing it to a video game. If two people can play Hollow Night: Silksong (ie, their training data for a task), and both beat the game (they accomplish the task), then they both likely have learned the same information (that is, unless one of them used cheats). Even though their brains are trained on different extraneous data from the rest of their lives, the two "neural networks" can complete the task with similar levels of success.

### Guiding question: Do different LLMs dream of the same electric sheep?

My initial thought process:
1. Some concepts are more subjective than others
	- Think math versus friendship
2. Do two models learn to represent math in the same way?
3. Do two models learn to represent friendship in the same way?
4. Do two models learn to represent good/evil in the same way?
	- If different, why do these models disagree?
5. If two models represent abstract concepts in the same way, can we mathematically define the "universal definition of [x] abstract concept?"
6. If they represent the same abstract concepts in different ways, why?
7. Do LLMs of different languages represent concepts in different ways?
	- How does a Chinese LLM represent friendship compared to an English LLM?
*(This kept going for a while, these are some of the key thinking points)*

### Goals
1. Train a simple sparse autoencoder on the last layer of an LLM
2. Find what strings maximally activate which parts of the LLM using the groupings from the SAE
3. Assign meaning to those features based on the content of those strings
4. Train a new SAE on the outputs of a different LLM
5. Assign meanings to that SAE's features
6. Compare and contrast what text activates SAE1 vs SAE2

### Outcome
By the end of this process, we should have all of the texts that maximally activate a set of features between two LLMs, and we can see what a model thinks about when it says something.

## Phase 1: What the heck am I supposed to do?
Initially, I didn't have any experience with mechanistic interpretability aside from taking apart an image recognition model I built a while ago, so all of this was new to me. 

*My notes for this whole process can be found in [[idea]], as I've documented my thought process through this project quite well*

I started by researching how SAEs are trained on LLM activations currently, playing around with [Gemma Scope](https://www.neuronpedia.org/gemma-scope), and asking ChatGPT a *lot* of questions. I eventually used MatPlotLib to make a graph of GPT-2's last hidden layer activations, and I got the graph below.
![[plottedGPT2Activations.png]]

What's interesting about this graph is that I made it by running 10 excerpts from [Project Gutenberg](https://ocw.mit.edu/ans7870/6/6.006/s08/lecturenotes/files/t8.shakespeare.txt), a compilation of the complete works of Shakespeare, and it looks very similar to the graph I get for any randomly sampled 10 excerpts. My theories as to why this graph was so predictable were:
1. These neurons are the ones that activate with tokens commonly at the end of an LLM's activations, like EOS, punctuation, and other boundary indicators commonly found at the end of a sentence
	- I disproved this by sampling different tokens; the graph looked the same.
2. These neurons are anisotropic, and just activate more than other neurons.
	- I disproved this by running other text through the network; it made different graphs.
3. Has the model picked up on the "feel" of the writing style, or maybe the Shakespeare-ness of a sentence?
	- Tested this first by writing some poetry myself, which produced a closer graph to the original
	- And then by having ChatGPT write some Shakespearean-style poetry, which gave a graph very similar to the one above!

This meant that, even in GPT-2, there were abstract concepts such as Shakespearean-ness being represented, and I was able to visualize that!

## Phase 2: Dataset
As with all language-based projects, the dataset is one of the most important factors. Garbage in, garbage out, as they say. The primary focus of my dataset search was to find one that represented as many concepts as possible, and I found that through Wikipedia.

The Wikimedia Foundation has kindly built a Kaggle dataset of the entirety of Wikipedia, a website known for its incredible information diversity. This helps ensure that all the data we get is as diverse as possible.

The dataset itself is 74.1GB and contains ~11 million lines of text. However, most of this information is metadata, and if piped through an LLM, would give a list of activations primed more toward metadata and how that metadata is formatted.

My data cleaning procedure involved extracting the paragraphs and summaries and shortening them to ~250 characters long. The shortening was required, as I do not have the computing power required to train on everything at once, even with incremental file loading. Shortening the text allowed for faster tinkering and iteration, and let me run my experiments with reckless abandon.

## Phase 3: Training and tuning
This was by far the most frustrating section for me for three reasons:
1. Sports/Competitions
	- Sports are an incredibly important thing around the world, and as such, there's a large proportion of the dataset that is sports-related. 
	- Not only sports, but competitions of all sorts. Elections, hackathons, etc, are all much more represented in the dataset than other topics
	- Due to this, I didn't realize something was off when I started getting seemingly random exemplar texts after the SAE was trained
2. Dates
	- Dates are also extremely important to people (both the fruit and the time representation). 
	- Seeing groupings of dates as exemplar features also made sense, as it represents a large portion of the dataset
3. Acronyms
	- A good chunk of large organizations, projects, and financial plans are referred to by acronyms, so for the same reason as the two previous examples, I was seeing groupings of Acronyms

However, no matter what the training statistics said, I would still always have some amount of noise in the exemplar features, which I assumed was normal, so I moved on to steering.

I eventually realized that I was using random selection from the dataset for training, and sequential selection to grab exemplar features. Realizing this caused an audible gasp, several excited keystrokes, and much more coherent exemplar features (a great triumph and relief to me at the time).

Some interesting exemplar topics that kept seeming to pop up were:
- Chemistry
- Taxonomy
- Sports
- Dates
- Acronyms

*Note: I find it very interesting that even when the exemplar features were grabbed correctly, the "evil three" features kept coming up, which reinforces my idea that they're a much larger part of the dataset.*
- I'm curious to go through the ~33 million lines of parsed data and do a linguistics analysis on what keywords are most common, and what percentage of the dataset these three are a part of

For hyperparameters, I generally tended to keep the following setup:
```python
codeDimsMultiplier = 4      # A multiplier for the "number of concepts" a model 
l1Strength         = 1e-3   # The strength of the l1 "lens"
epochs             = 100    # Iterations to train for
batchSize          = 512    # Samples to process at the same time
topkFeatures       = 64    # Number of top features to keep
learningRate       = 1e-3   # Learning rate to iterate at
seed               = 4738   # Random number seed, go Patribots!
```

My main tuning knobs were `l1Strength` and `topKFeatures`, and I found the best results came from using a low `l1Strength` and 32-64 `topkFeatures`. Having no `topK` seemed to spread out feature activations too much, and resulted in "blurry" features. `topKFeatures`  and `l1Strength` were the coarse and fine-tuning knobs I used to generally group features together and to refine the categories the SAE found.
## Phase 4: Comparison
I took the top 10 exemplar features for two LLMs, gemma-3-270m-it and facebook/opt-125m, with 5 text examples each, and ran them through ChatGPT to look for similarities. 
- I used ChatGPT for this because language models (especially newer ones) tend to be quite good at finding linguistic similarities in text.

I've pasted its results below, as I can't word it any better:

---
### Comparison of facebook.txt and gemma.txt Feature Groupings
**General Overlap in Categories**
Both models cluster features into high-level conceptual categories such as:
- Sports events
- Geography
- Historical figures
- Biology / taxonomy
- Laws / politics

This suggests they are segmenting text space into broadly similar domains.

**Differences in Grouping Granularity**
- **facebook**: Features often feel narrower and more fine-grained  
  (e.g., specific tournaments, species, or named people).  
- **gemma**: Features tend to group more broadly, with descriptive or contextual detail  
  (e.g., legal systems, historical movements, artistic practices).

**Stylistic Emphasis**
- **facebook**: Reads like raw encyclopedia snippets — factual, entity-centric.  
- **gemma**: More thematic and explanatory — often adding narrative context.

**Shared Conceptual Anchors**
Both contain clusters around:
- **Sports competitions** (football, basketball, athletics)  
- **Geography and settlements** (villages, towns, census records)  
- **Biological species** (plants, moths, fish, etc.)  
- **History / politics / governance** (wars, elections, constitutions, legal systems)

**Divergences**
- **facebook**: Isolates specific entities (e.g., one athlete, one moth, one event).  
- **gemma**: Ties entities into larger thematic patterns (e.g., classification systems, political frameworks).

**Summary**
- **facebook** = more *fine-grained*, entity-level clusters.  
- **gemma** = more *thematic*, relational, or contextual clusters.  

Both group concepts in similar directions, but at different levels of abstraction.

---

### Thoughts:
I would consider this a partial success. It seems as though Facebook is grouping concepts at a much higher level than Gemma is, and I have two main theories as to why:
1. Gemma is a larger model. With a parameter space of ~270m, it may be able to represent higher-level concepts than the ~125m parameter space of Facebook.
2. I didn't train Facebook as well, as I ran out of time toward the end of the project and was unable to tinker with the model's SAE as much as I did with Gemma

### Potential knobs to fiddle with:
The big three variables that I would like to fiddle with next are:
1. Dataset
	I have so much more data that I could be using, but I simply didn't have the computing power necessary to use all of it. Unfortunately, I didn't get access to CPP's HPC for this project, but I'm going to keep asking and using my monthly Google Colab credits to further tune and improve this project.
2. Speed
	There are several parallelization opportunities I have with file loading and activation gathering, and I'd like to implement all of them so I utilize my compute resources more effectively.
3. Comparison techniques
	Admittedly, my comparison techniques are lackluster, as all they involve is a light skim of the top 10 exemplar features. In the future, I'd like to get all the features and make a comparison table with their similarities.
	- This would be a great use of another LLM. By tokenizing and passing the exemplar features through another LLM, we'd get a more quantifiable measure of the concepts these LLMs are learning to represent
	- LLM-ception

There are definitely others, and next steps involve loading larger models like Llama-8b and applying the same concepts to it, but for now, refining the methods and trying to get a more definitive answer are my main goals for the future.
## So, did I accomplish my goals?
I've *definitely* learned something new. This project has been an awesome way to break into the mech interp space and get my hands *properly* dirty. While I haven't had a chance to explain mech interp to my mom yet, I've been talking to friends and coworkers about this project, and it's started several amazing conversations about philosophy, consciousness, and morality; all great things to come out of a research project on mech interp.

This project was the first time in a while I've had something to work on that's entirely new and entirely a challenge to do. Stepping so far outside my comfort zone of automation and robotics has not only expanded my knowledge about mech interp, but has expanded the way I think about my career and my goals. This project has held my attention the entire way through by giving me infinite problems to work on and just enough tests passed to keep progress moving. It was naive of me to assume my question would be answered in 20 hours, as the best answer I've gotten is "There's a chance! You should look into this more!" 

Unfortunately, this one will have to wait until I submit this project, so for now, I'll once again reference how exciting and inspiring this project has been for me. I explained what I was doing to my Mom, and she said, "So it's psychology for AI", and it helped me realize how exciting and important a career path this is.