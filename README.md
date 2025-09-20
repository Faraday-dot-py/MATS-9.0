# MATS-9.0
# Executive Summary
#### Do different AIs dream of the same electric sheep?
Exploring Whether Large Language Models Learn to Represent Concepts in Similar Ways
### Overview
This project dives into whether different large language models (LLMs) represent abstract concepts in comparable ways. Inspired by the idea that vector arithmetic in embeddings (e.g., $Hitler+Italy-Germany\approx Mussolini$), the central guiding question was:

***Do different LLMs "dream of the same electric sheep"*** - that is, do they learn to represent high-level concepts in similar ways?

By training sparse autoencoders (SAEs) on activations from two different LLMs and comparing the features they extract, the project aimed to reveal whether models converge on shared representations of ideas or diverge due to architecture, training data, or scale.

### Motivation and Approach
An analogy for this idea comes from human behavior: If two players accomplish a task independently, they must have internalized similar knowledge of how that task is done, even if they were not explicitly told how to do so, nor did their backgrounds overlap. Similarly, two LLMs of different architectures trained on different data seem to encode common abstract structures if they solve similar linguistic tasks.

The research approach followed four stages:
1. **Training SAEs on LLM outputs** - Capture interpretable features from later hidden states.
2. **Exemplar text identification** - Select input strings that maximally activate those features.
3. **Feature interpretation** - Assign semantic meaning to groups of activations
4. **Cross-model comparison** - Contrast features between two different LLMs

### Key Findings:
1. **Abstract Concepts Emerge Even in Small Models**
	Early experiments with GPT-2 showed feature activations corresponding to subjective writing style-based properties like "Shakespearean-ness." Even early, rudimentary models seem to capture surprisingly abstract features.
2. **Dataset Bias Shapes Feature Prominence**
	Using the Wikipedia dataset as a training base highlighted some recurring themes such as sports, dates, and acronyms. These dominated exemplar features, showing the importance of dataset distribution and normalization.
		*Note: Noise in exemplar results was reduced once random vs. sequential sampling inconsistencies were corrected *
3. **Models Differ in Level of Abstraction**
	Comparing Gemma and Facebook OPT revealed overlapping high-level conceptual domains such as sports, geography, biology, and politics, but at varying levels of granularity.
	- **OPT (125m)**: Narrow, entity-level clusters (one tournament, one species)
	- **Gemma (270m):** Broader, theme-based groupings (e.g., political systems, historical movements).
	This suggests larger parameter counts may support more abstract representations, while smaller models lean toward factual, entity-centric clusters.
4. **Partial Convergence Across Models**
	Despite differences, both models converged on intuitive groupings of human knowledge. This supports the hypothesis that LLMs trained on diverse text bases tend to partition "concept space" into somewhat similar regions, even if boundaries differ.

### Challenges Encountered
- Computational limits prevented training on the full Wikipedia dataset, requiring the shortening of inputs and harsh sampling
- The dataset is skewed toward sports, dates, and acronyms, which distorts feature balance
- Comparison methods were limited; reliance on manual review and secondary LLM assistance left significant room for more quantitative measurements of similarity

### Future Directions
1. **Expand datasets** - Utilize high-performance computing to incorporate more balanced and larger-scale data
2. **Improve feature comparison** - Use LLM-assisted feature labeling across larger feature sets
3. **Scale up models** - Apply the methodology to larger LLMs (e.g., LLaMA-8B) to test whether observed behavior extrapolates up with scale
4. **Cross-language analysis** - Compare models trained in different languages to analyze cultural variation in conceptual representation

The practical goal was to compare how smaller-scale models (Gemma-3-270m-it vs Facebook OPT-125m) represent concepts and abstract ideas found in Wikipedia articles.
### Goals for this project:
- [x] Learn something new.
	- I've found it quite difficult to make progress on my knowledge of mechanistic interpretability, as I tend to be more of a hands-on learner and haven't gotten my hands properly dirty yet. This project is a great excuse to do that!
	- Deliverable: Learn a fundamental concept of mech interp well enough to explain it to my mom, and for her to understand it.
- [x] Come up with an idea I'm excited about
	- Q: Why work on a project if I'm not excited about it? A: Excitement is that *special sauce* that turns "welp, back to work" into realizing the solution to a problem while you're doing something completely different.
	- Deliverable: Either answer my original question (and come up with follow-up questions), or get to a point where I can't move further without help.
- [ ] Get accepted to MATS 9.0 (Not accepted)
	- MATS provides an incredible opportunity to skip the bureaucratic nonsense involved with interning at some large companies, and instead connects you directly with proper AI researchers. This experience is invaluable and would drastically improve my research and communication skills, something I've been working on for a while.
	- Deliverable: Secure a 12-week research internship with Neel Nanda!
