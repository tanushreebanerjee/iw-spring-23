# running notes

## 01/23/2023

Next steps:

Experiment 1 - evaluate model with an image and randomly selected question, how often does it admit "I don't know" etc. vs attempting to provide an answer

- task: VQA
- model: ?
- dataset: ?

## 11/29/2022

### initial idea

Try methods to prevent hallucination of incorrect answers/unanswerable questions, for example by adding an explicit prompt (discrete and/or continuous)

- reducing hallucination in VQA may be more meaningful (and more easily quantifiable) than in image captioning

- For example, if a model is evaluated with an image and a randomly selected question (from the whole dataset), how often does it say “i don’t know” or “no answer” versus attempting to provide an answer (even if it’s a meaningless question on the image)?

- human study to assess what fraction of these random image/question pairs are actually meaningless and how often they happen to actually work well together?

### potential research questions

- How easy is to train a model that identifies when a question is meaningless for a given image?
- What would it take to effectively build a system that could consider this scenario?

### motivation

[Flamingo paper, DeepMind April 2022](https://arxiv.org/pdf/2204.14198.pdf): Appendix D.1, Figure 13

- Flamingo still suffers from “occasional hallucinations and ungrounded guesses"

### related Work

[Survey on hallucination in NLG](https://arxiv.org/pdf/2202.03629.pdf): Section 12

Image Captioning:

- Work done so far is mostly on image captioning
- Rohrbach et al: metric for measuring hallucination in image captioning
- approach 1: augment data to mitigate hallucination (based on the hypothesis that hallucination is caused by “systematic co-occurence of particular object categories in input images”.
- approach 2: using beam search while decoding to reduce hallucination by reducing uncertainty over predictions
- approach 3: using “object masked language modeling” as the pre-training objective to reduce hallucinations.

State of the art: [this](https://arxiv.org/pdf/2210.07688.pdf) - Oct 2022 (on image captioning)

- reduce object hallucination in image captioning by 17.4%

Thoughts:

- There isnt any work I could find on analysing hallucinations in any other VL tasks (so no metrics for measuring hallucinations or other mitigation methods - at least none that I could find
- I could not find any work on using prompting to mitigate the hallucination problem though
- Currently there seem to be no metrics for analysing hallucinations on anything other than image captioning (and the metric still requires a specified list of object categories, so may not generalise to other datasets)

Other cool things to note:

- Interestingly, there has also been some work on leveraging hallucinations to perform some tasks better (e.g. [this](https://aclanthology.org/2022.acl-long.373/)) - which I thought was cool - so hallucinations may not always be a bad thing?
