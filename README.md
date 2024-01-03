# cm-hate-speech-detection

Detecting and flagging hate speech and abuse on social media platforms is an
important and time sensitive task. While supervised learning approaches have
been successful in identifying hate speech in English and some other high-resource
languages, this is not the case for code-mixed text, which is a common way
of communication for many bilingual people. In this project, we evaluate the
effectiveness of Large Language Models for the task of Hindi-English code mixed
hate speech detection, and compare this to existing BERT-based models on an
existing "Hinglish" Indian Politics hate speech dataset. Additionally, we evaluate
the generalization capabilities of these models on a custom Hindi-English code-
mixed hate speech dataset. We find that smaller specialized finetuned models such
as Hing-RoBERTa outperform both prompted and finetuned LLaMa-2 on the
existing Hinglish Indian Politics dataset, and also generalize better to our newly
collected dataset.

We trained our models on the politics domain dataset and evaluated their cross-domain performance on our custom dataset. We evaluated the following approaches:

1. **Fine-tuning BERT-based models:** This involved fine-tuning multilingual and code-mixed BERT-based models like HingRoBERTa and XLM-RoBERTa. The code is present in the `bert-models` folder.

2. **LLM Prompting:** We explored prompting methods utilizing the LLaMa 70B Chat model in a zero-shot setting, employing in-context learning and chain of thought reasoning. The code is present in the `prompting` folder.

3. **LLM Fine-tuning:** We fine-tuned the LLaMa 7B and BLOOMZ multilingual LLM models in classification and causal language modeling scenarios. This involved generating the entire sequence and generating only the output label (completion-only). The code is present in the `linear_probing` and `causal_language_modeling` folders.

You can find more details and the results in the report entitled `Report.pdf` in this repository.