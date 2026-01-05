# PRODIGY_GA_01

Task Title
Fine-Tuning GPT-2 for Text Generation using Hugging Face and Google Colab

Project Overview
This project demonstates how to fine tune a pre trained GPT-2 Language model, which was developed by OpenAI to generate coherent and contextually relevant text based on a give prompt. This model is fine-tuned on a small custom dataset using the Hugging Face Transformers, Datasets modules and executed on a Tesla T4 GPU on Google Colab.

NLP- Natural Language Processing 
It is a branch of Artificial Intelligence that enables computers to understand, interpret and generate human language. 
In this project, NLP is applied to:
-Learn patterns from textual data
-Generate human-like text responses based on prompts.

Dataset
A custom dataset is stored in dataset.txt. It contains sentences related to:
-Artificial Intelligence
-Machine Learning
-Deep Learning
-Natural Language Processing
-Generative Models
It is loaded using the Hugging Face datasets library

Technologies used:
-Python
-Google Colab(Tesla T4 GPU)
-Hugging Face Transformers
-Hugging Face Datasets
-GPT-2

Implementation Steps:
1. Environment Setup:
   Verified GPU availability using nvidia-smi
   Installed required libraries
2. Dataset Creation and Loading
   Text data saved in dataset.txt
   Dataset loaded using load_dataset("text")
3. Tokenization
   GPT-2 tokenizer used to convert text into toen IDs
   Padding token set to eos_token
   Labels created from input_ids for casual language modeling
4. Model Fine- Tuning
   Pre trained GPT-2 model loaded
   Fine tuned using Hugging Face Trainer
   FP16 enabled for efficient GPU training
5. Text Generation
   Text generated using sample based decoding
   Prompts used include:
   -Artifiial intelligence
   -Machine learning
   -Deep learning

Text Generation Strategy Used:
This project uses Sampling-Based Text Generation, specifically:
-Temperature Sampling – Controls randomness in token selection
-Top-K Sampling – Limits selection to the top K probable tokens
-Top-P (Nucleus) Sampling – Selects tokens based on cumulative probability
-Multiple Return Sequences – Generates diverse outputs per prompt

Output:
Fine tuned model and tokenizer saved in the directory "gpt2-finetuned/". 
The generated outputs are displayed directly in the notebook

How to Run
-Open the notebook in Google Colab
-Enable GPU (Runtime → Change runtime type → GPU)
-Run all cells sequentially
-Modify the dataset or prompts if needed
-Generate and observe text outputs

Key Learnings
-Understanding GPT-2 and transformer architecture
-Tokenization, padding, and attention masks in NLP
-Fine-tuning pre-trained language models
-Sampling-based decoding techniques
-GPU-accelerated training using Google Colab
-Managing NLP pipelines using Hugging Face

References:
Hugging Face Blog – How to Generate Text
https://huggingface.co/blog/how-to-generate
Analytics Vidhya – Exploring Text Generation with GPT-2
https://www.analyticsvidhya.com/blog/2023/09/exploring-text-generation-with-gpt-2/#h-code-implementation
Kaggle – Text Generation with Hugging Face GPT-2
https://www.kaggle.com/code/tuckerarrants/text-generation-with-huggingface-gpt2
TimesPro – GPT-2 Decoded: Exploring Advanced Text Generation Strategies
https://timespro.com/blog/gpt-2-decoded-exploring-advanced-text-generation-strategies
Medium – Generating Text with GPT-2 in Under 10 Lines of Code
https://medium.com/@majd.farah08/generating-text-with-gpt2-in-under-10-lines-of-code-5725a38ea685

Author
Brunda Addagalla
This repository is part of the PRODIGY Internship Program.
