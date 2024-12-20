# LLM4all

Nowaydays you cannot live without Large Language Models(LLM).
Innovations about this are popping up on a weekly or even daily basis. As it is hard to follow everything, I made this list covering many links, from tutorial, resources, code for fine-tuning, RAG, compressions, and other things 🤖🤖🤖. I might keep updating it if novel tools will pop-up and some get obsolete. Enyjoy.

![](https://github.com/alecrimi/LLM4all/blob/main/pexels-googledeepmind-18069697.jpg) 

## Applications

**Build & Auto-optimize**

- [dspy](https://github.com/stanfordnlp/dspy) - DSPy is the framework for programming—not prompting—foundation models.

**Orchestration**

- [LangChain](https://github.com/hwchase17/langchain) — LangChain is a popular Python/JavaScript library for chaining sequences of language model prompts. Mostly focused on text from document extraction.
- [Haystack](https://github.com/deepset-ai/haystack) - Haystack is Python framework that allows you to build applications powered by LLMs.
  Mostly focused on vector data representation. 
- [LlamaIndex](https://github.com/jerryjliu/llama_index) — LlamaIndex is a Python library for augmenting LLM apps with data.

- [Cohere](https://cohere.com/) - Cohere is platform offering LLMs for natural language understanding and generation, suitable for building chatbots and text analysis tools.

**Prompt Optimization**

- [AutoPrompt](https://github.com/Eladlev/AutoPrompt) - AutoPrompt is a framework for prompt tuning using Intent-based Prompt Calibration
- [PromptFify](https://github.com/promptslab/Promptify) - PromptFify is a library for prompt engineering that simplifies NLP tasks (e.g., NER, classification) using LLMs like GPT.

## Pretraining

- [PyTorch](https://pytorch.org/) - PyTorch is an open source machine learning library based on the Torch library, used for applications such as computer vision and natural language processing.
- [TensorFlow](https://www.tensorflow.org/) - TensorFlow is an open source machine learning library developed by Google.
- [JAX](https://github.com/jax-ml/jax) - JAX is Google’s library for high-performance computing and automatic differentiation.
- [tinygrad](https://github.com/tinygrad/tinygrad) - tinygrad is a minimalistic deep learning library with a focus on simplicity and educational use, created by George Hotz.
- [micrograd](https://github.com/karpathy/micrograd) - micrograd is a simple, lightweight autograd engine for educational purposes, created by Andrej Karpathy.

## Vector databases for RAG
- [ChromaDB](https://github.com/chroma-core/chroma) 
- [Weaviate](https://github.com/weaviate/weaviate)
- [Qdrant](https://github.com/qdrant)
- [Milvus](https://github.com/milvus-io/milvus)
- [PgVector(connector to Postgress)](https://github.com/pgvector/pgvector)


## Fine-tuning

- [Transformers](https://huggingface.co/docs/transformers/en/installation) - Hugging Face Transformers is a popular library for Natural Language Processing (NLP) tasks, including fine-tuning large language models.
- [Unsloth](https://github.com/unslothai/unsloth) - Finetune Llama 3.2, Mistral, Phi-3.5 & Gemma 2-5x faster with 80% less memory!
- [LitGPT](https://github.com/Lightning-AI/litgpt) - 20+ high-performance LLMs with recipes to pretrain, finetune, and deploy at scale.

## Serving

- [TorchServe](https://pytorch.org/serve/) - TorchServe is an open-source model serving library developed by AWS and Facebook specifically for PyTorch models, enabling scalable deployment, model versioning, and A/B testing.

- [TensorFlow Serving](https://www.tensorflow.org/tfx/guide/serving) - TensorFlow Serving is a flexible, high-performance serving system for machine learning models, designed for production environments, and optimized for TensorFlow models but also supports other formats.

- [Ray Serve](https://docs.ray.io/en/latest/serve/index.html) - Ray Serve is part of the Ray ecosystem, Ray Serve is a scalable model-serving library that supports deployment of machine learning models across multiple frameworks, with built-in support for Python-based APIs and model pipelines.

- [NVIDIA TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) - TensorRT-LLM is NVIDIA's compiler for transformer-based models (LLMs), providing state-of-the-art optimizations on NVIDIA GPUs.
 
- [NVIDIA Triton Inference Server](https://developer.nvidia.com/triton-inference-server) - NVIDIA Triton Inference Server is a high-performance inference server supporting multiple ML/DL frameworks (TensorFlow, PyTorch, ONNX, TensorRT etc.), optimized for NVIDIA GPU deployments, and ideal for both cloud and on-premises serving.

- [ollama](https://github.com/ollama/ollama) - ollama is a lightweight, extensible framework for building and running large language models on the local machine.

- [llama.cpp](https://github.com/ggerganov/llama.cpp) - llama.cpp is a library for running LLMs in pure C/C++. Supported architectures include (LLaMA, Falcon, Mistral, MoEs, phi and more).

- [TGI](https://github.com/huggingface/text-generation-inference) - HuggingFace's text-generation-inference toolkit for deploying and serving LLMs, built on top of Rust, Python and gRPC.

- [vllm](https://github.com/vllm-project/vllm) - An optimized, high-throughput serving engine for large language models, designed to efficiently handle massive-scale inference with reduced latency.

- [sglang](https://github.com/sgl-project/sglang) - SGLang is a fast serving framework for large language models and vision language models.

- [LitServe](https://github.com/Lightning-AI/LitServe) - LitServe is a lightning-fast serving engine for any AI model of any size. Flexible. Easy. Enterprise-scale.

- [ModelDB](https://modeldb.science) - ModelDB provides an accessible location for storing and efficiently retrieving computational neuroscience models.

- [BentoML](https://www.bentoml.com) - BentoML is a flexible framework for serving machine learning models with built-in support for LLMs and easy integration with cloud platforms.

- [Seldon Core](https://www.seldon.io/) - Seldon Core is an open-source platform for deploying machine learning models on Kubernetes, supporting various frameworks including TensorFlow and PyTorch.

## Monitoring and Evaluation Tools

- [LangKit](https://whylabs.ai/langkit) -  LangKit is an advanced tool for monitoring AI models in production, providing telemetry data extraction to detect issues like toxicity and hallucinations.

- [Arize Phoenix](https://phoenix.arize.com) - Arize Phoenix offers observability into LLM applications, enabling debugging, experimentation, and evaluation throughout the development lifecycle.

- [Neptune.ai](https://neptune.ai) - Neptune.ai is a metadata store for MLOps that helps in tracking experiments and monitoring model performance in production.

- [Weights & Biases](https://wandb.ai/site) - Weight & Biases is a tool for tracking experiments, visualizing metrics, and collaborating on machine learning projects, including those involving LLMs.

## Prompt Management

- [Opik](https://github.com/comet-ml/opik) - Opik is an open-source platform for evaluating, testing and monitoring LLM applications.

## Datasets

Use Cases

- [Datasets](https://huggingface.co/docs/datasets/en/index) - A vast collection of ready-to-use datasets for machine learning tasks, including NLP, computer vision, and audio, with tools for easy access, filtering, and preprocessing.
- [Argilla](https://github.com/argilla-io/argilla) - A UI tool for curating and reviewing datasets for LLM evaluation or training.
- [distilabel](https://distilabel.argilla.io/latest) - A library for generating synthetic datasets with LLM APIs or models.
- [The Pile](https://pile.eleuther.ai) - The Pile is a 825 GiB diverse, open source language modelling data set that consists of 22 smaller, high-quality datasets combined together.
- [Common Crawl](https://commoncrawl.org) - Common Crawl is a regularly updated web archive that provides a vast amount of text data useful for training LLMs.

- [OpenWebText2](https://openwebtext2.readthedocs.io/en/latest) - OpenWebText2 is an enhanced version of the original OpenWebTextCorpus covering all Reddit submissions from 2005 up until April 2020.

Fine-tuning

- [LLMDataHub](https://github.com/Zjh-819/LLMDataHub) - LLMDataHub is a quick guide (especially) for trending instruction finetuning datasets.
- [LLM Datasets](https://github.com/mlabonne/llm-datasets) - LLM Datasets is a high-quality datasets, tools, and concepts for LLM fine-tuning.

Pretraining

- [IBM LLMs Granite 3.0](https://www.linkedin.com/feed/update/urn:li:activity:7259535100927725569?updateEntityUrn=urn%3Ali%3Afs_updateV2%3A%28urn%3Ali%3Aactivity%3A7259535100927725569%2CFEED_DETAIL%2CEMPTY%2CDEFAULT%2Cfalse%29) - Full list of datasets used to train IBM LLMs Granite 3.0

## Benchmarks

- [lighteval](https://github.com/huggingface/lighteval) - lighteval is a library for evaluating local LLMs on major benchmarks and custom tasks.

- [evals](https://github.com/openai/evals) - OpenAI's open sourced evaluation framework for LLMs and systems built with LLMs.
- [ragas](https://github.com/explodinggradients/ragas) - ragas is a library for evaluating and optimizing LLM applications, offering a rich set of eval metrics.

## Agents

- [OpenAI Swarms](https://www.crewai.com/open-source) - OpenAI educational project about deploying agents.
- [CrewAI](https://www.crewai.com/open-source) -  Build and deploy automated workflows using any LLM and cloud platform. 
- [Agency Swarm](https://github.com/VRSEN/agency-swarm) - Alternative implementation of swarming AI agents.
- [OpenHands](https://github.com/All-Hands-AI/OpenHands) - Open source agents for developers by [AllHands](https://www.all-hands.dev/).
- [CAMEL](https://github.com/camel-ai/camel) - First LLM multi-agent framework and an open-source community dedicated to finding the scaling law of agents. by [CAMEL-AI](https://www.camel-ai.org/).
- [AutoGen](https://github.com/microsoft/autogen) - A programming framework for agentic AI by Microsoft.

## Neuroscience papers

- [HippoRag](https://arxiv.org/abs/2405.14831) - Neurobiologically Inspired Long-Term Memory for Large Language Models
Bernal Jiménez Gutiérrez, Yiheng Shu, Yu Gu, Michihiro Yasunaga, Yu Su
- [TopoLM](https://arxiv.org/abs/2410.11516) -  brain-like spatio-functional organization in a topographic language model
Neil Rathi, Johannes Mehrer, Badr AlKhamissi, Taha Binhuraib, Nicholas M. Blauch, Martin Schrimpf
- [LLM and language neuroscience](https://arxiv.org/abs/2410.00812) - A generative framework to bridge data-driven models and scientific theories in language neuroscience Richard Antonello, Chandan Singh, Shailee Jain, Aliyah Hsu, Jianfeng Gao, Bin Yu, Alexander Huth
- [Language in Brains, Minds, and Machines](https://www.annualreviews.org/content/journals/10.1146/annurev-neuro-120623-101142) 
Greta Tuckute, Nancy Kanwisher, and Evelina Fedorenko

-----------------
If you have worthy additions, ping me or connect 
- [Twitter](https://x.com/Dr_Alex_Crimi)
- [LinkedIn](https://www.linkedin.com/in/alecrimi/)
- [Mastodon](https://mstdn.social/@AlexCrimi)
- [Threads](https://www.threads.net/@dr.alecrimi)
