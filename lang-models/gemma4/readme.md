# Build a coding assistant using Gemma 4 via Ollama

Build a coding assistant using [Gemma 4 via Ollama](https://ollama.com/library/gemma4), with a Gradio interface.

## Introduction

The app features a split-pane layout with a live code editor on the left and an agentic chat panel on the right. You can upload images or code files as context, enable tool use so the model can run and validate code, and toggle a thinking mode for harder problems.

By the end of this tutorial, you'll have a working local app that can:

- Write, explain, and debug code in 15+ languages
- Execute Python code in a sandboxed sub-process and return results
- Accept images and text files as multimodal context
- Stream responses in real time from a locally running Gemma 4 model
- Run as an agentic loop, calling tools and following up based on results

## What Is Gemma 4?

[Gemma 4](https://blog.google/innovation-and-ai/technology/developers-tools/gemma-4/) is Google DeepMind's open-weights model family, designed for both local deployment and research. It builds on the Gemma lineage with improved instruction following, longer context windows, and native multimodal input handling and is built from the same research infrastructure as Gemini 3.

Models like Gemma-4-26B(MOE) and Gemma-4-31B achieve Elo scores comparable to much larger models, indicating strong performance-per-parameter. The 31B model currently ranks as 3rd open model in the world on the Arena AI text leaderboard, and the 26B model secures the 6th spot. [](https://blog.google/innovation-and-ai/technology/developers-tools/gemma-4/) This makes Gemma 4 particularly well-suited for local and resource-constrained deployments without sacrificing capability.

### The Gemma 4 model family

Gemma 4 is released in four versatile sizes including Effective 2B (E2B), Effective 4B (E4B), 26B Mixture of Experts (MoE), and 31B Dense models. [](https://blog.google/innovation-and-ai/technology/developers-tools/gemma-4/) The family splits into two distinct tiers based on deployment target:

| **Model**  | **Architecture** | **Total Params** | **Active/Effective Params** | **Context Length** | **Modalities** |
|--|--|--|--|--|--|--|--|--|--|--|--|--|--|
| **Gemma-4-31B** | Dense  | Transformer | 31B | 31B | 256K tokens | Text, Vision, Video |
| **Gemma-4-26B-A4B** | MoE (128 Experts) | 26B | 3.8B active | 256K tokens | Text, Vision, Video |
| **Gemma-4-E4B** | Dense Transformer | 7.9B (with embeddings) | 4.5B effective | 128K tokens | Text, Audio, Vision, Video |
| **Gemma-4-E2B** | Dense Transformer | 5.1B (with embeddings) | 2.3B effective | 128K tokens | Text, Audio, Vision, Video |

#### Gemma 4 model Family

Let’s dive deeper into each variant:

![Visual Guide to Gemma 4](https://media.datacamp.com/cms/fe4f8a20f72a289f20706baa8e5f8c1f.png)

- 31B Dense is the flagship model optimized for data center deployment and complex reasoning workloads. It supports a 256K token context window with a 1024-token sliding window for efficient long-context processing.
- 26B-A4B (MoE) is Gemma's first MoE model routing tokens through 128 experts while keeping only 3.8B parameters active per forward pass. This gives it near-31B quality at a fraction of the compute cost per token making it well suited for high-throughput serving.
- E4B and E2B are the on-device and mobile tier. Unlike the larger variants, they include native audio input for speech recognition alongside vision and video, making them the most multimodally capable models in the family for edge deployment.

All four models are available under Apache 2.0 license and can be deployed locally via Ollama, vLLM, llama.cpp, or Unsloth.

For coding tasks, Gemma 4 excels at:

- Writing complete, structured code with explanations
- Reasoning over existing codebases when provided as context
- Tool use and agentic workflows when paired with an orchestration layer
- Handling images alongside code (e.g., reading a UI screenshot and generating matching HTML)

In this tutorial, we use the `gemma4:e4b` (9.6GB) variant via Ollama which is a quantized version well-suited for local inference on consumer hardware.

### Running Gemma 4 via Ollama

Ollama handles model downloading, quantization, serving and provides an OpenAI-compatible HTTP API. For this tutorial, Ollama acts as the inference backend, and our app communicates with it over `host:11434`.
