Project: Prompt-Specific-Adversarial-Image-Generation-via-Reinforcement-Learning



This project implements a reinforcement-learning-based framework for generating prompt-specific adversarial images. The system trains a Stable Diffusion 1.5 model (with LoRA) using a DDPO-style reinforcement learning loop. The goal is to produce images that influence a multimodal model’s response to a given text prompt.



A personal HuggingFace access token is required to download the necessary models.



Environment Requirements:

\- Python 3.12 (Ubuntu 22.04)

\- PyTorch 2.8.0

\- CUDA 12.8

\- GPU with large memory (recommended 48GB+)



Core Python Dependencies:

\- torch

\- torch.nn

\- torch.optim

\- diffusers (UNet2DConditionModel, AutoencoderKL, DDIMScheduler)

\- transformers (CLIPTextModel, CLIPTokenizer, AutoModelForCausalLM, AutoTokenizer)

\- transformers (Qwen2\_5\_VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig)

\- qwen\_vl\_utils (process\_vision\_info)

\- peft (LoraConfig, get\_peft\_model)

\- csv

\- random

\- re

\- matplotlib.pyplot

\- copy

\- PIL

\- json



Main Components:

\- Stable Diffusion 1.5 UNet with LoRA adapters for trainable image generation.

\- CLIP tokenizer and text encoder for prompt conditioning.

\- Qwen2.5-VL-7B-Instruct as the Blue Team model for generating responses conditioned on image + text.

\- Llama-3-8B-Instruct as the Judge model for scoring the safety or compliance of generated responses.

\- Reinforcement learning loop using DDIM sampling, reward computation, and DDPO-style updates.



Typical Workflow:

1\. Encode the text prompt using CLIP.

2\. Generate a diffusion trajectory using DDIM.

3\. Decode the final latent into an image.

4\. Query the Blue Team model with the image and prompt.

5\. Score the response using the Judge model.

6\. Compute reward and update the LoRA parameters of the UNet.

7\. Repeat for multiple training steps.



This repository contains the necessary code to run the full RL pipeline, including model loading, LoRA injection, sampling, reward computation, and optimization.

