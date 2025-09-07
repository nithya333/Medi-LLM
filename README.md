# Multi-Agent QLoRA-Optimized LLMs for Personalized Healthcare
Team: M. Nithyashree, Pavithra N, Krupa P. Nadgir

Summary: A lightweight, memory-efficient framework for deploying multi-task Large Language Models (LLMs) on edge devices in healthcare settings. Integrates real-time IoT sensing with quantized LoRA adapters to deliver diagnostic reasoning, health education, and one-day diet planning—all on resource-constrained hardware.

## Features
- Modular adapter-based fine-tuning
- 4-bit quantization with QLoRA for minimal memory usage
- Three specialized agents:
- - Diagnostic reasoning
- - Personalized health tips generation
- - One-day diet planning
- Real-time IoT integration (heart rate, SpO₂, ECG, temperature)
- Edge deployment of memory-efficient LLM models

## Architecture
- Base Model: DeepSeek-R1-Distill-LLaMA-8B

Hosted on Hugging Face, distilled and quantized to 4-bit

- Adapters:

LoRA for low-rank parameter updates. QLoRA adds adaptive scaling for quantized weights. Stored as .safetensors, dynamically loaded per task

- IoT Pipeline: Hardware sensors stream vitals to the edge device

- Deployment: Single quantized model in memory, Task switching by loading specific adapter

## Components

| Component                  | Description                                                                                |
|----------------------------|--------------------------------------------------------------------------------------------|
| Diagnostic Agent           | Interprets symptom descriptions and suggests possible medical conditions                   |
| Health Education Generator | Produces tailored preventive care and wellness guidance                                    |
| One-Day Diet Planner       | Generates culturally relevant, individualized meal plans based on real-time health metrics |
| Web Dashboard & API        | User interface for text/voice input, visualization of vitals, and real-time recommendations |

## Prerequisites
Python 3.10+

PyTorch 2.x

bitsandbytes

transformers

peft (Parameter-Efficient Fine-Tuning library)

NVIDIA Jetson Orin Nano (8–16 GB RAM)

<!-- ## Usage

git clone https://github.com/nithya333/Medi-LLM.git
cd multi-agent-qlora-healthcare

python run_inference.py --device jetson
Access the web dashboard at http://<edge_ip>:3000

Select a task (diagnosis, education, diet) and input patient data -->

## Results
* Fine-tuned models achieve coherent, context-aware outputs vs. generic base model responses

* Training convergence within ~1 k samples per task

* GPU power usage under 80 W; inference on 4-bit model with < 16 GB RAM

* Real-time switching between agents with minimal overhead

## Key Contributions
* Demonstrated feasibility of running advanced LLMs on low-power edge hardware

* Introduced dynamic adapter loading to support multiple healthcare tasks from a single base model

* Integrated live IoT data for truly personalized, context-aware AI-driven health insights

## Research paper published

Here is the link to our research paper on this for detailed information:

[Multi-Agent QLoRA-Optimized LLMs for Personalized Healthcare : M Nithyashree, Krupa P. Nadgir, Pavithra N and Sindhu D. V.] (https://drive.google.com/file/d/1iOIQPCFURs6PIhn7Y7YVTBb5yoASxk0u/view)
