# Tutorial: Running DeepSeek Model on LANTA

## Introduction
LANTA is Thailand's supercomputer, managed by the NSTDA Supercomputer Center (ThaiSC). It is a powerful tool for running AI models like **DeepSeek**, which is used for natural language processing (NLP) tasks. In this tutorial, we will guide you through setting up DeepSeek on LANTA and running a simple inference.

## Prerequisites
Before you begin, you should:
1. Have an account on **LANTA HPC** (request access from ThaiSC if necessary).
2. Understand basic **Linux commands**.
3. Know how to use a terminal and SSH.
4. Be familiar with Python programming.

## Step 1: Logging into LANTA
To access LANTA, use **SSH** from your local computer. Open a terminal and enter:
```sh
ssh your_username@lanta.nstda.or.th
```
Enter your password and verify code when prompted.

## Step 2: Loading Required Modules
LANTA uses an environment module system. Before running DeepSeek, load the necessary software:
```sh
module load cuda/11.8
module load python/3.10
```
These commands ensure your environment has CUDA (for GPU support) and Python.

## Step 3: Creating a Virtual Environment
A virtual environment isolates your Python packages.
```sh
python -m venv deepseek_env
source deepseek_env/bin/activate
```
Now, you are inside the virtual environment, and you can install required libraries.

## Step 4: Installing DeepSeek and Dependencies
Use **pip** to install the DeepSeek model and required libraries:
```sh
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers deepseek
```
These commands install **PyTorch** (with CUDA support) and **DeepSeek**.

## Step 5: Running DeepSeek Inference
Create a new Python script `deepseek_inference.py`:
```sh
nano deepseek_inference.py
```
Copy and paste the following code:
```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-coder-6.7b")
model = AutoModelForCausalLM.from_pretrained("deepseek-ai/deepseek-coder-6.7b", torch_dtype=torch.float16, device_map="auto")

text = "What is artificial intelligence?"
inputs = tokenizer(text, return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_length=200)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```
Save the file and exit (Ctrl+X, then Y, then Enter).

## Step 6: Running the Script on LANTA
To execute the script, run:
```sh
python deepseek_inference.py
```
This will load the **DeepSeek model** and generate a response based on the given text.

## Step 7: Submitting as a Batch Job (Optional)
For large-scale tasks, create a Slurm job script (`deepseek_job.slurm`):
```sh
nano deepseek_job.slurm
```
Add the following content:
```sh
#!/bin/bash
#SBATCH --job-name=deepseek_test
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gpus=1
#SBATCH --mem=32GB
#SBATCH --time=00:30:00
#SBATCH --output=deepseek_output.log

module load cuda/11.8
module load python/3.10
source deepseek_env/bin/activate
python deepseek_inference.py
```
Submit the job using:
```sh
sbatch deepseek_job.sh
```
This will run DeepSeek on **one GPU node** with **8 CPU cores and 32GB RAM**.

## Conclusion
You have successfully set up and run **DeepSeek** on **LANTA HPC**. You can now experiment with different prompts and explore more advanced features of the model.

