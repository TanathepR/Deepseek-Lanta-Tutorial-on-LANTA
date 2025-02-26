# คู่มือการรัน DeepSeek บน LANTA

## แนะนำ
**LANTA** เป็นซูเปอร์คอมพิวเตอร์ของประเทศไทยที่บริหารจัดการโดย **NSTDA Supercomputer Center (ThaiSC)** ซึ่งเป็นเครื่องมือที่ทรงพลังสำหรับการรันโมเดล AI เช่น **DeepSeek** ที่ใช้สำหรับงานประมวลผลภาษาธรรมชาติ (NLP) บทความนี้จะอธิบายขั้นตอนการตั้งค่าและรันโมเดล DeepSeek บน LANTA

## สิ่งที่ต้องเตรียมก่อนเริ่มต้น
ก่อนที่จะเริ่มต้นใช้งาน DeepSeek บน LANTA คุณต้องมี:
1. **บัญชีผู้ใช้บน LANTA HPC** (หากยังไม่มีสามารถขอสิทธิ์การใช้งานได้จาก ThaiSC)
2. **ความเข้าใจพื้นฐานเกี่ยวกับคำสั่ง Linux**
3. **ทักษะการใช้งาน Terminal และ SSH**
4. **ความคุ้นเคยกับภาษา Python**

## ขั้นตอนที่ 1: เข้าสู่ระบบ LANTA
ใช้ **SSH** เพื่อเข้าถึง LANTA จากเครื่องคอมพิวเตอร์ของคุณ โดยใช้คำสั่ง:
```sh
ssh your_username@lanta.nstda.or.th
```
ใส่รหัสผ่านและรหัสยืนยันเมื่อระบบร้องขอ

## ขั้นตอนที่ 2: โหลดโมดูลที่จำเป็น
LANTA ใช้ระบบ **environment module** ในการจัดการซอฟต์แวร์ ก่อนใช้งาน DeepSeek ให้โหลดโมดูลที่จำเป็นโดยใช้คำสั่ง:
```sh
module use /project/cb9009xx-hpctxx/modules
module load nano miniconda deepseek
```
คำสั่งนี้จะทำให้แน่ใจว่าระบบมี Python พร้อมใช้งาน

## ขั้นตอนที่ 5: ทดสอบรัน DeepSeek
สร้างไฟล์ **deepseek_inference.py**
```sh
nano deepseek_inference.py
```
คัดลอกและวางโค้ดต่อไปนี้:
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

## ขั้นตอนที่ 6: รันสคริปต์บน LANTA
ใช้คำสั่งต่อไปนี้เพื่อรันสคริปต์:
```sh
python deepseek_inference.py
```
ระบบจะโหลดโมเดล **DeepSeek** และสร้างผลลัพธ์จากข้อความที่กำหนด

## ขั้นตอนที่ 7: ส่งงานแบบ Batch Job (สำหรับงานขนาดใหญ่)
สำหรับงานที่ต้องใช้ทรัพยากรมาก สามารถใช้ **Slurm** เพื่อรันเป็น batch job ได้ โดยสร้างไฟล์ **deepseek_job.sh**
```sh
nano deepseek_job.sh
```
เพิ่มเนื้อหาต่อไปนี้:
```sh
#!/bin/bash
#SBATCH -p gpu
#SBATCH --job-name=deepseek_test
#SBATCH -N 1 -c 16    # จำนวนโหนดและ CPU ต่อ task
#SBATCH --gpus-per-task=1  # ใช้ 1 GPU ต่อ task
#SBATCH --ntasks-per-node=4  # จำนวน task ต่อโหนด
#SBATCH -A cb9009xx
#SBATCH --time=00:30:00
#SBATCH --output=deepseek_output.log
#SBATCH --error=deepseek_error.log

module use /project/cb900902-hpct01/modules
module load nano miniconda tree

python deepseek_inference.py
```

ส่งงานด้วยคำสั่ง:
```sh
sbatch deepseek_job.sh
```
คำสั่งนี้จะรัน DeepSeek บน **1 GPU node** โดยใช้ **16 CPU cores และ 32GB RAM**

## สรุป
คุณได้เรียนรู้วิธีติดตั้งและใช้งาน **DeepSeek** บน **LANTA HPC** สำเร็จแล้ว ตอนนี้คุณสามารถทดลองปรับแต่ง prompt และสำรวจฟีเจอร์ต่าง ๆ ของโมเดลเพิ่มเติมได้!

