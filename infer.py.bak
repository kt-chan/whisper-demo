import os, sys, time
import pprint

import torch
import torch_npu
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from datasets import load_dataset

print('argument list: ', sys.argv)
audio_file = sys.argv[1]

device = "npu:0"
torch_dtype = torch.float16
current_dir = os.path.dirname(os.path.realpath(__file__))
model_id = "/mnt/remote/models/whisper/whisper-large-v3/"

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
)
model.to(device)

processor = AutoProcessor.from_pretrained(model_id)

pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    max_new_tokens=128,
    chunk_length_s=30,
    batch_size=16,
    return_timestamps=True,
    torch_dtype=torch_dtype,
    device=device,
)

print(f'preload for warn up ....')
pipe(current_dir+"/voice/1715956235679.mp3", generate_kwargs={"language": "cantonese"})
#print(f'Loaded moded: {model_id}')
#start = time.time()
#result = pipe(current_dir+"/voice/1715956235679.mp3", generate_kwargs={"language": "cantonese"})
#duration = time.time() - start
#print(result["text"])
#print(f"Duration: {duration * 1000}ms")

start = time.time()
result = pipe(current_dir+"/"+audio_file, generate_kwargs={"language": "cantonese"})
duration = time.time() - start
print(result["text"])
print(f"Duration: {duration * 1000}ms")
