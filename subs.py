from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from bs4 import BeautifulSoup
import subprocess
import shutil
import torch
import tqdm
import ass
import os

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device is {device}.\n")
os.environ['HF_HOME'] = 'cache/'
for dr in ["input, temp, output"]:
	if not os.path.exists(dr):
		os.mkdir(dr)

tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-ru", cache_dir="cache", local_files_only=True, device_map = device)
model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-ru", cache_dir="cache", local_files_only=True, device_map = device)

def translate(text):
	input_ids = tokenizer(text, return_tensors="pt").input_ids
	outputs = model.generate(input_ids=input_ids.to(device), num_beams=5, num_return_sequences=3)
	return tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]

def remove_bad(text):
	if ">" in text:
		text = text.replace("><", "").split(">")[1].split("<")[0]
	if "}" in text:
		text = text.replace("}{", "").split("}")[1].split("{")[0]
	if len(text) < 2 or sum([1 for i in list(r"/|*_^~=+") if i in text]) or sum(i.isdigit() for i in text) > sum(i.isalpha() for i in text):
		return None
	else:
		return text

names = os.listdir("input")
for name in names:
	subprocess.run(f"ffmpeg -i input/{name} -map 0:s:0 temp/{name}.ass")
	with open(f"temp/{name}.ass", encoding='utf_8_sig') as f:
		doc = ass.parse(f)

	pbar = tqdm.tqdm(desc=f"Translating {name}", total=len(doc.events), unit='Dialogue')
	for i in doc.events:
		text0 = i.text
		soup = BeautifulSoup(text0, 'html.parser')
		text = soup.get_text()
		text_new = remove_bad(text)
		if text_new:
			if text_new in text0:
				idx = text0.index(text_new)
				text0 = list(text0)
				text0[idx:idx+len(text_new)] = list(translate(text_new))
				i.text = "".join(text0)
		pbar.update(1)
	pbar.close()

	with open(f"temp/{name}_2.ass", "w", encoding='utf_8_sig') as f:
		doc.dump_file(f)
	subprocess.run(f"ffmpeg -i input/{name} -i temp/{name}_2.ass -codec copy -map 0 -map 1 output/{name}")
	shutil.rmtree('temp')