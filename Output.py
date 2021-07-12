from transformers import AutoModelForSeq2SeqLM,AutoTokenizer
from utils import get_files_test,segment_text
import os

test_path = "./Files/test/"


if "Output" not in os.listdir():
  os.mkdir("Output")
  
tokenizer = AutoTokenizer.from_pretrained("./Models/Fine_Tuned")
model = AutoModelForSeq2SeqLM.from_pretrained("./Models/Fine_Tuned").to("cuda")

ctext_test , folder_test = get_files_test(test_path)
text_segments = segment_text(ctext_test,500)

for i,(fol,segments) in enumerate(zip(folder_test,text_segments)):
  summary = []
  for segment in segments:
    input_ids = tokenizer("summarize: " + segment, return_tensors="pt").input_ids.to("cuda")

    if(len(segment) < 300):
    	outputs = model.generate(input_ids=input_ids, num_beams=3, num_return_sequences=1,max_length=100)
    elif (len(segment) < 150):
    	outputs = model.generate(input_ids=input_ids, num_beams=3, num_return_sequences=1,max_length=50)
    else:
    	outputs = model.generate(input_ids=input_ids, num_beams=3, num_return_sequences=1,max_length=300)

    summary.append(tokenizer.batch_decode(outputs, skip_special_tokens=True))

  content = ""
  for seg in summary:
    seg = seg[0].replace(". ",".\n")
    content += seg +".\n"
  
  open("./Output/minutes_{}.txt".format(fol.split("_",1)[1]),"w").write(content)
  print("Completed {}/{} files".format(i+1,len(folder_test)))