import os

def get_files(path):
  text = []
  ctext = []
  folders = []
  
  for folder in os.listdir(path):
    p = path + folder + "/"
    t_name = [name for name in os.listdir(p) if 'transcript' in name]
    m_name = [name for name in os.listdir(p) if 'minutes' in name]
    
    best_file = None
    best_minute = None
    
    for f in m_name:
      text1 = open(p+f,"r",encoding="utf-8").read().lower()
      if best_minute == None or len(text1) > len(best_minute):
        best_minute = text1
        best_file = f
        
    minute = open(p+best_file,"r",encoding="utf-8").read().lower().replace("\n"," ")
    transcript = open(p+t_name[0],"r",encoding="utf-8").read().lower().replace("\n"," ")
    
    text.append(minute)
    ctext.append(transcript)
    folders.append(folder)
  return [text,ctext,folders]


def get_files_test(path):
  ctext = []
  folders = []
  
  for folder in os.listdir(path):
    p = path + folder + "/"
    t_name = [name for name in os.listdir(p) if 'transcript' in name]
    transcript = open(p+t_name[0],"r",encoding="utf-8").read().lower().replace("\n"," ")
    
    ctext.append(transcript)
    folders.append(folder)
  return [ctext,folders]

def segment_text(input,limit):
  text_segments = []
  sents = ""
  for text in input:
    word_count = 0
    segments = []
    for sent in text.split("."):
      words = sent.split(" ")
      if word_count + len(words) <= limit:
        sents += sent + ". "
        word_count += len(words)
      else:
        word_count = len(words)
        segments.append(sents)
        sents = sent
    text_segments.append(segments)
  return text_segments