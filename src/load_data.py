import json
import pandas as pd

def load_raw_data(path):
  with open(path) as json_file:
      data = json.load(json_file)
  i =1
  question_context = []
  context = []
  for title in data['data']:
      for paragraph in title['paragraphs']:
          for question in paragraph['qas']:
              question_context+=[{'question':question['question'],'context':i}]
          context+=[{'idc':i,'context':paragraph['context']}]
          i+=1
  question_context = pd.DataFrame(question_context)
  context = pd.DataFrame(context)
  return question_context,context
