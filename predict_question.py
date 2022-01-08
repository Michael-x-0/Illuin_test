# importing the required modules
import os
import argparse
from src.model import IlluinNetwork
from src.preprocess import preprocess
import pandas as pd


import compress_fasttext

# error messages
INVALID_FILETYPE_MSG = "Error: Invalid file format. %s must be a .json file."
INVALID_PATH_MSG = "Error: Invalid file path/name. Path %s does not exist."


def validate_file(file_name):
	'''
	validate file name and path.
	'''
	if not valid_path(file_name):
		print(INVALID_PATH_MSG%(file_name))
		quit()
	elif not valid_filetype(file_name):
		print(INVALID_FILETYPE_MSG%(file_name))
		quit()
	return
	
def valid_filetype(file_name):
	# validate file type
	return file_name.endswith('.json')

def valid_path(path):
	# validate file path
	return os.path.exists(path)
		
	

def main():
  # create parser object
  parser = argparse.ArgumentParser(description = "Prédiction du contexte correspondant à une question")

  # defining arguments for parser object
  parser.add_argument("question", type = str, nargs = 1,
            metavar = "question",
            help = "question dont on veut prédire le contexte")

  


  # parse the arguments from standard input
  args = parser.parse_args()
  question = args.question[0]


  print("loading Fasttext...")
  fasttext_model = compress_fasttext.models.CompressedFastTextKeyedVectors.load('fasttext/fasttext.model')
  
  print("loading context file...")
  context = pd.read_csv("tmp/context.csv")
  model = IlluinNetwork(embedding_model = fasttext_model)
  model.fit_context(None,load = True)
  print("preprocessing de la question")
  pre_ques = preprocess(question)
  pred = model.predict([pre_ques])
  print("le contexte est : "+context[context['idc']==pred[0][0].data.item()].iloc[0,1])

  


if __name__ == "__main__":
	# calling the main function
	main()
