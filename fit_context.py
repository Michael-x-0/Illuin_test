# importing the required modules
import os
import argparse
from src.load_data import load_raw_data
from src.utils import eval
from src.model import IlluinNetwork
from src.preprocess import preprocess


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
  parser = argparse.ArgumentParser(description = "génération de fichiers indispensable la prédiction sur un dataset")

  # defining arguments for parser object
  parser.add_argument("dataset", type = str, nargs = 1,
            metavar = "dataset_file",
            help = "charge le dataset pour l'évaluation")


  # parse the arguments from standard input
  args = parser.parse_args()
  file_name = args.dataset[0]
  validate_file(file_name)

  print("loading dataset...")
  question_context,context = load_raw_data(file_name)
  context.to_csv("data/context.csv",index = None)
  print("preprocessing...")
  context['context'] = context['context'].apply(lambda x:preprocess(x))

  print("loading Fasttext...")
  fasttext_model = compress_fasttext.models.CompressedFastTextKeyedVectors.load('fasttext/fasttext.model')

  model = IlluinNetwork(embedding_model = fasttext_model)
  print("fitting context")
  model.fit_context(context)
  print("creating file")
  model. save_context(path_XC = "tmp/XC", path_NC = "tmp/NC")

  


if __name__ == "__main__":
	# calling the main function
	main()
