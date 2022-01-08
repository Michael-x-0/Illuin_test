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
  parser = argparse.ArgumentParser(description = "Evaluation sur un dataset SQUAD pour un nombre donné de questions")

  # defining arguments for parser object
  parser.add_argument("dataset", type = str, nargs = 1,
            metavar = "dataset_file",
            help = "charge le dataset pour l'évaluation")

  parser.add_argument("-n", "--nombre", type = int, nargs = 1,
            metavar = "nb_echantillons", default = None,
            help = "nombre d'échantillons (questions) du dataset sur lesquelles faire l'évaluation")

  parser.add_argument("-k", "--k_accuracy", type = int, nargs = 1,
            metavar = "top_k_accuracy", default = [1],
            help = "nombre de prédiction à prendre en compte dans l'évaluation")

  parser.add_argument("-b", "--bach_size", type = int, nargs = 1,
            metavar = "bach_size", default = [100],
            help = "nombre de questions préditent en parallèle par itération")


  # parse the arguments from standard input
  args = parser.parse_args()
  file_name = args.dataset[0]
  validate_file(file_name)

  print("loading dataset...")
  question_context,context = load_raw_data(file_name)
  n = args.nombre[0] if args.nombre[0]!=None else test_pre_question_context.shape[0]
  print("preprocessing...")
  context['context'] = context['context'].apply(lambda x:preprocess(x))
  question_context = question_context.sample(n)
  question_context['question'] = question_context['question'].apply(lambda x:preprocess(x))

  print("loading Fasttext...")
  fasttext_model = compress_fasttext.models.CompressedFastTextKeyedVectors.load('fasttext/fasttext.model')

  model = IlluinNetwork(embedding_model = fasttext_model)
  print("fitting context")
  model.fit_context(context)
  k = args.k_accuracy[0]
  bach_size = args.bach_size[0]
  eval(model,question_context,batch_size = bach_size, k = k)

  


if __name__ == "__main__":
	# calling the main function
	main()
