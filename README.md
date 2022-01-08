### Install dependencies
!pip installpip install compress-fasttext

### Présentation générale
Consulter le notebook presentation.ipynb

### Fonctionnalités:
1- Si vous voulez évaluer la performance de notre modèle sur un dataset SQUAD il faut exécuté le script "evaluate_model.py". Exemple: "python evaluate_model.py -h" pour voir comment ça s'utilise

2- Si vous voulez prédire le contexte pour une question donnée, il faut d'abord exécuter le script "fit_context.py" avec en argument le chemin du dataset SQUAD. Ce commande créera des fichiers essentiels dans le dossier tmp. Ensuite, il faut exécuter le script "predict_question.py" en passant la question en argument. À chaque fois la commande "python script.py -h" permet de savoir comment s'utilise un script donné.
