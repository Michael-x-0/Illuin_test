import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def train (model,data, epochs =1,batch_size = 100):
  
  #Creation du dataLoader
  X_train = data.question.values
  y_train = torch.tensor(data.context.values -1)
  loader = DataLoader([[x,y] for (x,y) in zip(X_train,y_train)] ,shuffle = True, batch_size=batch_size)

  #Entraînement
  model.train()
  criterion = nn.CrossEntropyLoss()
  learning_rate = 0.1
  optimizer = torch.optim.SGD(model.parameters(),lr=learning_rate)

  for epoch in range(epochs):
    running_loss = 0.0
    running_corrects = 0
    print('EPOCH {}:'.format(epoch+1))
    j=0
    for X,y in loader:
      out = model.forward(model.VectorizeQuestion(X))
      y = y.to(device)
      loss = criterion(out,y)
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      
      running_loss += loss.data.item()
      preds = torch.argsort(out,axis=1)[:,-1:]
      running_corrects += torch.sum(y==preds.flatten())
      print("\t batch {} loss {} correct {}".format(j*batch_size+len(X),loss.data.item(),torch.sum(y==preds.flatten()).data.item()))
      j+=1
    epoch_loss = running_loss /len(loader)
    epoch_acc = running_corrects.data.item()/len(loader.dataset)
    print('Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))

def eval (model,data,batch_size = 100, k = 1, device = torch.device('cpu')):

  #Creation du dataLoader
  X_eval = data.question.values
  y_eval = torch.tensor(data.context.values -1)
  loader = DataLoader([[x,y] for (x,y) in zip(X_eval,y_eval)] ,shuffle = True, batch_size=batch_size)

  model.eval()
  running_corrects = 0
  j=0
  print("evaluation of {} data".format(len(loader.dataset)))
  for X,y in loader:
    out = model.forward(model.VectorizeQuestion(X))
    y = y.to(device)
    preds = torch.argsort(out,axis=1)[:,-k:]
    correct,acc = nb_correct_acc(preds,y)
    running_corrects += correct
    print("\t batch : {} correct : {} Acc : {:.4f}".format(j*batch_size+len(X),correct,acc))
    j+=1
  acc = running_corrects/len(loader.dataset)
  print('{} - Acc: {:.4f}'.format(k,acc))

def nb_correct_acc(preds,context):
    context = context 
    s = 0
    for i,x in enumerate(preds):
        if context[i] in x:
            s +=1
    return s,s/len(preds)