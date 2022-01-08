import torch
import torch.nn as nn
import numpy as np

###    debut    ###
class IlluinNetwork(nn.Module):

  def __init__(self, embedding_model, device = None, max_word = 100):
    super(IlluinNetwork,self).__init__()
    self.fitted = False
    if device == None:
      self.__device = torch.device("cpu")
    else:
      self.__device = device
    self.embedding_model = embedding_model
    self.max_word = max_word
    self.mean_layer = nn.Linear(max_word,1,dtype=float,bias=False)
    self.mean_layer.weight.data.fill_(1)
    self.proj = torch.from_numpy(np.identity(max_word)).to(self.__device)

  def to(self,device):
    super(IlluinNetwork,self).to(device)
    self.__device = device
    self.proj = self.proj.to(device)
  
  def cpu(self):
    super(IlluinNetwork,self).cpu()
    self.__device = torch.device("cpu")
    self.proj = self.proj.cpu()

  def cuda(self):
    super(IlluinNetwork,self).cuda()
    self.__device = torch.device("cuda:0")
    self.proj = self.proj.cuda()

  def __generate_vector_context(self,x,vector):
    if len(np.array(str(x).split()))>0:
        temp = self.embedding_model[np.array(str(x).split())]
        norm = np.linalg.norm(temp,axis = 1).reshape(-1,1)
        norm[norm==0]=1
        temp = temp/norm
        vector +=[temp]
    else:
        vector +=[np.zeros((1,300))]
  
  def VectorizeQuestion(self,questions):
    vec = []
    for ques in questions:
      self.__generate_vector_context(ques,vec)
    return vec
  
  def __CorrespondanceMatrix(self,X):
    N = np.array([len(x) for x in X])
    return np.concatenate(X),N

  def __smartMean(self,X,N):
    res = []
    y = 0
    for x in N:
        x2 = min(x,self.max_word)
        temp =self.mean_layer(self.proj[0:x2,:]) #(self.proj[0:x,:]@self.W)
        res+=[temp.T@(torch.sort(X[y:y+x,:],0, descending=True).values[0:x2,:])/torch.sum(temp)]
        y+=x
    return torch.concat(res)
  
  def __smartMax(self,X,N):
      res = []
      y = 0
      for x in N:
          res+=[list(np.max(X[:,y:y+x], axis = 1))]
          y+=x
      return  np.array(res).T

  def fit_context(self,pre_context,load = False, XC_path = "tmp/XC.npy", NC_path = "tmp/NC.npy"):
    self.fitted = True
    if(load):
      XC = np.load(XC_path)
      NC = np.load(NC_path)
      self.XC = XC
      self.NC = NC
    else:
      vector_context_fasttext = []
      pre_context['context'].apply(lambda x:self.__generate_vector_context(x, vector_context_fasttext) )
      XC,NC = self.__CorrespondanceMatrix(vector_context_fasttext)
      self.XC = XC
      self.NC = NC
    self.longest_word = max(NC)

  def save_context(self, path_XC = "tmp/XC", path_NC = "tmp/NC"):
    if self.fitted:
      np.save(path_XC,self.XC)
      np.save(path_NC, self.NC)

  def forward(self, X):
    X,N = self.__CorrespondanceMatrix(X)
    smax = torch.from_numpy(self.__smartMax(X@self.XC.T,self.NC)).to(self.__device)
    res = self.__smartMean(smax,N)
    return res

  def predict(self,questions,batch_size = 100, k = 1):
    end = batch_size
    temp = []
    while(end<len(questions)+batch_size):
        if(end>len(questions)):
            temp+=[self.forward(self.VectorizeQuestion(questions[end-batch_size:]))]
            print("{} sur {}".format(len(questions),len(questions)))
            break
        else:

          
            temp+=[self.forward(self.VectorizeQuestion(questions[end-batch_size:end]))]
            print("{} sur {}".format(end,len(questions)) )
            end+=batch_size
    return torch.argsort(torch.concat(temp),axis=1)[:,-k:]
  