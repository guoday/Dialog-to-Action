import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
from torch.nn import init

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size,num_layers=1,bidirectional=True)
        
    def forward(self, input, hidden):
        embedded = self.embedding(input).view(-1, 1, self.hidden_size)
        output = embedded
        output, hidden = self.gru(output, hidden)
        return output,hidden

    def initHidden(self):
        result = Variable(torch.zeros(2, 1, self.hidden_size))
        return result

class EncoderTemplate(nn.Module):
    def __init__(self,hidden_size):
        super(EncoderTemplate, self).__init__()
        self.hidden_size = hidden_size
        self.gru = nn.GRU(hidden_size, hidden_size,num_layers=1,bidirectional=False)
        
    def forward(self, input, hidden):
        output, hidden = self.gru(input, hidden)
        return output,hidden

    def initHidden(self,batch_size):
        result = Variable(torch.zeros(1, batch_size, self.hidden_size))
        return result    
    
class EncoderLf(nn.Module):
    def __init__(self,hidden_size):
        super(EncoderLf, self).__init__()
        self.hidden_size = hidden_size
        self.gru = nn.GRU(hidden_size, hidden_size,num_layers=1,bidirectional=False)
        
    def forward(self, input, hidden):
        output, hidden = self.gru(input, hidden)
        return output,hidden

    def initHidden(self,batch_size):
        result = Variable(torch.zeros(1, batch_size, self.hidden_size))
        return result
    
    
class DecoderRNN(nn.Module):
    def __init__(self, output_size,hidden_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(output_size, hidden_size)
        #pretrained_weight=pickle.load(open('SMP/nparser/model/emb_encoder.pkl','rb'))
        #self.embedding.weight=nn.Parameter(torch.FloatTensor(pretrained_weight))
        self.gru = nn.GRU(hidden_size*3, hidden_size*2)
        self.out = nn.Linear(hidden_size*2, output_size)
        self.proj = nn.Linear(hidden_size*2, hidden_size*2)
        self.proj_ = nn.Linear(hidden_size*2, hidden_size*2)

    def forward(self, input, hidden,encoder_hidden):
        hidden_=self.proj(hidden)[0]
        weights=F.softmax(torch.sum(hidden_[None,:,:]*encoder_hidden,-1),0)
        att=torch.sum(weights[:,:,None]*encoder_hidden,0).view(1,-1,self.hidden_size*2)
        input=torch.cat([input,att],-1)
        output, hidden = self.gru(input, hidden)
        output=self.out(output[0])
        output = F.softmax(output,-1)
        
        hidden_=self.proj_(hidden)[0]
        weights=F.softmax(torch.sum(hidden_[None,:,:]*encoder_hidden,-1),0)
        att=torch.sum(weights[:,:,None]*encoder_hidden,0).view(1,-1,self.hidden_size*2)
        hidden_att=torch.cat([hidden,att],-1)[0]
        return output, hidden[0],hidden_att

    def initHidden(self):
        result = Variable(torch.zeros(1, 1, self.hidden_size*2))
        return result