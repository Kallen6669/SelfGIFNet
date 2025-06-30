import jittor as jt
import jittor.nn as nn

class real_GIFNet(nn.Module):
  def __init__(self, s, n, channel, stride):
          super(GIFNet, self).__init__()
          pass
  #trainingTag = 1, IVIF task; trainingTag = 2, MFIF task;
  def forward_MultiTask_branch(self, fea_com_ivif, fea_com_mfif, trainingTag = 2):
      pass
      # x = self.extractor_multask(fea_com_ivif, fea_com_mfif, trainingTag);
      # return x;
  
  def forward_mixed_decoder(self, fea_com, fea_fused):
      pass
      # x = self.cnnDecoder([fea_com,fea_fused]);
      # return x;
      
  def forward_rec_decoder(self, fea_com):
      pass
      # return self.decoder_rec(fea_com);                
      
  def forward(self, x, y):
      fea_com = self.forward_encoder(x, y);
      output = self.forward_MultiTask_branch(fea_com_ivif = fea_com, fea_com_mfif = fea_com, trainingTag = 2)   
      return output

# 测试用
class GIFNet(nn.Module):
  def __init__(self, s, n, channel, stride):
        super(GIFNet, self).__init__()
        pass
  def forward(self, x, y):
      output = x
      return output
