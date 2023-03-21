from torch import nn
from Encoder import Encoder
from Decoder import Decoder 
'''合并编码器和解码器'''
class EncoderDecoder(nn.Module):
	def __init__(self,encoder,decoder,**kwargs):
		super(EncoderDecoder,self).__init__(**kwargs)
		self.encoder = encoder
		self.decoder = decoder

	def forward(self,enc_X,dec_X,*args):
		enc_outputs = self.encoder(enc_X,*args)
		dec_state = self.decoder.init_state(enc_outputs,*args)
		return self.decoder(dec_X,dec_state)


