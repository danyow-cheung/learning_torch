import torch 
from torch import nn 
from d2l import torch as d2l 
class AttentionDecoder(d2l.Decoder):
    
    def __init__(self, **kwargs):
        super(AttentionDecoder,self).__init__(**kwargs)
    
    @property
    def attention_weights(self):
        raise NotImplementedError



class Seq2SeqAttentionDecoder(AttentionDecoder):
    '''
    實現帶有Bahdanau注意力的循環神經網路解碼器
    首先初始化解碼器狀態，需要下面的輸入
    1. 編碼器在所有時間步的最終層隱狀態，將做為注意力的鍵和值
    2. 上一時間步的編碼器全層隱狀態，將作為初始化隱狀態的隱狀態
    3. 編碼器的有效長度
    '''
    def __init__(self,vocab_size,embed_size,num_hiddens,num_layers,dropout=0, **kwargs):
        super(Seq2SeqAttentionDecoder,self).__init__(**kwargs)
        self.attention = d2l.AdditiveAttention(
            num_hiddens,num_hiddens,num_hiddens,dropout
        )
        self.embedding = nn.Embedding(vocab_size,embed_size)
        self.rnn = nn.GRU(
            embed_size+num_hiddens,
            num_hiddens,
            num_layers,
            dropout=dropout
        )
        self.dense = nn.Linear(num_hiddens,vocab_size)
    
    def init_state(self, enc_outputs,enc_valid_lens, *args):
        # outputs的形狀(batch_size,num_steps,num_hiddens)
        # hidden_state形狀(num_layers,batch_size,num_hiddens)
        outputs,hidden_state = enc_outputs

        return (outputs.permute(1,0,2),hidden_state,enc_valid_lens)
    
    def forward(self, X, state):
        # enc_outputs的形狀(batch_size,num_steps,num_hiddens)
        # hidden_state形狀(num_layers,batch_size,num_hiddens)
        enc_outputs,hidden_state ,enc_valid_lens = state 
        # 輸出X的形狀(num_steps,batch_size,embed_size)
        X= self.embedding(X).permute(1,0,2)
        outputs,self._attention_weights = [],[]
        for x in X:
            # query得形狀(batch_size,1,num_hiddens)
            query = torch.unsqueeze(hidden_state[-1],dim=1)
            # context 的形狀(batch_size,1,num_hiddens)
            context = self.attention(
                query,enc_outputs,enc_outputs,enc_valid_lens
            )
            # 在特徵維度上連結
            x = torch.cat((context, torch.unsqueeze(x, dim=1)), dim=-1)
            # 將x變形為
            out,hidden_state = self.rnn(x.permute(1,0,2),hidden_state)
            outputs.append(out)
            self._attention_weights.append(self.attention.attention_weights)
        # 全連結層變換後,outputs的形狀為
        # (num_steps,batch_size,vocab_size)
        outputs = self.dense(torch.cat(outputs,dim=0))
        return outputs.permute(1,0,2),[enc_outputs,hidden_state,enc_valid_lens]
    @property
    def attention_weights(self):
        return self._attention_weights


if __name__ =='__main__':
    # 使用7個時間步長的序列輸入小批量測試Bahdanau注意力解碼器
    # encoder = d2l.Seq2SeqEncoder(vocab_size=10,embed_size=8,num_hiddens=16,num_layers=2)
    # encoder.eval()
    # decoder = Seq2SeqAttentionDecoder(vocab_size=10,embed_size=8,num_hiddens=16,num_layers=2)
    # decoder.eval()

    # X = torch.zeros((4,7),dtype=torch.long)
    # state = decoder.init_state(encoder(X),None)
    # output,state = decoder(X,state)
    # print(output.shape,len(state),state[0].shape,len(state[1]),state[1][0].shape)


    # 訓練
    embed_size = 32 
    num_hiddens = 32 
    num_layers = 2 
    dropout = 0.1 
    batch_size = 64 
    num_steps = 10 

    lr,num_epochs = 0.005,1
    train_iter,src_vocab ,tgt_vocab = d2l.load_data_nmt(
        batch_size,num_steps,
    )
    encoder=d2l.Seq2SeqEncoder(
        len(src_vocab),embed_size,num_hiddens,num_layers,dropout,
        )
    decoder = Seq2SeqAttentionDecoder(
        len(tgt_vocab),embed_size,num_hiddens,num_layers,dropout
    )

    net = d2l.EncoderDecoder(encoder,decoder)
    d2l.train_seq2seq(net,train_iter,lr,num_epochs,tgt_vocab,device='cpu')


    # 計算BLEIU分數
    engs = ['go .', "i lost .", 'he\'s calm .', 'i\'m home .']
    fras = ['va !', 'j\'ai perdu .', 'il est calme .', 'je suis chez moi .']
    for eng,fra in zip(engs,fras):
        translation,dec_attention_weight_seq = d2l.predict_seq2seq(
            net,eng,src_vocab,tgt_vocab,num_steps,'cpu',True
        )
        print(f'{eng}=>{translation}',f'BLEU{d2l.bleu(translation,fra,k=2):.3f}')



