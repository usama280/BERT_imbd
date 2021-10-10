import torch
import pytorch_lightning as pl
import torch.nn as nn

import nlp
import transformers
#import IPython; IPython.embed(); exit(1)

torch.version.cuda

torch.__version__

import torch.nn as nn
import torch.nn.functional as F
import torch

import math


class Attention(nn.Module):
    """
    Compute 'Scaled Dot Product Attention
    """

    def forward(self, query, key, value, mask=None, dropout=None):
        scores = torch.matmul(query, key.transpose(-2, -1))/math.sqrt(query.size(-1)) #Attention formula QK^T/SqRt(dk)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9) #if mask element is 0, shut it off

        p_attn = F.softmax(scores, dim=-1) #dim=-1 --> key

        if dropout is not None:
            p_attn = dropout(p_attn)

        return torch.matmul(p_attn, value), p_attn #Scale down values


class MultiHeadedAttention(nn.Module):
    """
    Take in model size and number of heads.
    """
    
    def __init__(self, h, d_model, dropout=0.1):
        super().__init__()
        
        assert d_model % h == 0 #embed needs to be divisible by h
        
        # We assume d_v always equals d_k
        self.d_k = d_model // h  
        self.h = h

        self.linear_layers = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(3)]) #k,v,q
        self.output_linear = nn.Linear(d_model, d_model)#fc output
        
        self.attention = Attention()
        self.dropout = nn.Dropout(p=dropout)
        
        
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0) 

        # 1) Do all the linear projections in batch from d_model => h x d_k  -??
        query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linear_layers, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, attn = self.attention(query, key, value, mask=mask, dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k) #Flattening

        return self.output_linear(x) #fc

class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

    
class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        
        self.norm = LayerNorm(size) #Normalize per example (Batchnorm does it per batch)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x))) #+ --> residual connect


class GELU(nn.Module):
    """
    Paper Section 3.4, last paragraph notice that BERT used the GELU instead of RELU
    """

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
    
    
class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = GELU() #Could replace with ReLU

    def forward(self, x):
        return self.w_2(self.dropout(self.activation(self.w_1(x))))

class TransformerBlock(nn.Module):
    """
    Bidirectional Encoder = Transformer (self-attention)
    Transformer = MultiHead_Attention + Feed_Forward with sublayer connection
    """

    def __init__(self, hidden, attn_heads, feed_forward_hidden, dropout):
        """
        :param hidden: hidden size of transformer
        :param attn_heads: head sizes of multi-head attention
        :param feed_forward_hidden: feed_forward_hidden, usually 4*hidden_size
        :param dropout: dropout rate
        """

        super().__init__()
        self.attention = MultiHeadedAttention(h=attn_heads, d_model=hidden) #Attention
        
        #Layernorm and dropout
        self.input_sublayer = SublayerConnection(size=hidden, dropout=dropout)
        self.output_sublayer = SublayerConnection(size=hidden, dropout=dropout)
                                                                        #Expansion*d_model
        self.feed_forward = PositionwiseFeedForward(d_model=hidden, d_ff=feed_forward_hidden, dropout=dropout)
        
        self.dropout = nn.Dropout(p=dropout)

        
    def forward(self, x, mask):
                                                                      #k,v,q
        x = self.input_sublayer(x, lambda _x: self.attention.forward(_x, _x, _x, mask=mask))
        x = self.output_sublayer(x, self.feed_forward)
        return self.dropout(x)

class TokenEmbedding(nn.Embedding):
    def __init__(self, vocab_size, embed_size=512):
        super().__init__(vocab_size, embed_size, padding_idx=0)
        

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]
    
    
class SegmentEmbedding(nn.Embedding):
    def __init__(self, embed_size=512):
        super().__init__(3, embed_size, padding_idx=0)

class BERTEmbedding(nn.Module):
    """
    BERT Embedding which is consisted with under features
        1. TokenEmbedding : normal embedding matrix
        2. PositionalEmbedding : adding positional information using sin, cos
        2. SegmentEmbedding : adding sentence segment info, (sent_A:1, sent_B:2)
        sum of all these features are output of BERTEmbedding
    """

    def __init__(self, vocab_size, embed_size, dropout=0.1):
        """
        :param vocab_size: total vocab size
        :param embed_size: embedding size of token embedding
        :param dropout: dropout rate
        """
        super().__init__()
        
        self.token = TokenEmbedding(vocab_size=vocab_size, embed_size=embed_size)
        self.position = PositionalEmbedding(d_model=self.token.embedding_dim)
        self.segment = SegmentEmbedding(embed_size=self.token.embedding_dim)
        
        self.dropout = nn.Dropout(p=dropout)
        self.embed_size = embed_size

    '''
    def forward(self, sequence, segment_label):
        x = self.token(sequence) + self.position(sequence) + self.segment(segment_label)
        return self.dropout(x)
    '''
    def forward(self, sequence):
        x = self.token(sequence) + self.position(sequence)
        return self.dropout(x)

class BERT(nn.Module):
    def __init__(self, vocab_size, hidden=768, n_layers=12, attn_heads=12, dropout=0.1):
        """
        :param vocab_size: vocab_size of total words
        :param hidden: BERT model hidden size
        :param n_layers: numbers of Transformer blocks(layers)
        :param attn_heads: number of attention heads
        :param dropout: dropout rate
        """

        super().__init__()
        self.hidden = hidden #Hidden size ??
        self.n_layers = n_layers #Num of transformer blocks
        self.attn_heads = attn_heads #Num of attention heads

        # paper noted they used 4*hidden_size for ff_network_hidden_size
        self.feed_forward_hidden = hidden * 4

        # embedding for BERT, sum of positional, segment, token embeddings
        self.embedding = BERTEmbedding(vocab_size=vocab_size, embed_size=hidden)

        # multi-layers transformer blocks, deep network
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(hidden, attn_heads, hidden * 4, dropout) for _ in range(n_layers)])        
        
    '''
    def forward(self, x, segment_info):
        # attention masking for padded token
        # torch.ByteTensor([batch_size, 1, seq_len, seq_len)
        mask = (x > 0).unsqueeze(1).repeat(1, x.size(1), 1).unsqueeze(1)

        # embedding the indexed sequence to sequence of vectors
        x = self.embedding(x, segment_info)

        # running over multiple transformer blocks
        for transformer in self.transformer_blocks:
            x = transformer.forward(x, mask) #k,v,q, mask
        
#         x = x.reshape(x.shape[0],-1)
#         x = nn.Linear(self.hidden, 2)
        
        return x
    '''    
    def forward(self, x):
        # attention masking for padded token
        # torch.ByteTensor([batch_size, 1, seq_len, seq_len)
        mask = (x > 0).unsqueeze(1).repeat(1, x.size(1), 1).unsqueeze(1)

        # embedding the indexed sequence to sequence of vectors
        x = self.embedding(x)

        # running over multiple transformer blocks
        for transformer in self.transformer_blocks:
            x = transformer.forward(x, mask) #k,v,q, mask
        
#         x = x.reshape(x.shape[0],-1)
#         x = nn.Linear(self.hidden, 2)
        
        return x

class BERTseqclass(nn.Module):
    def __init__(self, bert, vocab_size, c_out):
        
        super().__init__()
        self.bert = bert(vocab_size)
        
        self.fc1 = nn.Linear(self.bert.hidden,self.bert.hidden)
        self.activ = nn.Tanh()
        self.dropout = nn.Dropout(p=.1)
        self.fc2 = nn.Linear(self.bert.hidden, c_out)
        
        
        
    def forward(self, x):
        x = self.bert(x)
        x = self.fc1(x)
        x = self.activ(x)
        #x = x.reshape(x.shape[0],-1) ??
        x = self.fc2(x)
        
        return x

class IMDBSentiClassifier(pl.LightningModule):
    def __init__(self):
        super().__init__()
        #self.model = transformers.BertForSequenceClassification.from_pretrained('bert-base-uncased')
        self.model = BERTseqclass(BERT, 32000, 2)
        self.loss = torch.nn.CrossEntropyLoss(reduction='none')
        
        
    def prepare_data(self):
        tokenizer = transformers.BertTokenizerFast.from_pretrained('bert-base-uncased')
        
        def _tokenize(x):
            #contains both text and encoded values
            x['input_ids'] = tokenizer.encode(
                    x['text'], 
                    max_length=32, 
                    pad_to_max_length=True)
            
            return x
        
        def _prepare_ds(folder):
            ds = nlp.load_dataset('imdb', split=f'{folder}[:5%]')
            ds = ds.map(_tokenize)
            ds.set_format(type='torch', columns=['input_ids', 'label'])
            
            return ds
        
        
        self.train_ds, self.test_ds = map(_prepare_ds, ('train', 'test'))
        
        
    def forward(self, input_ids):
        #mask = (input_ids != 0).float()
        #preds = self.model(input_ids,mask)
        preds = self.model(input_ids) #,mask
        return preds
    
    
    
    def training_step(self, batch, batch_idx):
        preds = self.forward(batch['input_ids'])
        loss = self.loss(preds, batch['label']).mean()
        self.log('train_loss', loss)
        return {'loss':loss, 'log':{'train_loss':loss}}

    
    def validation_step(self, batch, batch_idx):
        preds = self.forward(batch['input_ids'])
        loss = self.loss(preds, batch['label'])
        acc = (preds.argmax(-1)==batch['label']).float()
        
        return {'loss':loss, 'acc':acc}
    
    
    def validation_epoch_end(self, outputs):
        loss = torch.cat([o['loss'] for o in outputs], 0).mean()
        acc = torch.cat([o['acc'] for o in outputs], 0).mean()
        out = {'val_loss':loss, 'val_acc':acc}
        self.log('val_loss', loss)
        self.log('val_acc', acc)
        return {**out, 'log':out}#appending dic **  
    
    
    
    def train_dataloader(self):
        return torch.utils.data.DataLoader(
                    self.train_ds,
                    batch_size=8,
                    drop_last=True,
                    shuffle=True
                )
    
    def val_dataloader(self):
        return torch.utils.data.DataLoader(
                    self.test_ds,
                    batch_size=8,
                    drop_last=False,
                    shuffle=False
                )
    
    
    def configure_optimizers(self):
        return torch.optim.SGD(
                    self.parameters(),
                    lr=1e-2,
                    momentum=.9
                )

def main():
    model = IMDBSentiClassifier() 
    
    trainer = pl.Trainer(
        default_root_dir='logs',
        gpus=(1 if torch.cuda.is_available() else 0),
        max_epochs=2,
        logger=pl.loggers.TensorBoardLogger('logs/', name='imdb', version=0)
    )
    
    trainer.fit(model)

main()