import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import matplotlib.pyplot as plt
from d2l import torch as d2l


def sequence_mask(X, valid_len, value=0):
    maxlen = X.size(1)
    # 先生成X中的列下标
    mask = torch.arange((maxlen), dtype=torch.float32,
                        device=X.device)
    # 添加一个维度变成(1, sequence_len)
    mask = mask.unsqueeze(0)
    # 添加一个维度变成(batch_size, 1)
    valid_len = valid_len.unsqueeze(1)
    # >=这个最大len的下标变为value
    # 广播之后变成batch_size,sequence_len,每一个batch_size的大于valid_len的下标都为True
    index = mask >= valid_len
    X[index] = value
    return X


def mask_softmax(X, valid_lens):
    # X是注意力权重，(batch_size, sequence_length, sequence_length)
    if valid_lens == None:
        return F.softmax(X, dim=-1)
    else:
        shape = X.shape
        # 把valid_lens展开为一维 [[2],[3]] -> [2,3] 分别对应batch中每个sequence的合法长度
        if valid_lens.dim() == 1:
            valid_lens = torch.repeat_interleave(valid_lens, shape[1])
        else:
            valid_lens = valid_lens.reshape(-1)
        # 把X变为batch * sequence_length, sequence_length)后进行mask修饰，用小值替代目的是做softmax后的值为0
        # 第一个seq_len维度代表查询序列(Query)的索引 也就是输入的Query的下标，第二个seq_len维度代表值序列(Value)的索引 也就是这个Value下标分配的权重
        # 第一个seq_len维度仅用于匹配Value序列, 对第二个seq_len维度做softmax用来当权重，相当于，所以只需要让第二个seq_len维度的权重为0来掩盖即可
        X = sequence_mask(X.reshape(-1, shape[-1]), valid_lens, 1e-9)
        return F.softmax(X.reshape(shape), dim=-1)


class DotProductAttention(nn.Module):
    def __init__(self, dropout, **kwargs) -> None:
        super().__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)

    def forward(self, Q, K, V, valid_lens):
        # (batch_size, sequence_length, hidden_num)
        d = Q.shape[-1]
        # (batch_size, sequence_length, sequence_length)
        # print(Q.shape)
        # print(K.shape)
        # print(K.transpose(1, 2).shape)
        weight = torch.bmm(Q, K.transpose(1, 2)) / math.sqrt(d)

        weight = mask_softmax(weight, valid_lens)
        # (batch_size, sequence_length, hidden_num)
        return torch.bmm(self.dropout(weight), V)


def transpose_qkv(X, num_heads):
    # (batch_size, sequence_length, hidden_num)
    # (batch_size, sequence_length, num_head, hidden_num / num_head)
    X = X.reshape(X.shape[0], X.shape[1], num_heads, -1)
    # (batch_size, num_head, sequence_length, hidden_num / num_head)
    X = X.permute(0, 2, 1, 3)
    # (batch_size * num_head, sequence_length, hidden_num / num_head)
    return X.reshape(-1, X.shape[2], X.shape[3])


def transpose_output(X, num_head):
    # (batch_size` * num_heads, sequence_length, num_hiddens / num_heads)
    # (batch_size, num_head, sequence_length, num_hiddens / num_heads)
    X = X.reshape(-1, num_head, X.shape[1], X.shape[2])
    # (batch_size, sequence_length, num_heads, num_hiddens / num_heads)
    X = X.permute(0, 2, 1, 3)
    # (batch_size, sequence_length, num_hiddens)
    return X.reshape(X.shape[0], X.shape[1], -1)


class MultiHeadAttention(nn.Module):
    def __init__(self, Q_size, K_size, V_size, num_head, num_hiddens, dropout, bias=False, **kwargs) -> None:
        super().__init__(**kwargs)
        self.W_Q = nn.Linear(Q_size, num_hiddens, bias=bias)
        self.W_K = nn.Linear(K_size, num_hiddens, bias=bias)
        self.W_V = nn.Linear(V_size, num_hiddens, bias=bias)
        self.W_O = nn.Linear(num_hiddens, num_hiddens, bias=bias)
        self.attention = DotProductAttention(dropout)
        self.num_head = num_head

    def forward(self, Q, K, V, valid_lens):
        # Q,K,V (batch_size, sequence_length, hidden_num)
        # after (batch_size * num_head, sequence_length, hidden_num / num_head)
        Q = transpose_qkv(self.W_Q(Q), self.num_head)
        K = transpose_qkv(self.W_K(K), self.num_head)
        V = transpose_qkv(self.W_V(V), self.num_head)
        if valid_lens is not None:
            # 在行这一维度copy num_head次
            valid_lens = torch.repeat_interleave(
                valid_lens, repeats=self.num_head, dim=0)
        # (batch_size` * num_heads, sequence_length, num_hiddens / num_heads)
        output = self.attention(Q, K, V, valid_lens)
        output_cat = transpose_output(output, self.num_head)
        return self.W_O(output_cat)


class FFN(nn.Module):
    def __init__(self, ffn_num_input, ffn_num_hidden, ffn_num_output, **kwargs) -> None:
        super().__init__(**kwargs)
        self.layer1 = nn.Linear(ffn_num_input, ffn_num_hidden)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(ffn_num_hidden, ffn_num_output)

    def forward(self, X):
        return self.layer2(self.relu(self.layer1(X)))


class PositionalEncoding(nn.Module):
    def __init__(self, num_hidden, dropout, max_len=1000, **kwargs) -> None:
        super().__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)
        # batchsize, max_len(pos), dimension
        self.P = torch.zeros((1, max_len, num_hidden))
        # (max_len, 1) , (num_hidden / 2)
        # 结果为(max_len, num_hidden / 2)
        X = torch.arange(max_len, dtype=torch.float32).reshape(
            -1, 1) / torch.pow(10000, torch.arange(
                0, num_hidden, 2)/num_hidden)

        # 广播为batchsize, max_len(pos), dimension / 2
        # 其中最后一维是特征维度，特征维度为偶数用sin 维奇数用cos
        self.P[:, :, 0::2] = torch.sin(X)
        self.P[:, :, 1::2] = torch.cos(X)

    def forward(self, X):
        # X.shape[1]是 seq_len
        return self.dropout(X + self.P[:, :X.shape[1], :]).to(X.device)


class AddNorm(nn.Module):
    def __init__(self, normalize_shape, dropout, **kwargs) -> None:
        super().__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(normalize_shape)

    def forward(self, X, Y):
        return self.ln(self.dropout(Y) + X)


class EncoderBlock(nn.Module):
    def __init__(self, Q_size, K_size, V_size, num_hidden, num_head, dropout, bias, norm_shape,
                 ffn_num_input, ffn_num_hidden, ffn_num_output) -> None:
        super().__init__()
        self.attention = MultiHeadAttention(
            Q_size, K_size, V_size, num_head, num_hidden, dropout, bias)
        self.addnorm1 = AddNorm(norm_shape, dropout)
        self.ffn = FFN(ffn_num_input, ffn_num_hidden, ffn_num_output)
        self.addnorm2 = AddNorm(norm_shape, dropout)

    def forward(self, X, valid_lens):
        Y = self.addnorm1(self.attention(X, X, X, valid_lens), X)
        return self.addnorm2(self.ffn(X), X)


X = torch.ones((2, 100, 24))
valid_lens = torch.tensor([3, 2])
encoder_blk = EncoderBlock(
    24, 24, 24, 24, 8,  0.5, True, [100, 24], 24, 48, 24)
encoder_blk.eval()
# print(encoder_blk(X, valid_lens).shape)


class TransformerEncoder(d2l.Encoder):
    def __init__(self, vocab_size, Q_size, K_size, V_size, num_hidden, num_head, dropout, bias, norm_shape,
                 ffn_num_input, ffn_num_hidden, ffn_num_output, num_layers) -> None:
        super().__init__()
        self.num_hiddens = num_hidden
        self.embedding = nn.Embedding(vocab_size, num_hidden)
        self.pos_encoding = PositionalEncoding(num_hidden, dropout)
        self.blks = nn.Sequential()
        for i in range(num_layers):
            self.blks.add_module('block'+str(i),
                                 EncoderBlock(Q_size, K_size, V_size, num_hidden, num_head,
                                 dropout, bias, norm_shape, ffn_num_input, ffn_num_hidden, ffn_num_output))

    def forward(self, X, valid_lens):
        X = self.pos_encoding(self.embedding(X) * math.sqrt(self.num_hiddens))
        for i, blk in enumerate(self.blks):
            X = blk(X, valid_lens)
        return X


encoder = TransformerEncoder(
    200, 24, 24, 24, 24, 8, 0.5, False, [100, 24], 24, 48, 24, 2)
encoder.eval()
valid_lens = torch.tensor([3, 2])
# print(encoder(torch.ones((2, 100), dtype=torch.long), valid_lens).shape)


class DecoderBlock(nn.Module):
    def __init__(self, Q_size, K_size, V_size, num_hiddens, num_head, dropout, bias, norm_shape,
                 ffn_num_input, ffn_num_hidden, ffn_num_output, i, **kwargs) -> None:
        super().__init__(**kwargs)
        self. i = i
        self.attention1 = MultiHeadAttention(
            Q_size, K_size, V_size, num_head, num_hiddens, dropout, bias)
        self.addnorm1 = AddNorm(norm_shape, dropout)
        self.attention2 = MultiHeadAttention(
            Q_size, K_size, V_size, num_head, num_hiddens, dropout, bias)
        self.addnorm2 = AddNorm(norm_shape, dropout)
        self.ffn = FFN(ffn_num_input, ffn_num_hidden, ffn_num_output)
        self.addnorm3 = AddNorm(norm_shape, dropout)

    def forward(self, X, state):
        enc_outputs, enc_valid_lens = state[0], state[1]
        # 如果曾经没有输入
        if state[2][self.i] is None:
            key_values = X
        # 曾经有输入 则把曾经的输入和现在的输入结合在一起，让X拼在seq_len的后面
        else:
            key_values = torch.cat((state[2][self.i], X), axis=1)
        state[2][self.i] = key_values
        if self.training:
            # num_step
            batch_size, num_steps, _ = X.shape
            dec_valid_lens = torch.arange(
                1, num_steps + 1, device=X.device).repeat(batch_size, 1)
        else:
            dec_valid_lens = None
        X2 = self.attention1(X, key_values, key_values, dec_valid_lens)
        Y = self.addnorm1(X, X2)
        Y2 = self.attention2(Y, enc_outputs, enc_outputs, enc_valid_lens)
        Z = self.addnorm2(Y, Y2)
        return self.addnorm3(Z, self.ffn(Z)), state


decoder_blk = DecoderBlock(24, 24, 24, 24, 8, 0.5, False, [
                           100, 24], 24, 48, 24, 0)
decoder_blk.eval()
X = torch.ones((2, 100, 24))
state = [encoder_blk(X, valid_lens), valid_lens, [None]]
# print(decoder_blk(X, state)[0].shape)


class TransformerDecoder(d2l.AttentionDecoder):
    def __init__(self, vocab_size, Q_size, K_size, V_size, num_hiddens, num_head, dropout, bias, norm_shape,
                 ffn_num_input, ffn_num_hidden, ffn_num_output, num_layers, **kwargs) -> None:
        super().__init__(**kwargs)
        self.num_hiddens = num_hiddens
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size, num_hiddens)
        self.pos_encoding = PositionalEncoding(num_hiddens, dropout)
        self.blks = nn.Sequential()
        for i in range(num_layers):
            self.blks.add_module(
                'block'+str(i), DecoderBlock(Q_size, K_size, V_size, num_hiddens, num_head, dropout, bias, norm_shape,
                                             ffn_num_input, ffn_num_hidden, ffn_num_output, i))
        self.dense = nn.Linear(num_hiddens, vocab_size)

    def init_state(self, enc_outputs, enc_valid_lens):
        return [enc_outputs, enc_valid_lens, [None] * self.num_layers]

    def forward(self, X, state):
        X = self.pos_encoding(self.embedding(X) * math.sqrt(self.num_hiddens))
        for i, blk in enumerate(self.blks):
            X, state = blk(X, state)
        return self.dense(X), state


encoder = TransformerEncoder(
    200, 24, 24, 24, 24, 8, 0.5, False, [100, 24], 24, 48, 24, 2)

encoder_outputs = encoder(torch.ones((2, 100), dtype=torch.long), valid_lens)


decoder = TransformerDecoder(200, 24, 24, 24, 24, 8, 0.5, False, [
                             100, 24], 24, 48, 24, 2)
state = decoder.init_state(encoder_outputs, valid_lens)

print(decoder(torch.ones((2, 100), dtype=torch.long), state)[0].shape)
