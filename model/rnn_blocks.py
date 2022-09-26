import json
import torch
import torch.nn as nn
import torch.nn.functional as F


class BaselineLSTMEncoder(nn.Module):
    def __init__(self, params={}) -> None:
        super().__init__()
        assert 'encoder_vocab_size' in params
        assert 'start_token' in params
        assert 'end_token' in params
        encoder_vocab_size = params['encoder_vocab_size']
        
        word_vec_dim = 128
        if 'word_vec_dim' in params:
            word_vec_dim = params['word_vec_dim']
        hidden_size = 256
        if 'hidden_size' in params:
            hidden_size = params['hidden_size']
        n_layers = 2
        if 'n_layers' in params:
            n_layers = params['n_layers']
        input_dropout = 0
        if 'input_dropout' in params:
            input_dropout = params['input_dropout']
        dropout = 0
        if 'dropout' in params:
            dropout = params['dropout']
        bidirectional = True
        if 'bidirectional' in params:
            bidirectional = params['bidirectional']
        
        self.embedding = nn.Embedding(encoder_vocab_size, word_vec_dim)
        self.rnn = nn.LSTM(
            word_vec_dim,
            hidden_size,
            n_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout
        )
        if input_dropout > 0:
            self.input_dropout = nn.Dropout(p=input_dropout)
        else:
            self.input_dropout = None
        self.end_token = params['end_token']

    def forward(self, input_seq, variable_len_inp=True):
        embedded = self.embedding(input_seq)
        if self.input_dropout:
            embedded = self.input_dropout(embedded)
        if variable_len_inp:
            input_lengths = self._get_input_lengths(input_seq, self.end_token)
            embedded = nn.utils.rnn.pack_padded_sequence(
                embedded,
                input_lengths,
                batch_first=True,
                enforce_sorted=False
            )
        output, hidden = self.rnn(embedded)
        if variable_len_inp:
            output, _ = nn.utils.rnn.pad_packed_sequence(
                output,
                batch_first=True
            )
        return output, hidden
    
    def _get_input_lengths(self, inp, token_val):
        # We have BxN
        match_matrix = (inp == token_val)
        if torch.any(match_matrix.sum(dim=1) != 1):
            print('Incorrect end token')
        indices = match_matrix.nonzero()[:, 1].cpu()
        lengths = indices + 1
        return lengths


class BaselineLSTMDecoder(nn.Module):
    def __init__(self, params={}) -> None:
        super().__init__()
        assert 'decoder_vocab_size' in params
        assert 'start_token' in params
        assert 'end_token' in params
        decoder_vocab_size = params['decoder_vocab_size']

        self.max_len = 15

        word_vec_dim = 128
        if 'word_vec_dim' in params:
            word_vec_dim = params['word_vec_dim']
        hidden_size = 256
        if 'hidden_size' in params:
            hidden_size = params['hidden_size']
        n_layers = 2
        if 'n_layers' in params:
            n_layers = params['n_layers']
        input_dropout = 0
        if 'input_dropout' in params:
            input_dropout = params['input_dropout']
        dropout = 0
        if 'dropout' in params:
            dropout = params['dropout']
        bidirectional = True
        if 'bidirectional' in params:
            bidirectional = params['bidirectional']

        if bidirectional:
            hidden_size = 2 * hidden_size

        self.bidirectional_encoder = bidirectional

        self.embedding = nn.Embedding(decoder_vocab_size, word_vec_dim)
        self.rnn = nn.LSTM(
            word_vec_dim,
            hidden_size,
            n_layers,
            batch_first=True,
            dropout=dropout
        )
        self.out_linear = nn.Linear(hidden_size, decoder_vocab_size)

        if input_dropout > 0:
            self.input_dropout = nn.Dropout(p=input_dropout)
        else:
            self.input_dropout = None

        self.attention = DecoderAttention(hidden_size)

    def forward(self, y, encoder_outputs, encoder_hidden):
        decoder_hidden = self._init_state(encoder_hidden)
        decoder_outputs, decoder_hidden, attn_map = self.forward_step(
            y, decoder_hidden, encoder_outputs
        )
        return decoder_outputs, decoder_hidden

    def forward_step(self, inp, hidden, enc_outs):
        embedded = self.embedding(inp)
        if self.input_dropout:
            embedded = self.input_dropout(embedded)

        decoder_outputs, decoder_hidden = self.rnn(embedded, hidden)

        output, attn = self.attention(decoder_outputs, enc_outs)
        output = self.out_linear(output)

        return output, decoder_hidden, attn

    # From seq2seq model:
    def _init_state(self, encoder_hidden):
        """ Initialize the encoder hidden state. """
        if encoder_hidden is None:
            return None
        if isinstance(encoder_hidden, tuple):
            encoder_hidden = tuple([self._cat_directions(h) for h in encoder_hidden])
        else:
            encoder_hidden = self._cat_directions(encoder_hidden)
        return encoder_hidden

    # From seq2seq model:
    def _cat_directions(self, h):
        """ If the encoder is bidirectional, do the following transformation.
            (#directions * #layers, #batch, hidden_size) -> (#layers, #batch, #directions * hidden_size)
        """
        if self.bidirectional_encoder:
            h = torch.cat([h[0:h.size(0):2], h[1:h.size(0):2]], 2)
        return h

    def sample(self, start_tok, enc_outs, enc_hidden, end_tok):
        decoder_hidden = self._init_state(enc_hidden)
        output_symbols = []
        output_lengths = torch.tensor([self.max_len] * start_tok.shape[0])
        decoder_input = start_tok

        def decode(i, out):
            symbols = out.topk(1)[1]
            output_symbols.append(symbols.squeeze())
            eos = symbols.data.eq(end_tok)
            if eos.dim() > 0:
                eos = eos.cpu().view(-1).numpy()
                # Mask places where end symbol appeared and output_length
                # was never updated for length update
                update_idx = ((output_lengths > i) & eos) != 0
                # Address by bools and update with current prog length
                output_lengths[update_idx] = len(output_symbols)
            symbols = symbols.squeeze(2)
            # print(symbols.shape)
            return symbols

        for i in range(self.max_len):
            decoder_outputs, decoder_hidden, attn_map = self.forward_step(
                decoder_input, decoder_hidden, enc_outs
            )
            decoder_input = decode(i, decoder_outputs)

        if len(output_symbols[0].shape) < 1:
            output_symbols = torch.stack(output_symbols)
        else:
            output_symbols = torch.stack(output_symbols, dim=1)
        return output_symbols


class DecoderAttention(nn.Module):
    def __init__(self, hidden_size, weighted=False):
        super(DecoderAttention, self).__init__()
        self.weighted = weighted
        self.hidden_size = hidden_size
        if self.weighted:
            self.attn_weight = nn.Linear(hidden_size, hidden_size, bias=False)
        self.linear_output = nn.Linear(2 * hidden_size, hidden_size)

    def forward(self, decoder_out, encoder_out):
        # decoder_out BxOxH
        # encoder_out BxIxH
        B, _, H = decoder_out.shape
        assert H == self.hidden_size, "Size mismatch"

        output = decoder_out

        if self.weighted:
            output = self.attn_weight(output)

        attn = torch.bmm(output, encoder_out.transpose(1, 2))   # OxH x HxI
        attn = F.softmax(attn, dim=2)

        context = torch.bmm(attn, encoder_out)
        combined = torch.cat((context, output), dim=2)
        output = torch.tanh(
            self.linear_output(combined)
        )

        return output, attn