import torch
import torch.nn as nn
import torch.nn.functional as F
from .encoder import build_encoder
from .decoder import build_decoder
from .loss import RNNTLoss
from .cnn_encoder import build_cnn_encoder


class JointNet(nn.Module):
    def __init__(self, input_size, inner_dim, vocab_size):
        super(JointNet, self).__init__()
        self.forward_layer = nn.Linear(input_size, inner_dim, bias=True)
        self.tanh = nn.Tanh()
        self.project_layer = nn.Linear(inner_dim, vocab_size, bias=True)

    def forward(self, enc_state, dec_state):
        if enc_state.dim() == 3 and dec_state.dim() == 3:
            dec_state = dec_state.unsqueeze(1)
            enc_state = enc_state.unsqueeze(2)

            t = enc_state.size(1)
            u = dec_state.size(2)

            enc_state = enc_state.repeat([1, 1, u, 1])
            dec_state = dec_state.repeat([1, t, 1, 1])
        else:
            assert enc_state.dim() == dec_state.dim()

        concat_state = torch.cat((enc_state, dec_state), dim=-1)
        outputs = self.forward_layer(concat_state)
        outputs = self.tanh(outputs)
        outputs = self.project_layer(outputs)

        return outputs


class Transducer(nn.Module):
    def __init__(self, config):
        super(Transducer, self).__init__()
        self.config = config

        #################################
        # Build CNN encoder
        self.cnn_encoder = build_cnn_encoder(config["cnn_encoder"])
        #################################

        # Build LSTM encoder & decoder
        self.encoder = build_encoder(config)
        self.decoder = build_decoder(config)

        # Build joint network
        self.joint = JointNet(
            input_size=config["joint"]["input_size"],
            inner_dim=config["joint"]["inner_size"],
            vocab_size=config["vocab_size"]
        )

        # Optionally share embedding weights
        if config.get("share_embedding", False):
            assert self.decoder.embedding.weight.size() == self.joint.project_layer.weight.size(), \
                f"{self.decoder.embedding.weight.size(1)} != {self.joint.project_layer.weight.size(1)}"
            self.joint.project_layer.weight = self.decoder.embedding.weight

        # Loss function
        self.crit = RNNTLoss(
            blank=config.get("blank", 0),
            reduction=config.get("reduction", "mean")
        )

    def forward(self, inputs, inputs_length, targets, targets_length):
        zero = torch.zeros((targets.shape[0], 1)).long()
        if targets.is_cuda: zero = zero.cuda()
        
        targets_add_blank = torch.cat((targets, zero), dim=1)
        #################################
        # Process through CNN encoder first
        cnn_output = self.cnn_encoder(inputs)
        #################################
        # Then through LSTM encoder
        enc_state, _ = self.encoder(cnn_output, inputs_length)
        
        dec_state, _ = self.decoder(targets_add_blank, (targets_length+1).cpu())

        # Joint network
        logits = self.joint(enc_state, dec_state)

        return logits

    def recognize(self, inputs, inputs_length):
        batch_size = inputs.size(0)
        #################################
        # Process through CNN encoder first
        cnn_output = self.cnn_encoder(inputs)
        #################################
        # Then through LSTM encoder
        enc_states, _ = self.encoder(cnn_output, inputs_length)

        zero_token = torch.LongTensor([[1]])
        if inputs.is_cuda:
            zero_token = zero_token.cuda()

        def decode(enc_state, lengths):
            token_list = []
            dec_state, hidden = self.decoder(zero_token)

            for t in range(lengths):
                logits = self.joint(enc_state[t].view(-1), dec_state.view(-1))
                out = F.softmax(logits, dim=0).detach()
                pred = torch.argmax(out, dim=0).item()

                if pred == 2: 
                    break

                if pred not in (0,1,2,4):
                    token_list.append(pred)
                    token = torch.LongTensor([[pred]])
                    if enc_state.is_cuda:
                        token = token.cuda()
                    dec_state, hidden = self.decoder(token, hidden=hidden)

            return token_list

        results = [decode(enc_states[i], inputs_length[i]) for i in range(batch_size)]
        return results
