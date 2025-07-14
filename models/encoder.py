import torch
import torch.nn as nn

class BaseEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers, dropout=0.2, bidirectional=False):
        super(BaseEncoder, self).__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
            bidirectional=bidirectional
        )
        
        self.output_proj = nn.Linear(
            2 * hidden_size if bidirectional else hidden_size,
            output_size, bias=True
        )

        # self.input_bn = nn.BatchNorm1d(input_size)

    def forward(self, inputs, input_lengths):
        assert inputs.dim() == 3  # [B, T, F]

        # print("inputs", inputs.shape)
        B, T, F = inputs.shape
        # inputs = self.input_bn(inputs.view(-1, F)).view(B, T, F)

        if input_lengths is not None:
            sorted_seq_lengths, indices = torch.sort(input_lengths, descending=True)
            inputs_sorted = inputs[indices]


            packed_inputs = nn.utils.rnn.pack_padded_sequence(
                inputs_sorted, sorted_seq_lengths.cpu(), batch_first=True, enforce_sorted=True
            )

        else:
            packed_inputs = inputs

        self.lstm.flatten_parameters()

        # print("packed_inputs", packed_inputs.data.shape)

        outputs, hidden = self.lstm(packed_inputs)


        if input_lengths is not None:
            unpacked_outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)


            _, desorted_indices = torch.sort(indices)
            outputs = unpacked_outputs[desorted_indices]

        else:
            outputs = outputs


        logits = self.output_proj(outputs)


        return logits, hidden




def build_encoder(config):
    if config["enc"]["type"] == 'lstm':
        return BaseEncoder(
            input_size=config["feature_dim"],
            hidden_size=config["enc"]["hidden_size"],
            output_size=config["enc"]["output_size"],
            n_layers=config["enc"]["n_layers"],
            dropout=config["dropout"],
            bidirectional=config["enc"]["bidirectional"]
        )
    else:
        raise NotImplementedError("Encoder type not implemented.")