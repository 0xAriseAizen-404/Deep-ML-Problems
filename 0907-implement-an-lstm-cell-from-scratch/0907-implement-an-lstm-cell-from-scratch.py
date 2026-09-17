import math
import torch
import torch.nn as nn

class LSTMCell(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        k = 1 / math.sqrt(hidden_size)

        self.W_ih = nn.Parameter(torch.empty(4 * hidden_size, input_size))
        self.W_hh = nn.Parameter(torch.empty(4 * hidden_size, hidden_size))
        self.b_ih = nn.Parameter(torch.empty(4 * hidden_size))
        self.b_hh = nn.Parameter(torch.empty(4 * hidden_size))
        with torch.no_grad():
            self.W_ih.uniform_(-k, k)
            self.W_hh.uniform_(-k, k)
            self.b_ih.uniform_(-k, k)
            self.b_hh.uniform_(-k, k)

    def forward(self, x, state):
        h_prev, c_prev = state

        chunks_Wi = self.W_ih.chunk(4, dim=0)
        chunks_Wh = self.W_hh.chunk(4, dim=0)
        Wi = torch.cat((chunks_Wi[0], chunks_Wh[0]), dim=1)
        Wf = torch.cat((chunks_Wi[1], chunks_Wh[1]), dim=1)
        Wc = torch.cat((chunks_Wi[2], chunks_Wh[2]), dim=1)
        Wo = torch.cat((chunks_Wi[3], chunks_Wh[3]), dim=1)

        chunks_bi = self.b_ih.chunk(4, dim=0)
        chunks_bh = self.b_hh.chunk(4, dim=0)
        bi = chunks_bi[0] + chunks_bh[0]
        bf = chunks_bi[1] + chunks_bh[1]
        bc = chunks_bi[2] + chunks_bh[2]
        bo = chunks_bi[3] + chunks_bh[3]

        combined = torch.cat((x, h_prev), dim=1)
        # Input Gate
        IG = torch.sigmoid((combined @ Wi.T) + bi)
        # Forget Gate
        FG = torch.sigmoid((combined @ Wf.T) + bf)
        # Cell Candidate
        C_dash = torch.tanh((combined @ Wc.T) + bc)
        # Cell State Updation
        c_new = c_prev * FG + IG * C_dash
        # Output Gate
        OG = torch.sigmoid((combined @ Wo.T) + bo)
        h_new = OG * torch.tanh(c_new)
        return h_new, c_new

        # h_prev, c_prev = state
        # gates = (x @ self.W_ih.T) + self.b_ih + (h_prev @ self.W_hh.T) + self.b_hh
        # i, f, c_dash, o = gates.chunk(4, dim=1)
        # i = torch.sigmoid(i)
        # f = torch.sigmoid(f)
        # c_dash = torch.tanh(c_dash)
        # c_new = f * c_prev + i * c_dash
        # o = torch.sigmoid(o)
        # h_new = o * torch.tanh(c_new)
        # return h_new, c_new