import math
import torch
from torch import nn
from torch.nn import Module, Parameter
import torch.nn.functional as F

class GNN(Module):
    def __init__(self, hidden_size, step=1,num_heads=2):
        super(GNN, self).__init__()
        self.step = step
        self.hidden_size = hidden_size
        self.input_size = hidden_size * 2
        self.gate_size = 3 * hidden_size
        self.w_ih = Parameter(torch.Tensor(self.gate_size, self.input_size))
        self.w_hh = Parameter(torch.Tensor(self.gate_size, self.hidden_size))
        self.b_ih = Parameter(torch.Tensor(self.gate_size))
        self.b_hh = Parameter(torch.Tensor(self.gate_size))
        self.b_iah_a = Parameter(torch.Tensor(self.hidden_size))
        self.b_oah_a = Parameter(torch.Tensor(self.hidden_size))
        self.b_iah_b = Parameter(torch.Tensor(self.hidden_size))
        self.b_oah_b = Parameter(torch.Tensor(self.hidden_size))

        self.linear_edge_in_a = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_edge_out_a = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_edge_in_b = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_edge_out_b = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_edge_f = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        embed_dim = hidden_size

        self.multihead_attn_i = nn.MultiheadAttention(embed_dim, num_heads)
        self.multihead_attn_h = nn.MultiheadAttention(embed_dim, num_heads)

    def GNNCell(self, A, hidden):
        input_in_a = torch.matmul(A[:, :, :A.shape[1]], self.linear_edge_in_a(hidden)) + self.b_iah_a
        input_out_a = torch.matmul(A[:, :, A.shape[1]: 2 * A.shape[1]], self.linear_edge_out_a(hidden)) + self.b_oah_a
        inputs = torch.cat([input_in_a, input_out_a], 2)
        gi = F.linear(inputs, self.w_ih, self.b_ih)
        gh = F.linear(hidden, self.w_hh, self.b_hh)
        q_i,k_i,v_i = gi.chunk(3, 2)
        q_h, k_h, v_h = gh.chunk(3, 2)
        attn_output_i, attn_output_weights_i = self.multihead_attn_i(q_i, k_i, v_i)
        attn_output_h, attn_output_weights_h = self.multihead_attn_h(q_h, k_h, v_h)
        hy = attn_output_i + attn_output_h
        return hy

    def forward(self, A, hidden):
        for i in range(self.step):
            hidden = self.GNNCell(A, hidden)
        return hidden


class SessionGraph(Module):
    def __init__(self, opt, n_node):
        super(SessionGraph, self).__init__()
        #opt
        self.opt = opt
        self.hidden_size = opt.hiddenSize
        self.n_node = n_node
        self.nonhybrid = opt.nonhybrid
        self.embedding = nn.Embedding(self.n_node, self.hidden_size)
        self.gnn = GNN(self.hidden_size, step=opt.step,num_heads=opt.num_heads)
        self.linear_one = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_two = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_three = nn.Linear(self.hidden_size, 1, bias=False)
        self.linear_transform = nn.Linear(self.hidden_size * 2, self.hidden_size, bias=True)
        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=opt.lr, weight_decay=opt.l2)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=opt.lr_dc_step, gamma=opt.lr_dc)
        self.A_in = None
        self.A_out = None
        self.A_fusion = None
        self.delta = None
        self.z_si_local_output = None
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def L1_projection(self, x, lambda_):
        # component-wise projection onto the l∞ ball of radius λ1.
        return torch.clamp(x, min=-lambda_, max=lambda_)

    def L21_projection(self, x, lambda_):
        # row-wise projection on the l2 ball of radius λ1.
        row_norm = torch.norm(x, p=2, dim=1)
        scale = torch.clamp(row_norm, max=lambda_)
        index = row_norm > 0
        scale[index] = scale[index] / row_norm[index]  # avoid to be devided by 0
        return scale.unsqueeze(1) * x

    #local
    def local_smoothing(self, hidden, mask):
        ht = hidden[torch.arange(mask.shape[0]).long(), torch.sum(mask, 1) - 1]  # batch_size x latent_size
        q1 = self.linear_one(ht).view(ht.shape[0], 1, ht.shape[1])  # batch_size x 1 x latent_size
        q2 = self.linear_two(hidden)  # batch_size x seq_length x latent_size
        alpha = self.linear_three(torch.sigmoid(q1 + q2))
        a = torch.sum(alpha * hidden * mask.view(mask.shape[0], -1, 1).float(), 1)
        self.z_si_local_output = hidden.clone()

        col_value = torch.sum(mask, 1)
        k_col_value = col_value - self.opt.k
        mask_value = 0
        rows = torch.arange(mask.size(0)).unsqueeze(1).cuda()
        cols = torch.arange(mask.size(1)).unsqueeze(0).cuda()
        k_mask = cols < k_col_value.unsqueeze(1)
        mask[k_mask] = mask_value

        torch_sum_items = alpha * hidden * mask.view(mask.shape[0], -1, 1).float()
        z_si_local = torch.sum(torch_sum_items, 1)

        z_si_local_l1 = self.L1_projection(z_si_local, 3)
        z_si_local_l21 = self.L21_projection(z_si_local, 3)
        self.z_si_local_output[:, -1, :] = hidden[:, -1] + z_si_local_l21
        #hidden.shape: torch.Size([90, 145, 100])
        print("hidden.shape:", hidden.shape)
        #hidden[:, -1].shape: torch.Size([90, 100])
        print("hidden[:, -1].shape:",hidden[:, -1].shape)
        #self.z_si_local_output.shape: torch.Size([90, 145, 100])
        print("self.z_si_local_output.shape:", self.z_si_local_output.shape)
        print("z_si_local_l21.shape:", z_si_local_l21.shape)
        #z_si_local_l21.shape: torch.Size([90, 100])
        z_si_local = z_si_local_l21
        return z_si_local

    #global_smoothing
    def global_smoothing(self):
        A_fusion = F.pad(self.A_fusion, (0, self.z_si_local_output.shape[1] - self.A_fusion.shape[1],
                                         0, self.z_si_local_output.shape[1] - self.A_fusion.shape[1]), mode='constant',
                         value=0)
        delta = F.pad(self.delta, (0, self.z_si_local_output.shape[1] - self.delta.shape[1],
                                   0, self.z_si_local_output.shape[1] - self.delta.shape[1]), mode='constant', value=0)

        o_t = torch.zeros(self.z_si_local_output.shape[0], self.z_si_local_output.shape[1], self.z_si_local_output.shape[2]).cuda()
        h_0 = self.z_si_local_output.clone()
        h_t = self.z_si_local_output.clone()
        M = self.opt.M
        for i in range(M):
            y = 0.25 * h_0 + (1 - 0.25) * (A_fusion @ h_t)
            h_tb = y - 0.25 * delta @ o_t
            o_t = o_t - 2.0 * (delta @ h_tb)
            h_t = y - 0.25 * (delta @ self.L21_projection(o_t, 3))

        z_si_global = torch.mean(h_t, dim=1)
        return z_si_global

    def forward(self, inputs, A, alias_inputs, mask, delta):
        self.A_in = A[:, :, :A.shape[1]]
        self.A_out = A[:, :, A.shape[1]: 2 * A.shape[1]]
        self.A_fusion = self.A_in + self.A_out
        self.delta = delta
        #Embedding
        hidden = self.embedding(inputs)
        #GNN
        hidden = self.gnn(A, hidden)
        get = lambda i: hidden[i][alias_inputs[i]]
        seq_hidden = torch.stack([get(i) for i in torch.arange(len(alias_inputs)).long()])

        #local
        z_si_local = self.local_smoothing(seq_hidden, mask)
        #global
        z_si_global = self.global_smoothing()
        #local and global
        z_si_lg = self.linear_transform(torch.cat([z_si_local, z_si_global], 1))
        z_si_lg_scores = torch.matmul(z_si_lg, self.embedding.weight[1:].transpose(1, 0))
        return z_si_lg_scores
