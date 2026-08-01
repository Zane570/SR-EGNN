import numpy as np
import torch
from torch_sparse import SparseTensor
import torch.nn.functional as F

class Data(object):
    def __init__(self, batch_size, inputs_ori, targets):
        # Batch size for division
        self.batch_size = batch_size
        # Original input data
        self.inputs_ori = inputs_ori
        # Length of the data
        self.data_len = len(self.inputs_ori)
        # Target indices
        self.targets = targets
        # Batch training input, padded with zeros
        self.inputs_arr_fillz = None
        # Batch training, padded with mask (1 for valid values)
        self.inputs_arr_mask = None
        # Maximum length for batch padding
        self.inputs_len_max = None
        # Divided indices
        self.all_batch_index = None

        ## Execute padding function
        self.data_fillz_masks()
        ## Execute index division function
        self.get_train_index()

    # Get batch training indices
    def get_train_index(self):
        # Length of dataset
        data_len = self.data_len
        batch_size = self.batch_size
        n_batch = int(self.data_len / self.batch_size)
        if self.data_len % batch_size != 0:
            n_batch += 1
        # Ensure proper division
        slices = np.split(np.arange(n_batch * batch_size), n_batch)
        # Ensure all elements are included
        slices[-1] = slices[-1][:(data_len - batch_size * (n_batch - 1))]
        self.all_batch_index = slices

    def get_inc(self, edge_index):
        # compute the incident matrix
        size = edge_index.sizes()[1]

        row_index = edge_index.storage.row()
        col_index = edge_index.storage.col()

        # What happens if duplicate edges are not deleted
        mask = row_index >= col_index  # remove duplicate edge and self loop
        row_index = row_index[mask]
        col_index = col_index[mask]

        # .numel() Returns the number of elements in the array
        edge_num = row_index.numel()
        row = torch.cat([torch.arange(edge_num), torch.arange(edge_num)]).cuda()
        col = torch.cat([row_index, col_index]).cuda()
        value = torch.cat([torch.ones(edge_num), -1 * torch.ones(edge_num)]).cuda()
        # print("size",size)
        # print("edge_num",edge_num)
        # print("size",size)
        inc = SparseTensor(row=row, rowptr=None, col=col, value=value,
                           sparse_sizes=(edge_num, size))
        # Modify
        # row = torch.cat([torch.arange(edge_num)]).cuda()
        # col = torch.cat([row_index])
        # value = torch.cat([torch.ones(edge_num)]).cuda()
        # inc = SparseTensor(row=row, rowptr=None, col=col, value=value,
        #                 sparse_sizes=(edge_num, size))
        # print("inc.to_dense().shape",inc.to_dense().shape)
        # print("inc.to_dense()",inc.to_dense())
        return inc.to_dense()

    # Align vector dimensions for each batch
    def data_fillz_masks(self):
        inputs_len_arr = [len(tinputs) for tinputs in self.inputs_ori]
        inputs_len_max = max(inputs_len_arr)
        # Pad with zeros
        inputs_arr_fillz = [tinputs + [0] * (inputs_len_max - le) for tinputs, le in zip(self.inputs_ori, inputs_len_arr)]
        # inputs_mask: 1 for valid values
        inputs_arr_mask = [[1] * le + [0] * (inputs_len_max - le) for le in inputs_len_arr]
        self.inputs_arr_fillz = np.asarray(inputs_arr_fillz)
        self.inputs_arr_mask = np.asarray(inputs_arr_mask)
        self.inputs_len_max = np.asarray(inputs_len_max)
        # return np.asarray(inputs_arr_fillz), np.asarray(inputs_arr_mask), np.asarray(inputs_len_max)

    # Get model batch inputs, items, etc.
    def get_model_input_func(self,batch_index_i):
        # Test samples
        batch_inputs, batch_mask, batch_targets = self.inputs_arr_fillz[batch_index_i], self.inputs_arr_mask[batch_index_i], self.targets[batch_index_i]

        model_inputs, n_node, A, alias_inputs,delta = [], [], [], [],[]
        # Get maximum length of n_node
        for u_input in batch_inputs:
            n_node.append(len(np.unique(u_input)))
        max_n_node = np.max(n_node)
        # Process each batch_inputs, [[37151 37149 37149     0     0     0  ]]
        for u_input in batch_inputs:
            node = np.unique(u_input)
            ## Process element values
            # node_list = node.tolist()
            # if 0 in node_list:
            #     node_list.remove(0)
            #     model_inputs.append(node_list + (max_n_node - len(node) + 1) * [0])
            # else:
            #     model_inputs.append(node.tolist() + (max_n_node - len(node)) * [0])

            model_inputs.append(node.tolist() + (max_n_node - len(node)) * [0])

            u_A = np.zeros((max_n_node, max_n_node))
            for i in np.arange(len(u_input) - 1):
                if u_input[i + 1] == 0:
                    break
                u = np.where(node == u_input[i])[0][0]
                v = np.where(node == u_input[i + 1])[0][0]
                u_A[u][v] = 1
            u_sum_in = np.sum(u_A, 0)
            u_sum_in[np.where(u_sum_in == 0)] = 1
            u_A_in = np.divide(u_A, u_sum_in)
            u_sum_out = np.sum(u_A, 1)
            u_sum_out[np.where(u_sum_out == 0)] = 1
            u_A_out = np.divide(u_A.transpose(), u_sum_out)
            u_A = np.concatenate([u_A_in, u_A_out]).transpose()

            #calculate delta
            u_fusion = u_A_in + u_A_out
            u_fusion_t = torch.Tensor(u_fusion).to_sparse()
            rows = u_fusion_t.indices()[0]
            cols = u_fusion_t.indices()[0]
            values = u_fusion_t.values()
            edge_index = SparseTensor(row=rows, rowptr=None, col=cols, value=values)
            temp_delta = self.get_inc(edge_index)
            target_shape = (u_A_in.shape[0], u_A_in.shape[0])
            pad_height = target_shape[0] - temp_delta.shape[0]
            pad_width = target_shape[1] - temp_delta.shape[1]
            padded_temp_delta = F.pad(temp_delta, (0, pad_width, 0, pad_height), "constant", 0)
            delta.append(padded_temp_delta.tolist())
            A.append(u_A)
            alias_inputs.append([np.where(node == i)[0][0] for i in u_input])
        return alias_inputs, A, model_inputs, batch_mask, batch_targets,delta
