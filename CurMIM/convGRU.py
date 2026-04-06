import os
import torch
from torch import nn
import torch.nn.functional as F



class ConvGRUCell(nn.Module):
    def __init__(self, input_size, input_dim, hidden_dim, kernel_size, bias, dtype):
        super(ConvGRUCell, self).__init__()
        self.height, self.width = input_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.hidden_dim = hidden_dim
        self.bias = bias
        self.dtype = dtype
        self.input_dim = input_dim
        

        self.conv_gates = nn.Conv2d(in_channels=self.hidden_dim,
                                    out_channels=self.hidden_dim*2,
                                    kernel_size=kernel_size,
                                    padding=self.padding,
                                    bias=self.bias)

        self.conv_can = nn.Conv2d(in_channels=2*self.hidden_dim,
                              out_channels=self.hidden_dim, 
                              kernel_size=kernel_size,
                              padding=self.padding,
                              bias=self.bias)
        
        self.trans_input = nn.Linear(self.input_dim, self.hidden_dim, bias=True)
        self.trans_h = nn.Linear(self.hidden_dim, self.hidden_dim, bias=True)

    def init_hidden(self, batch_size):
        return (torch.zeros(batch_size, self.hidden_dim, self.height, self.width)).type(self.dtype)

    def forward(self, input_tensor, h_cur):
        
        b,c,h,w = input_tensor.shape
        
        input_tensor = input_tensor.view(b,c,h*w).transpose(1,2) 

        h_cur = h_cur.view(b, h_cur.shape[1], h*w).transpose(1,2)

        input_tensor = self.trans_input(input_tensor).view(b,self.hidden_dim,h,w)
        h_cur = self.trans_h(h_cur).view(b, h_cur.shape[-1], h,w)
      
        combined = input_tensor + h_cur

        combined_conv = self.conv_gates(combined)
        gamma, beta = torch.split(combined_conv, self.hidden_dim, dim=1)
        gamma = F.normalize(gamma, p=2, dim=-1)
        beta = F.normalize(beta, p=2, dim=-1)
        reset_gate = gamma*torch.sigmoid(gamma)
        
        update_gate = beta*torch.sigmoid(beta)
        
        combined = torch.cat([input_tensor, reset_gate*h_cur], dim=1)
        cc_cnm = self.conv_can(combined)
        cnm = torch.tanh(cc_cnm)

        h_next = (1 - update_gate) * h_cur + update_gate * cnm
        return h_next


class ConvGRU(nn.Module):
    def __init__(self, input_size, input_dim, hidden_dim, kernel_size, num_layers,
                 dtype, batch_first=False, bias=True, return_all_layers=False):
        super(ConvGRU, self).__init__()
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim  = self._extend_for_multilayer(hidden_dim, num_layers)
        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError('Inconsistent list length.')

        self.height, self.width = input_size
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.dtype = dtype
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers

        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = input_dim if i == 0 else hidden_dim[i - 1]
            cell_list.append(ConvGRUCell(input_size=(self.height, self.width),
                                         input_dim=cur_input_dim,
                                         hidden_dim=self.hidden_dim[i],
                                         kernel_size=self.kernel_size[i],
                                         bias=self.bias,
                                         dtype=self.dtype))

        self.cell_list = nn.ModuleList(cell_list)
        self.transfer = nn.Linear(self.hidden_dim[0], 1, bias=True)


    def forward(self, input_tensor, hidden_state):
        if not self.batch_first:
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)
            
        if hidden_state is None:
            hidden_state = self._init_hidden(batch_size=input_tensor.size(0))

        layer_output_list = []
        last_state_list   = []

        # seq_len = input_tensor.size(1)
        cur_layer_input = input_tensor

        for layer_idx in range(self.num_layers):
            h = hidden_state[layer_idx]
            output_inner = []
                
            h = self.cell_list[layer_idx](input_tensor=cur_layer_input[:, 0, :, :, :], h_cur=h)
            
            h_i = h
            output_inner.append(h)
            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output
            
            
            layer_output_list.append(layer_output)
            last_state_list.append([h])

        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list   = last_state_list[-1:]
        
        layer_output_list = layer_output_list[0]
        last_state_list = last_state_list[0][0]
        b, t, c, h ,w = layer_output_list.size()
        layer_output_list = layer_output_list.view(b, t, c, h*w) 
        layer_output_list = layer_output_list.squeeze(dim=1).transpose(1, 2)
        layer_output_list = self.transfer(layer_output_list)
        layer_output_list = layer_output_list.squeeze(-1)
        layer_output_list = F.normalize(layer_output_list, p=2, dim=-1)
        return layer_output_list, last_state_list, h_i

    def _init_hidden(self, batch_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size))
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or
                    (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param