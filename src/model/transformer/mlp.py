import torch
import torch.nn as nn

class Expert(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Expert, self).__init__()


class Router(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_mixtures):
        super(Router, self).__init__()


class MixtureOfExperts(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_mixtures):
        super(MixtureOfExperts, self).__init__()