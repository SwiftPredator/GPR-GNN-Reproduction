import torch
from torch.nn import Dropout, ReLU, Linear, LogSoftmax, ELU
import numpy as np
from torch.nn import Parameter
from torch_geometric.nn.conv.gcn_conv import gcn_norm

from torch.nn import Sequential as torchSequential

from torch_geometric.nn import (
    GATConv,
    GCNConv,
    ChebConv,
    MessagePassing,
    APPNP,
    JumpingKnowledge,
    Sequential,
    SGConv,
    MLP,
)


class GPR_prop(MessagePassing):
    """
    propagation class for GPR_GNN
    """

    def __init__(self, K, alpha, Init, Gamma=None, bias=True, **kwargs):
        super(GPR_prop, self).__init__(aggr="add", **kwargs)
        self.K = K
        self.Init = Init
        self.alpha = alpha

        assert Init in ["SGC", "PPR", "NPPR", "Random", "WS"]
        if Init == "SGC":
            # SGC-like, note that in this case, alpha has to be a integer. It means where the peak at when initializing GPR weights.
            TEMP = 0.0 * np.ones(K + 1)
            TEMP[alpha] = 1.0
        elif Init == "PPR":
            # PPR-like
            TEMP = alpha * (1 - alpha) ** np.arange(K + 1)
            TEMP[-1] = (1 - alpha) ** K
        elif Init == "NPPR":
            # Negative PPR
            TEMP = (alpha) ** np.arange(K + 1)
            TEMP = TEMP / np.sum(np.abs(TEMP))
        elif Init == "Random":
            # Random
            bound = np.sqrt(3 / (K + 1))
            TEMP = np.random.uniform(-bound, bound, K + 1)
            TEMP = TEMP / np.sum(np.abs(TEMP))
        elif Init == "WS":
            # Specify Gamma
            TEMP = Gamma

        self.temp = Parameter(torch.tensor(TEMP))

    def reset_parameters(self):
        torch.nn.init.zeros_(self.temp)
        for k in range(self.K + 1):
            self.temp.data[k] = self.alpha * (1 - self.alpha) ** k
        self.temp.data[-1] = (1 - self.alpha) ** self.K

    def forward(self, x, edge_index, edge_weight=None):
        edge_index, norm = gcn_norm(
            edge_index, edge_weight, num_nodes=x.size(0), dtype=x.dtype
        )

        hidden = x * (self.temp[0])
        for k in range(self.K):
            x = self.propagate(edge_index, x=x, norm=norm)
            gamma = self.temp[k + 1]
            hidden = hidden + gamma * x
        return hidden

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def __repr__(self):
        return "{}(K={}, temp={})".format(self.__class__.__name__, self.K, self.temp)


class GPRGNN(torch.nn.Module):
    def __init__(self, dataset, args):
        super(GPRGNN, self).__init__()

        if args.ppnp == "PPNP":
            self.prop1 = APPNP(args.K, args.alpha)
        elif args.ppnp == "GPR_prop":
            self.prop1 = GPR_prop(args.K, args.alpha, args.Init, args.Gamma)

        self.Init = args.Init
        self.dprate = args.dprate
        self.dropout = args.dropout

        self.model = Sequential(
            "x",
            [
                (Dropout(p=self.dropout), "x -> x"),
                Linear(dataset.num_features, args.hidden),
                ReLU(inplace=True),
                Dropout(p=self.dropout),
                Linear(args.hidden, dataset.num_classes),
            ],
        )

    def forward(self, data):
        x = self.model(data.x)
        if self.dprate == 0.0:
            x = self.prop1(x, data.edge_index)
            return LogSoftmax(dim=1)(x)
        else:
            x = Dropout(p=self.dprate)(x)
            x = self.prop1(x, data.edge_index)
            return LogSoftmax(dim=1)(x)

    def get_params(self):
        return [
            self.model._modules["module_1"].parameters(),
            self.model._modules["module_4"].parameters(),
            self.prop1.parameters(),
        ]


class GCN_Net(torch.nn.Module):
    def __init__(self, dataset, args):
        super(GCN_Net, self).__init__()

        self.model = Sequential(
            "x, edge_index",
            [
                (GCNConv(dataset.num_features, args.hidden), "x, edge_index -> x"),
                ReLU(inplace=True),
                Dropout(p=args.dropout),
                (GCNConv(args.hidden, dataset.num_classes), "x, edge_index -> x"),
                LogSoftmax(dim=1),
            ],
        )

    def forward(self, data):
        return self.model(data.x, data.edge_index)


class ChebNet(torch.nn.Module):
    def __init__(self, dataset, args):
        super(ChebNet, self).__init__()

        in_dim = dataset.num_features
        out_dim = dataset.num_classes

        self.model = Sequential(
            "x, edge_index",
            [
                (ChebConv(in_dim, 32, K=2), "x, edge_index -> x"),
                ReLU(inplace=True),
                (Dropout(p=args.dropout), "x -> x"),
                (ChebConv(32, out_dim, K=2), "x, edge_index -> x"),
                LogSoftmax(dim=1),
            ],
        )

    def forward(self, data):
        return self.model(data.x, data.edge_index)


class GAT_Net(torch.nn.Module):
    def __init__(self, dataset, args):
        super(GAT_Net, self).__init__()

        self.model = Sequential(
            "x, edge_index",
            [
                (Dropout(p=args.dropout), "x -> x"),
                (
                    GATConv(
                        dataset.num_features,
                        args.hidden,
                        heads=args.heads,
                        dropout=args.dropout,
                    ),
                    "x, edge_index -> x",
                ),
                ELU(inplace=True),
                (Dropout(p=args.dropout)),
                (
                    GATConv(
                        args.hidden * args.heads,
                        args.hidden,
                        heads=args.output_heads,
                        concat=False,
                        dropout=args.dropout,
                    ),
                    "x, edge_index -> x",
                ),
                LogSoftmax(dim=1),
            ],
        )

    def forward(self, data):
        return self.model(data.x, data.edge_index)


class APPNP_Net(torch.nn.Module):
    def __init__(self, dataset, args):
        super(APPNP_Net, self).__init__()

        self.model = Sequential(
            "x, edge_index",
            [
                (Dropout(p=args.dropout), "x -> x"),
                Linear(dataset.num_features, args.hidden),
                ReLU(inplace=True),
                Dropout(p=args.dropout),
                Linear(args.hidden, dataset.num_classes),
                (APPNP(args.K, args.alpha), "x, edge_index -> x"),
                LogSoftmax(dim=1),
            ],
        )

    def forward(self, data):
        return self.model(data.x, data.edge_index)

    def get_params(self):
        return [
            self.model._modules["module_1"].parameters(),
            self.model._modules["module_4"].parameters(),
            self.model._modules["module_5"].parameters(),
        ]


class GCN_JKNet(torch.nn.Module):
    def __init__(self, dataset, args):
        super(GCN_JKNet, self).__init__()

        in_dim = dataset.num_features
        out_dim = dataset.num_classes

        self.model = Sequential(
            "x, edge_index",
            [
                (GCNConv(in_dim, 16), "x, edge_index -> x1"),
                ReLU(inplace=True),
                (Dropout(p=0.5), "x1 -> x1"),
                (GCNConv(16, 16), "x1, edge_index -> x2"),
                ReLU(inplace=True),
                (Dropout(p=0.5), "x2 -> x2"),
                (lambda x1, x2: [x1, x2], "x1, x2 -> xs"),
                (JumpingKnowledge("lstm", 16, num_layers=4), "xs -> x"),
                (APPNP(K=1, alpha=0), "x, edge_index -> x"),
                (Linear(16, out_dim), "x -> x"),
                (LogSoftmax(dim=1)),
            ],
        )

    def forward(self, data):
        return self.model(data.x, data.edge_index)


class MLPNet(torch.nn.Module):
    def __init__(self, dataset, args):
        super(MLPNet, self).__init__()

        in_dim = dataset.num_features
        out_dim = dataset.num_classes

        self.model = torchSequential(
            Dropout(p=args.dropout),
            Linear(dataset.num_features, args.hidden),
            ReLU(inplace=True),
            Dropout(p=args.dropout),
            Linear(args.hidden, dataset.num_classes),
            LogSoftmax(dim=1),
        )

        # self.model = MLP(channel_list=[in_dim, 64, out_dim], batch_norm=False)

    def forward(self, data):
        x = self.model(data.x)
        return LogSoftmax(dim=1)(x)


class SGCNet(torch.nn.Module):
    def __init__(self, dataset, args):
        super(SGCNet, self).__init__()
        self.model = Sequential(
            "x, edge_index",
            [
                (
                    SGConv(dataset.num_features, dataset.num_classes, K=2, cached=True),
                    "x, edge_index -> x",
                ),
                LogSoftmax(dim=1),
            ],
        )  # K from paper

    def forward(self, data):
        return self.model(data.x, data.edge_index)


class GraphSAGE(torch.nn.Module):
    """
    Cant be reproduced since no information about sampling given 😡
    """

    def __init__(self, dataset, args):
        super(GraphSAGE, self).__init__()


class geomGCN(torch.nn.Module):
    def __init__(self, dataset, args):
        super(geomGCN, self).__init__()
        # 🥲
