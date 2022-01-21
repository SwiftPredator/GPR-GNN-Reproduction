import torch
from torch.nn import Dropout, ReLU, Linear, LogSoftmax, ELU

from torch_geometric.nn import (
    GATConv,
    GCNConv,
    ChebConv,
    MessagePassing,
    APPNP,
    JumpingKnowledge,
    Sequential,
)

# model = Sequential('x, edge_index', [
#     (GCNConv(in_channels, 64), 'x, edge_index -> x'),
#     ReLU(inplace=True),
#     (GCNConv(64, 64), 'x, edge_index -> x'),
#     ReLU(inplace=True),
#     Linear(64, out_channels),
# ])

# model = Sequential('x, edge_index, batch', [
#     (Dropout(p=0.5), 'x -> x'),
#     (GCNConv(dataset.num_features, 64), 'x, edge_index -> x1'),
#     ReLU(inplace=True),
#     (GCNConv(64, 64), 'x1, edge_index -> x2'),
#     ReLU(inplace=True),
#     (lambda x1, x2: [x1, x2], 'x1, x2 -> xs'),
#     (JumpingKnowledge("cat", 64, num_layers=2), 'xs -> x'),
#     (global_mean_pool, 'x, batch -> x'),
#     Linear(2 * 64, dataset.num_classes),
# ])


class GPR_prop(MessagePassing):
    ...


class GPRGNN(torch.nn.Module):
    def __init__(self, dataset, args):
        super(GPRGNN, self).__init__()
        self.dprate = args.dprate

        if args.ppnp == "PPNP":
            self.prop1 = APPNP(args.K, args.alpha)
        elif args.ppnp == "GPR_prop":
            self.prop1 = GPR_prop(args.K, args.alpha, args.Init, args.Gamma)

        self.model = Sequential(
            "x, edge_index",
            [
                (Dropout(p=self.dropout, training=self.training)),
                Linear(dataset.num_features, args.hidden),
                ReLU(inplace=True),
                Dropout(p=self.dropout, training=self.training),
                Linear(args.hidden, dataset.num_classes),
            ],
        )

        def forward(self, data):
            x = self.model(data.x, data.edge_index)
            if self.dprate == 0.0:
                x = self.prop1(data.x, data.edge_index)
                return LogSoftmax(x, dim=1)
            else:
                x = Dropout(x, p=self.dprate, training=self.training)
                x = self.prop1(data.x, data.edge_index)
                return LogSoftmax(x, dim=1)


class GCN_Net(torch.nn.Module):
    def __init__(self, dataset, args):
        super(GCN_Net, self).__init__()

        self.model = Sequential(
            "x, edge_index",
            [
                (GCNConv(dataset.num_features, args.hidden)),
                (ReLU(inplace=True)),
                (Dropout(p=args.dropout, training=self.training)),
                (GCNConv(args.hidden, dataset.num_classes)),
                (LogSoftmax(dim=1)),
            ],
        )

    def forward(self, data):
        return self.model(data.x, data.edge_index)


class ChebNet(torch.nn.Module):
    def __init__(self, dataset, args):
        super(ChebNet, self).__init__()

        in_dim = dataset.num_features
        out_dim = dataset.num_output

        self.model = Sequential(
            "x, edge_index",
            [
                (ChebConv(in_dim, 32, K=2), "x -> x"),
                ReLU(inplace=True),
                (Dropout(p=args.dropout, training=self.training), "x -> x")(
                    ChebConv(32, out_dim, K=2), "x -> x"
                ),
                (LogSoftmax(dim=1)),
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
                (Dropout(p=args.dropout, training=self.training)),
                (
                    GATConv(
                        dataset.num_features,
                        args.hidden,
                        heads=args.heads,
                        dropout=args.dropout,
                    )
                ),
                ELU(inplace=True),
                (Dropout(p=args.dropout, training=self.training)),
                GATConv(
                    dataset.num_features,
                    args.hidden,
                    heads=args.heads,
                    dropout=args.dropout,
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
                (Dropout(p=args.dropout, training=self.training)),
                (Linear(dataset.num_features, args.hidden)),
                (ReLU(inplace=True)),
                (Dropout(p=args.dropout, training=self.training)),
                (Linear(args.hidden, dataset.num_classes)),
                (APPNP(args.K, args.alpha)),
                (LogSoftmax(dim=1)),
            ],
        )

    def forward(self, data):
        return self.model(data.x, data.edge_index)


class GCN_JKNet(torch.nn.Module):
    def __init__(self, dataset, args):
        in_dim = dataset.num_features
        out_dim = dataset.num_output

        super(GCN_JKNet, self).__init__()
        self.model = Sequential(
            "x, edge_index",
            [
                (GCNConv(in_dim, 16), "x, edge_index -> x1"),
                ReLU(inplace=True),
                (Dropout(p=0.5, training=self.training), "x1 -> x1"),
                (GCNConv(16, 16), "x1, edge_index -> x2"),
                ReLU(inplace=True),
                (Dropout(p=0.5, training=self.training), "x2 -> x2"),
                (lambda x1, x2: [x1, x2], "x1, x2 -> xs"),
                (JumpingKnowledge("lstm", 16, num_layers=4), "xs -> x"),
                (APPNP(K=1, alpha=0), "x, edge_index -> x"),
                (Linear(16, out_dim), "x -> x"),
                (LogSoftmax(dim=1)),
            ],
        )

    def forward(self, data):
        return self.model(data.x, data.edge_index)

