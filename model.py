import dgl
import math
import torch as th
import torch.nn as nn
from torch.nn import init
import dgl.function as fn
import dgl.nn.pytorch as dglnn
import torch.nn.functional as F

from utils import get_activation, to_etype_name
from torch.nn.parameter import Parameter

th.set_printoptions(profile="full")


class GCMCGraphConv(nn.Module):

    def __init__(self,
                 in_feats,
                 out_feats,
                 weight=True,
                 device=None,
                 dropout_rate=0.0):
        super(GCMCGraphConv, self).__init__()
        self._in_feats = in_feats  # 909
        self._out_feats = out_feats  # 600
        self.device = device
        self.dropout = nn.Dropout(dropout_rate)

        if weight:
            self.weight = nn.Parameter(th.Tensor(in_feats, out_feats))
        else:
            self.register_parameter('weight', None)
        self.reset_parameters()

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        if self.weight is not None:
            init.xavier_uniform_(self.weight)
        # init.xavier_uniform_(self.att)

    def forward(self, graph, feat, weight=None, Two_Stage=False):
        """Compute graph convolution.

        Normalizer constant :math:`c_{ij}` is stored as two node data "ci"
        and "cj".

        Parameters
        ----------
        graph : DGLGraph
            The graph.
        feat : torch.Tensor
            The input feature
        weight : torch.Tensor, optional
            Optional external weight tensor.
        dropout : torch.nn.Dropout, optional
            Optional external dropout layer.

        Returns
        -------
        torch.Tensor
            The output feature
        """
        with graph.local_scope():
            if isinstance(feat, tuple):
                feat, _ = feat  # dst feature not used [drug or disease num , 3]
            cj = graph.srcdata['cj']
            ci = graph.dstdata['ci']
            if self.device is not None:
                cj = cj.to(self.device)
                ci = ci.to(self.device)
            if weight is not None:
                if self.weight is not None:
                    raise dgl.DGLError('External weight is provided while at the same time the'
                                       ' module has defined its own weight parameter. Please'
                                       ' create the module with flag weight=False.')
            else:
                weight = self.weight

            if weight is not None:
                feat = dot_or_identity(feat, weight, self.device)

            feat = feat * self.dropout(cj)
            graph.srcdata['h'] = feat
            graph.update_all(fn.copy_src(src='h', out='m'),
                             fn.sum(msg='m', out='h'))
            rst = graph.dstdata['h']
            rst = rst * ci

        return rst


class GCMCLayer(nn.Module):
    r"""GCMC layer

    .. math::
        z_j^{(l+1)} = \sigma_{agg}\left[\mathrm{agg}\left(
        \sum_{j\in\mathcal{N}_1}\frac{1}{c_{ij}}W_1h_j, \ldots,
        \sum_{j\in\mathcal{N}_R}\frac{1}{c_{ij}}W_Rh_j
        \right)\right]

    After that, apply an extra output projection:

    .. math::
        h_j^{(l+1)} = \sigma_{out}W_oz_j^{(l+1)}

    The equation is applied to both user nodes and movie nodes and the parameters
    are not shared unless ``share_user_item_param`` is true.

    Parameters
    ----------
    rating_vals : list of int or float
        Possible rating values.
    user_in_units : int
        Size of user input feature
    movie_in_units : int
        Size of movie input feature
    msg_units : int
        Size of message :math:`W_rh_j`
    out_units : int
        Size of of final output user and movie features
    dropout_rate : float, optional
        Dropout rate (Default: 0.0)
    agg : str, optional
        Function to aggregate messages of different ratings.
        Could be any of the supported cross type reducers:
        "sum", "max", "min", "mean", "stack".
        (Default: "stack")
    agg_act : callable, str, optional
        Activation function :math:`sigma_{agg}`. (Default: None)
    out_act : callable, str, optional
        Activation function :math:`sigma_{agg}`. (Default: None)
    share_user_item_param : bool, optional
        If true, user node and movie node share the same set of parameters.
        Require ``user_in_units`` and ``move_in_units`` to be the same.
        (Default: False)
    device: str, optional
        Which device to put data in. Useful in mix_cpu_gpu training and
        multi-gpu training
    """

    def __init__(self, rating_vals,  # [0, 1]
                 user_in_units,  # 909
                 movie_in_units,  # 909
                 msg_units,  # 1800
                 out_units,  # 75
                 dropout_rate=0.0,  # 0.3
                 agg='stack',  # 'sum'
                 agg_act=None,  # Tanh()
                 share_user_item_param=False,  # True
                 basis_units=4, device=None):  # True 4
        super(GCMCLayer, self).__init__()
        self.rating_vals = rating_vals  # [0, 1]
        self.agg = agg  # sum
        self.share_user_item_param = share_user_item_param  # True
        self.ufc = nn.Linear(msg_units, out_units)  # Linear(in_features=1800, out_features=75, bias=True)
        self.user_in_units = user_in_units  # 909
        self.msg_units = msg_units  # 1800
        if share_user_item_param:
            self.ifc = self.ufc
        else:
            self.ifc = nn.Linear(msg_units, out_units)
        if agg == 'stack':
            # divide the original msg unit size by number of rel_values to keep
            # the dimensionality
            assert msg_units % len(rating_vals) == 0
            msg_units = msg_units // len(rating_vals)

        msg_units = msg_units // 3  # 600
        self.msg_units = msg_units  # 600
        self.dropout = nn.Dropout(dropout_rate)
        self.W_r = {}
        subConv = {}
        self.basis_units = basis_units  # 4
        self.att = nn.Parameter(th.randn(len(self.rating_vals), basis_units))  # [2, 4]
        self.basis = nn.Parameter(th.randn(basis_units, user_in_units, msg_units))  # [4, 909, 600]
        for i, rating in enumerate(rating_vals):
            # PyTorch parameter name can't contain "."
            rating = to_etype_name(rating)
            rev_rating = 'rev-%s' % rating
            if share_user_item_param and user_in_units == movie_in_units:
                subConv[rating] = GCMCGraphConv(user_in_units,  # 909
                                                msg_units,  # 840
                                                weight=False,  # False
                                                device=device,
                                                dropout_rate=dropout_rate)
                subConv[rev_rating] = GCMCGraphConv(user_in_units,
                                                    msg_units,
                                                    weight=False,
                                                    device=device,
                                                    dropout_rate=dropout_rate)
            else:
                self.W_r = None
                subConv[rating] = GCMCGraphConv(user_in_units,
                                                msg_units,
                                                weight=True,
                                                device=device,
                                                dropout_rate=dropout_rate)
                subConv[rev_rating] = GCMCGraphConv(movie_in_units,
                                                    msg_units,
                                                    weight=True,
                                                    device=device,
                                                    dropout_rate=dropout_rate)
        self.conv = dglnn.HeteroGraphConv(subConv, aggregate=agg)
        self.agg_act = get_activation(agg_act)
        self.device = device
        self.reset_parameters()

    def partial_to(self, device):
        """Put parameters into device except W_r

        Parameters
        ----------
        device : torch device
            Which device the parameters are put in.
        """
        assert device == self.device
        if device is not None:
            self.ufc.cuda(device)
            if self.share_user_item_param is False:
                self.ifc.cuda(device)
            self.dropout.cuda(device)

    def reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, graph, drug_feat=None, dis_feat=None, Two_Stage=False):
        in_feats = {'drug': drug_feat, 'disease': dis_feat}
        mod_args = {}
        self.W = th.matmul(self.att, self.basis.view(self.basis_units, -1))
        self.W = self.W.view(-1, self.user_in_units, self.msg_units)
        for i, rating in enumerate(self.rating_vals):
            rating = to_etype_name(rating)
            rev_rating = 'rev-%s' % rating

            mod_args[rating] = (self.W[i, :, :] if self.W_r is not None else None, Two_Stage)
            mod_args[rev_rating] = (self.W[i, :, :] if self.W_r is not None else None, Two_Stage)

        out_feats = self.conv(graph, in_feats, mod_args=mod_args)
        drug_feat = out_feats['drug']
        dis_feat = out_feats['disease']

        if in_feats['disease'].shape == dis_feat.shape:
            ufeat = dis_feat.view(dis_feat.shape[0], -1)
            ifeat = drug_feat.view(drug_feat.shape[0], -1)

        drug_feat = self.agg_act(drug_feat)
        drug_feat = self.dropout(drug_feat)

        dis_feat = self.agg_act(dis_feat)
        dis_feat = self.dropout(dis_feat)

        drug_feat = self.ifc(drug_feat)
        dis_feat = self.ufc(dis_feat)

        return drug_feat, dis_feat


class MLPDecoder(nn.Module):
    def __init__(self,
                 in_units,
                 dropout_rate=0.2):
        super(MLPDecoder, self).__init__()
        self.dropout = nn.Dropout(dropout_rate)
        self.sigmoid = nn.Sigmoid()

        self.lin1 = nn.Linear(2 * in_units, 128)
        self.lin2 = nn.Linear(128, 64)
        self.lin3 = nn.Linear(64, 1)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
        self.lin3.reset_parameters()

    def forward(self, graph, drug_feat, dis_feat):
        with graph.local_scope():
            graph.nodes['drug'].data['h'] = drug_feat
            graph.nodes['disease'].data['h'] = dis_feat
            graph.apply_edges(udf_u_mul_e)
            out = graph.edata['m']

            out = F.relu(self.lin1(out))
            out = self.dropout(out)

            out = F.relu(self.lin2(out))
            out = self.dropout(out)

            # out = self.sigmoid(self.lin3(out))
            out = self.lin3(out)
        return out


def udf_u_mul_e_norm(edges):
    return {'reg': edges.src['reg'] * edges.dst['ci']}
    # out_feats = edges.src['reg'].shape[1] // 3 return {'reg' : th.cat([edges.src['reg'][:, :out_feats] * edges.dst[
    # 'ci'], edges.src['reg'][:, out_feats:out_feats*2], edges.src['reg'][:, out_feats*2:]], 1)}


def udf_u_mul_e(edges):
    return {'m': th.cat([edges.src['h'], edges.dst['h']], 1)}
    # return {'m': (edges.src['h']) * (edges.dst['h'])}


def dot_or_identity(A, B, device=None):
    # if A is None, treat as identity matrix. A feat, B weight
    # feat size [313, 3] weight size [909, 600]
    if A is None:
        return B
    elif A.shape[1] == 3:
        if device is None:
            return th.cat([B[A[:, 0].long()], B[A[:, 1].long()], B[A[:, 2].long()]], 1)
        else:
            # return th.cat([B[A[:, 0].long()], B[A[:, 2].long()]], 1).to(device)  # only train one-hop
            # return th.cat([B[A[:, 0].long()], B[A[:, 1].long()]], 1).to(device)  # only train two-hop
            # return B[A[:, 0].long()].to(device)
            return th.cat([B[A[:, 0].long()], B[A[:, 1].long()], B[A[:, 2].long()]], 1).to(device)
    else:
        return A


class SEDR(nn.Module):
    def __init__(self, args):
        super(SEDR, self).__init__()
        self.layers = args.layers
        self._act = get_activation(args.model_activation)
        self.TGCN = nn.ModuleList()
        self.TGCN_llm = nn.ModuleList()
        self.TGCN.append(GCMCLayer(args.rating_vals,  # [0, 1]
                                   args.src_in_units,  # 909
                                   args.dst_in_units,  # 909
                                   args.gcn_agg_units,  # 1800
                                   args.gcn_out_units,  # 75
                                   args.dropout,  # 0.3
                                   args.gcn_agg_accum,  # sum
                                   agg_act=self._act,  # Tanh()
                                   share_user_item_param=args.share_param,  # True
                                   device=args.device))
        self.gcn_agg_accum = args.gcn_agg_accum  # sum
        self.rating_vals = args.rating_vals  # sum[0, 1]
        self.device = args.device
        self.gcn_agg_units = args.gcn_agg_units  # 1800
        self.src_in_units = args.src_in_units  # 909
        for i in range(1, args.layers):
            if args.gcn_agg_accum == 'stack':
                gcn_out_units = args.gcn_out_units * len(args.rating_vals)
            else:
                gcn_out_units = args.gcn_out_units
            self.TGCN.append(GCMCLayer(args.rating_vals,  # [0, 1]
                                       args.gcn_out_units,  # 75
                                       args.gcn_out_units,  # 75
                                       gcn_out_units,  # 75
                                       args.gcn_out_units,  # 75
                                       args.dropout,
                                       args.gcn_agg_accum,
                                       agg_act=self._act,
                                       share_user_item_param=args.share_param,
                                       device=args.device))

        for i in range(args.layers):
            if args.gcn_agg_accum == 'stack':
                gcn_out_units = args.gcn_out_units * len(args.rating_vals)
            else:
                gcn_out_units = args.gcn_out_units
            self.TGCN_llm.append(GCMCLayer(args.rating_vals,  # [0, 1]
                                       args.gcn_out_units,  # 75
                                       args.gcn_out_units,  # 75
                                       gcn_out_units,  # 75
                                       args.gcn_out_units,  # 75
                                       args.dropout,
                                       args.gcn_agg_accum,
                                       agg_act=self._act,
                                       share_user_item_param=args.share_param,
                                       device=args.device))

        self.llm_dim = getattr(args, "llm_dim", None)
        if self.llm_dim is not None and self.llm_dim > 0:
            self.drug_llm_proj = nn.Sequential(
                nn.Linear(self.llm_dim, args.gcn_out_units),
                nn.ReLU(),
                nn.Dropout(args.dropout)
            )
            self.dis_llm_proj = nn.Sequential(
                nn.Linear(self.llm_dim, args.gcn_out_units),
                nn.ReLU(),
                nn.Dropout(args.dropout)
            )
        else:
            self.drug_llm_proj = None
            self.dis_llm_proj = None
        self.decoder = MLPDecoder(in_units=args.gcn_out_units)
        self.rating_vals = args.rating_vals

    def forward(self, dec_graph, drug_feat, dis_feat, subgraphs, Two_Stage=False):

        all_drug_out_subgraphs = []
        all_dis_out_subgraphs = []
        all_drug_out_subgraphs_llm = []
        all_dis_out_subgraphs_llm = []

        for subgraph in subgraphs:
            drug_feat_sub, dis_feat_sub = drug_feat, dis_feat
            drug_out_subgraph, dis_out_subgraph = None, None
            for i in range(0, self.layers):
                drug_o_sub, dis_o_sub = self.TGCN[i](subgraph, drug_feat_sub, dis_feat_sub, Two_Stage)
                if i == 0:
                    drug_out_subgraph = drug_o_sub
                    dis_out_subgraph = dis_o_sub
                else:
                    drug_out_subgraph += drug_o_sub / float(i + 1)
                    dis_out_subgraph += dis_o_sub / float(i + 1)

                drug_feat_sub = drug_o_sub
                dis_feat_sub = dis_o_sub

            all_drug_out_subgraphs.append(drug_out_subgraph)
            all_dis_out_subgraphs.append(dis_out_subgraph)

        drug_out_subgraph_combined = sum(all_drug_out_subgraphs) / len(all_drug_out_subgraphs)
        dis_out_subgraph_combined = sum(all_dis_out_subgraphs) / len(all_dis_out_subgraphs)

        drug_out_combined = drug_out_subgraph_combined
        dis_out_combined = dis_out_subgraph_combined

        if self.drug_llm_proj is not None and hasattr(self, "drug_llm_feat") \
                and self.drug_llm_feat is not None:
            llm_drug = self.drug_llm_proj(self.drug_llm_feat)

        if self.dis_llm_proj is not None and hasattr(self, "dis_llm_feat") \
                and self.dis_llm_feat is not None:
            llm_dis = self.dis_llm_proj(self.dis_llm_feat)

        for subgraph in subgraphs:  # 遍历五个子图
            drug_feat_sub_llm, dis_feat_sub_llm = llm_drug, llm_dis
            drug_out_subgraph_llm, dis_out_subgraph_llm = None, None
            for i in range(0, self.layers):
                drug_o_sub_llm, dis_o_sub_llm = self.TGCN_llm[i](subgraph, drug_feat_sub_llm, dis_feat_sub_llm, Two_Stage)
                if i == 0:
                    drug_out_subgraph_llm = drug_o_sub_llm
                    dis_out_subgraph_llm = dis_o_sub_llm
                else:
                    drug_out_subgraph_llm += drug_o_sub_llm / float(i + 1)
                    dis_out_subgraph_llm += dis_o_sub_llm / float(i + 1)

                drug_feat_sub_llm = drug_o_sub_llm
                dis_feat_sub_llm = dis_o_sub_llm

            all_drug_out_subgraphs_llm.append(drug_out_subgraph_llm)
            all_dis_out_subgraphs_llm.append(dis_out_subgraph_llm)

        drug_out_subgraph_combined_llm = sum(all_drug_out_subgraphs_llm) / len(all_drug_out_subgraphs_llm)
        dis_out_subgraph_combined_llm = sum(all_dis_out_subgraphs_llm) / len(all_dis_out_subgraphs_llm)

        drug_out_combined_llm = drug_out_subgraph_combined_llm
        dis_out_combined_llm = dis_out_subgraph_combined_llm

        drug_out = drug_out_combined + drug_out_combined_llm
        dis_out = dis_out_combined + dis_out_combined_llm

        pred_ratings = self.decoder(dec_graph, drug_out, dis_out)

        return pred_ratings, drug_out_combined, dis_out_combined, drug_out_combined_llm, dis_out_combined_llm
