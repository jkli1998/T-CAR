import torch
import copy
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from maskrcnn_benchmark.modeling.utils import cat
from .utils_motifs import obj_edge_vectors, to_onehot, nms_overlaps, encode_box_info
from maskrcnn_benchmark.modeling.roi_heads.relation_head.model_transformer import MultiHeadAttention, \
    PositionwiseFeedForward, TransformerEncoder


def union_box(boxes1, boxes2, wid, hei):
    # min x1 y1, max x2, y2
    union_b = copy.deepcopy(boxes1)
    mask_min_x1 = boxes1[:, 0] > boxes2[:, 0]
    union_b[:, 0][mask_min_x1] = boxes2[:, 0][mask_min_x1]

    mask_min_y1 = boxes1[:, 1] > boxes2[:, 1]
    union_b[:, 1][mask_min_y1] = boxes2[:, 1][mask_min_y1]

    mask_max_x2 = boxes1[:, 2] < boxes2[:, 2]
    union_b[:, 2][mask_max_x2] = boxes2[:, 2][mask_max_x2]

    mask_max_y2 = boxes1[:, 3] < boxes2[:, 3]
    union_b[:, 3][mask_max_y2] = boxes2[:, 3][mask_max_y2]

    union_wh = union_b[:, 2:] - union_b[:, :2] + 1.0
    union_xy = union_b[:, :2] + 0.5 * union_wh
    w, h = union_wh.split([1, 1], dim=-1)
    x, y = union_xy.split([1, 1], dim=-1)
    x1, y1, x2, y2 = union_b.split([1, 1, 1, 1], dim=-1)

    assert wid * hei != 0
    info = torch.cat([w / wid, h / hei, x / wid, y / hei, x1 / wid, y1 / hei, x2 / wid, y2 / hei,
                      w * h / (wid * hei)], dim=-1).view(-1, 9)
    return info


def relative_box(sbj_boxes, obj_boxes):
    # negative box for no intersection
    inter_s2o = torch.zeros(sbj_boxes.shape, device=sbj_boxes.device)
    inter_o2s = torch.zeros(obj_boxes.shape, device=obj_boxes.device)
    sbj_wh = sbj_boxes[:, 2:] - sbj_boxes[:, :2] + 1.0
    sbj_w, sbj_h = sbj_wh.split([1, 1], dim=-1)
    obj_wh = obj_boxes[:, 2:] - obj_boxes[:, :2] + 1.0
    obj_w, obj_h = obj_wh.split([1, 1], dim=-1)

    obj_w, obj_h, sbj_w, sbj_h = obj_w[:, 0], obj_h[:, 0], sbj_w[:, 0], sbj_h[:, 0]
    inter_s2o[:, 0] = (sbj_boxes[:, 0] - obj_boxes[:, 0]) / obj_w
    inter_s2o[:, 1] = (sbj_boxes[:, 1] - obj_boxes[:, 1]) / obj_h
    inter_s2o[:, 2] = (sbj_boxes[:, 2] - obj_boxes[:, 2]) / obj_w
    inter_s2o[:, 3] = (sbj_boxes[:, 3] - obj_boxes[:, 3]) / obj_h

    inter_o2s[:, 0] = (obj_boxes[:, 0] - sbj_boxes[:, 0]) / sbj_w
    inter_o2s[:, 1] = (obj_boxes[:, 1] - sbj_boxes[:, 1]) / sbj_h
    inter_o2s[:, 2] = (obj_boxes[:, 2] - sbj_boxes[:, 2]) / sbj_w
    inter_o2s[:, 3] = (obj_boxes[:, 3] - sbj_boxes[:, 3]) / sbj_h

    relative_wh = torch.zeros(sbj_boxes.shape, device=sbj_boxes.device)
    relative_wh[:, 0] = torch.log(sbj_w/obj_w)
    relative_wh[:, 1] = torch.log(sbj_h/obj_h)
    relative_wh[:, 2] = torch.log(obj_w/sbj_w)
    relative_wh[:, 3] = torch.log(obj_h/sbj_h)

    info = torch.cat((inter_s2o, inter_o2s, relative_wh), dim=-1)
    return info


def encode_rel_box_info(proposals, rel_pair_idx):
    """
        encode proposed box information sbj (x1, y1, x2, y2), obj (x3, y3, x4, y4) to
        union box (), intersect box (), relative coord (), relative size ().
        (cx/wid, cy/hei, w/wid, h/hei, x1/wid, y1/hei, x2/wid, y2/hei, wh/wid*hei)
    """
    assert proposals[0].mode == 'xyxy'
    boxes_info = []
    for pair_idx, proposal in zip(rel_pair_idx, proposals):
        boxes = proposal.bbox
        img_size = proposal.size
        wid = img_size[0]
        hei = img_size[1]
        sbj_boxes = boxes[pair_idx[:, 0]]
        obj_boxes = boxes[pair_idx[:, 1]]
        union_info = union_box(sbj_boxes, obj_boxes, wid, hei)
        relative_info = relative_box(sbj_boxes, obj_boxes)
        info = torch.cat((union_info, relative_info), dim=-1)
        boxes_info.append(info)
    return torch.cat(boxes_info, dim=0)


class CountFusion(nn.Module):
    def __init__(self, dim_x, dim_y, output_dim):
        super(CountFusion, self).__init__()
        self.dense_x = nn.Linear(dim_x, output_dim)
        self.dense_y = nn.Linear(dim_y, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x, y):
        x1 = self.dense_x(x)
        y1 = self.dense_y(y)
        item1 = self.relu(x1 + y1)
        item2 = (x1 - y1) * (x1 - y1)
        return item1 - item2


class FusionPosTransRelContext(nn.Module):
    def __init__(self, config, obj_classes, rel_classes, in_channels):
        super().__init__()
        self.cfg = config
        # setting parameters
        if self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_BOX:
            self.mode = 'predcls' if self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL else 'sgcls'
        else:
            self.mode = 'sgdet'
        self.obj_classes = obj_classes
        self.rel_classes = rel_classes
        self.num_obj_cls = len(obj_classes)
        self.num_rel_cls = len(rel_classes)
        self.in_channels = in_channels
        self.obj_dim = in_channels
        self.embed_dim = self.cfg.MODEL.ROI_RELATION_HEAD.EMBED_DIM
        self.nms_thresh = self.cfg.TEST.RELATION.LATER_NMS_PREDICTION_THRES

        self.dropout_rate = self.cfg.MODEL.ROI_RELATION_HEAD.TRANSFORMER.DROPOUT_RATE
        self.obj_layer = self.cfg.MODEL.ROI_RELATION_HEAD.TRANSFORMER.OBJ_LAYER
        self.edge_layer = self.cfg.MODEL.ROI_RELATION_HEAD.TRANSFORMER.REL_LAYER
        self.num_head = self.cfg.MODEL.ROI_RELATION_HEAD.TRANSFORMER.NUM_HEAD
        self.inner_dim = self.cfg.MODEL.ROI_RELATION_HEAD.TRANSFORMER.INNER_DIM
        self.hidden_dim = self.cfg.MODEL.ROI_RELATION_HEAD.CONTEXT_HIDDEN_DIM
        self.k_dim = self.cfg.MODEL.ROI_RELATION_HEAD.TRANSFORMER.KEY_DIM
        self.v_dim = self.cfg.MODEL.ROI_RELATION_HEAD.TRANSFORMER.VAL_DIM

        # the following word embedding layer should be initalize by glove.6B before using
        embed_vecs = obj_edge_vectors(self.obj_classes, wv_dir=self.cfg.GLOVE_DIR, wv_dim=self.embed_dim)
        self.obj_embed1 = nn.Embedding(self.num_obj_cls, self.embed_dim)
        self.obj_embed2 = nn.Embedding(self.num_obj_cls, self.embed_dim)
        with torch.no_grad():
            self.obj_embed1.weight.copy_(embed_vecs, non_blocking=True)
            self.obj_embed2.weight.copy_(embed_vecs, non_blocking=True)

        # position embedding
        self.bbox_embed = nn.Sequential(*[
            nn.Linear(9, 32), nn.LeakyReLU(inplace=True, negative_slope=0.2), nn.Dropout(0.1),
            nn.Linear(32, 128), nn.LeakyReLU(inplace=True, negative_slope=0.2), nn.Dropout(0.1),
        ])

        self.rel_bbox = nn.Sequential(*[
            nn.Linear(21, 64), nn.LeakyReLU(inplace=True, negative_slope=0.2), nn.Dropout(0.1),
            nn.Linear(64, 128), nn.LeakyReLU(inplace=True, negative_slope=0.2), nn.Dropout(0.1),
        ])

        input_obj, input_edge = self.in_channels, self.in_channels + self.hidden_dim
        input_obj += 128

        self.fusion1 = CountFusion(self.hidden_dim, self.hidden_dim, self.hidden_dim)
        self.fusion2 = CountFusion(self.hidden_dim, self.hidden_dim, self.hidden_dim)

        self.fuse_pos_union = nn.Linear(128 + 4096, self.hidden_dim)
        self.lin_obj_visual = nn.Linear(input_obj, self.hidden_dim)
        self.lin_edge_visual1 = nn.Linear(input_edge, 2 * self.hidden_dim)

        self.relu = nn.ReLU()

        self.out_obj = nn.Linear(self.hidden_dim, self.num_obj_cls)
        self.context_obj = TransformerEncoder(self.obj_layer, self.num_head, self.k_dim, self.v_dim,
                                              self.hidden_dim, self.inner_dim, self.dropout_rate)
        self.context_edge = TransformerEncoder(self.edge_layer, self.num_head, self.k_dim, self.v_dim,
                                               self.hidden_dim, self.inner_dim, self.dropout_rate)

    def forward(self, roi_features, proposals, rel_pair_idxs, union_feats, logger=None):
        # labels will be used in DecoderRNN during training
        use_gt_label = self.training or self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL
        obj_labels = cat([proposal.get_field("labels") for proposal in proposals], dim=0) if use_gt_label else None

        # label/logits embedding will be used as input
        if self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL:
            obj_labels = obj_labels.long()
            obj_embed = self.obj_embed1(obj_labels)
        else:
            obj_logits = cat([proposal.get_field("predict_logits") for proposal in proposals], dim=0).detach()
            obj_embed = F.softmax(obj_logits, dim=1) @ self.obj_embed1.weight

        # bbox embedding will be used as input
        assert proposals[0].mode == 'xyxy'
        pos_embed = self.bbox_embed(encode_box_info(proposals))

        # encode objects with transformer

        num_objs = [len(p) for p in proposals]

        packed_feats = [roi_features, pos_embed]

        if len(packed_feats) == 1:
            obj_pre_rep = packed_feats[0]
        else:
            obj_pre_rep = cat(packed_feats, -1)

        obj_pre_rep_vis = self.lin_obj_visual(obj_pre_rep)
        obj_feats = self.context_obj(obj_pre_rep_vis, num_objs)

        # predict obj_dists and obj_preds
        if self.mode == 'predcls':
            obj_preds = obj_labels
            obj_dists = to_onehot(obj_preds, self.num_obj_cls)

            packed_feats = [roi_features, obj_feats]
            obj_refine_vis = cat(packed_feats, dim=-1)
        else:
            obj_dists = self.out_obj(obj_feats)
            use_decoder_nms = self.mode == 'sgdet' and not self.training
            if use_decoder_nms:
                boxes_per_cls = [proposal.get_field('boxes_per_cls') for proposal in proposals]
                obj_preds = self.nms_per_cls(obj_dists, boxes_per_cls, num_objs)
            else:
                obj_preds = obj_dists[:, 1:].max(1)[1] + 1

            packed_feats = [roi_features, obj_feats]
            obj_refine_vis = cat(packed_feats, dim=-1)

        num_edges = [pairs.shape[0] for pairs in rel_pair_idxs]

        obj_refine_vis = self.lin_edge_visual1(obj_refine_vis)
        obj_refine_vis = obj_refine_vis.view(obj_refine_vis.size(0), 2, self.hidden_dim)
        obj_refine_vis = self.relu(obj_refine_vis)
        # edge context
        edge_pos_rep = self.rel_bbox(encode_rel_box_info(proposals, rel_pair_idxs))
        sbj_reps, obj_reps = self.compose_edge_rep(obj_refine_vis, rel_pair_idxs, num_objs)

        pos_union_rep = cat((union_feats, edge_pos_rep), -1)
        
        pos_union_rep = self.fuse_pos_union(pos_union_rep)
        pos_union_rep = self.relu(pos_union_rep)

        fusion1 = self.fusion1(sbj_reps, obj_reps)
        edge_vis_rep = self.fusion2(fusion1, pos_union_rep)
        edge_ctx = self.context_edge(edge_vis_rep, num_edges)
        return obj_dists, obj_preds, edge_ctx

    def compose_edge_rep(self, obj_repr_for_edge, rel_pair_idxs, num_objs):
        head_rep = obj_repr_for_edge[:, 0].contiguous().view(-1, self.hidden_dim)
        tail_rep = obj_repr_for_edge[:, 1].contiguous().view(-1, self.hidden_dim)
        head_reps = head_rep.split(num_objs, dim=0)
        tail_reps = tail_rep.split(num_objs, dim=0)

        sbj_reps, obj_reps = [], []
        for pair_idx, head_rep, tail_rep in zip(rel_pair_idxs, head_reps, tail_reps):
            sbj_reps.append(head_rep[pair_idx[:, 0]])
            obj_reps.append(tail_rep[pair_idx[:, 1]])
        sbj_reps = cat(sbj_reps, dim=0)
        obj_reps = cat(obj_reps, dim=0)
        return sbj_reps, obj_reps

    def nms_per_cls(self, obj_dists, boxes_per_cls, num_objs):
        obj_dists = obj_dists.split(num_objs, dim=0)
        obj_preds = []
        for i in range(len(num_objs)):
            is_overlap = nms_overlaps(boxes_per_cls[i]).cpu().numpy() >= self.nms_thresh # (#box, #box, #class)

            out_dists_sampled = F.softmax(obj_dists[i], -1).cpu().numpy()
            out_dists_sampled[:, 0] = -1

            out_label = obj_dists[i].new(num_objs[i]).fill_(0)

            for i in range(num_objs[i]):
                box_ind, cls_ind = np.unravel_index(out_dists_sampled.argmax(), out_dists_sampled.shape)
                out_label[int(box_ind)] = int(cls_ind)
                out_dists_sampled[is_overlap[box_ind,:,cls_ind], cls_ind] = 0.0
                out_dists_sampled[box_ind] = -1.0 # This way we won't re-sample

            obj_preds.append(out_label.long())
        obj_preds = torch.cat(obj_preds, dim=0)
        return obj_preds


