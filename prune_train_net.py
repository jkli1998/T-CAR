from maskrcnn_benchmark.utils.env import setup_environment  # noqa F401 isort:skip

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import tqdm
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
import copy
from sklearn import metrics

from maskrcnn_benchmark.data.datasets.visual_genome import load_graphs, load_info
from maskrcnn_benchmark.modeling.roi_heads.relation_head.utils_motifs import obj_edge_vectors, rel_edge_vectors


class CountFusion(nn.Module):
    def __init__(self, dim_x, dim_y, output_dim=512):
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

class RelationPrune(nn.Module):
    def __init__(self, obj_embed_vecs, rel_embed_vecs, num_objs=151, num_rels=51, embed_dim=200, hidden_dim=512):
        super(RelationPrune, self).__init__()
        self.num_obj_cls = num_objs
        self.num_rel_cls = num_rels
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.sbj_embed = nn.Embedding(self.num_obj_cls, self.embed_dim)
        self.obj_embed = nn.Embedding(self.num_obj_cls, self.embed_dim)
        self.rel_embed = nn.Embedding(self.num_rel_cls, self.embed_dim)
        with torch.no_grad():
            self.sbj_embed.weight.copy_(obj_embed_vecs, non_blocking=True)
            self.obj_embed.weight.copy_(obj_embed_vecs, non_blocking=True)
            self.rel_embed.weight.copy_(rel_embed_vecs, non_blocking=True)
        
        self.cnt_fusion_so = CountFusion(embed_dim, embed_dim, output_dim=hidden_dim)
        self.cnt_fusion_sr = CountFusion(embed_dim, embed_dim, output_dim=hidden_dim)
        self.cnt_fusion_or = CountFusion(embed_dim, embed_dim, output_dim=hidden_dim)
        self.dense_s = nn.Linear(embed_dim, hidden_dim)
        self.dense_o = nn.Linear(embed_dim, hidden_dim)
        self.dense_r = nn.Linear(embed_dim, hidden_dim)
        self.project = nn.Sequential(
            nn.Linear(3 * hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, sbj_token, obj_token, rel_token):

        sbj_embed = self.sbj_embed(sbj_token)
        obj_embed = self.sbj_embed(obj_token)
        rel_embed = self.rel_embed(rel_token)
        fused_so = self.cnt_fusion_so(sbj_embed, obj_embed)
        fused_sr = self.cnt_fusion_sr(sbj_embed, rel_embed)
        fused_or = self.cnt_fusion_or(obj_embed, rel_embed)

        proj_s = self.dense_s(sbj_embed)
        proj_o = self.dense_o(obj_embed)
        proj_r = self.dense_r(rel_embed)

        fused_so, fused_sr, fused_or = torch.sigmoid(fused_so), torch.sigmoid(fused_sr), torch.sigmoid(fused_or)
        proj_r, proj_o, proj_s = torch.sigmoid(proj_r), torch.sigmoid(proj_o), torch.sigmoid(proj_s)

        act_sor = fused_so * proj_r
        act_sro = fused_sr * proj_o
        act_ors = fused_or * proj_s

        concat = torch.cat((act_sor, act_sro, act_ors), dim=-1)
        logit = torch.sigmoid(self.project(concat))
        return logit

class RelationData(Dataset):
    def __init__(self, seen_triplets, num_objs=151, num_rels=51):
        obj_idx_list = np.arange(1, num_objs)
        rel_idx_list = np.arange(1, num_rels)

        sbj_dim = np.repeat(obj_idx_list, num_objs - 1)
        un_sqz_obj_idx_list = obj_idx_list.reshape(-1, 1)
        un_sqz_rel_idx_list = rel_idx_list.reshape(-1, 1)
        obj_dim = np.repeat(un_sqz_obj_idx_list, num_objs - 1, axis=1).T.reshape(-1)
        sbj_obj = np.stack((sbj_dim, obj_dim), axis=0)

        so_dim = np.repeat(sbj_obj, num_rels - 1, axis=1)
        rel_dim = np.repeat(un_sqz_rel_idx_list, sbj_obj.shape[1], axis=1).T.reshape(1, -1)
        sor_list = np.concatenate((so_dim, rel_dim), axis=0).T
        self.compose_space = torch.tensor(sor_list, dtype=torch.long)
        self.labels = self.gen_label(seen_triplets)

    def gen_label(self, seen_triplets):
        labels = []
        for i in tqdm.tqdm(range(self.compose_space.shape[0])):
            item = self.compose_space[i, :]
            tpt = (int(item[0]), int(item[1]), int(item[2]))
            if tpt in seen_triplets:
                labels.append(1)
            else:
                labels.append(0)
        labels = torch.tensor(labels, dtype=torch.float)
        return labels

    def __getitem__(self, index):
        item = self.compose_space[index, :]
        y = self.labels[index]
        return item[0], item[1], item[2], y

    def __len__(self):
        return self.compose_space.shape[0]

def run_test(model, dataloader, ep):
    model.eval()
    y_preds, y_labels = [], []
    matrix = torch.zeros((151, 151, 51))
    for token_s, token_o, token_r, y_batch in dataloader:
        token_s, token_o, token_r = token_s.cuda(), token_o.cuda(), token_r.cuda()
        with torch.no_grad():
            y_pred = model(token_s, token_o, token_r)
        y_preds.append(y_pred)
        y_labels.append(y_batch)

        token_s, token_o, token_r = token_s.cpu(), token_o.cpu(), token_r.cpu()
        y_pred = y_pred.cpu()
        matrix[token_s, token_o, token_r] = y_pred.reshape(-1)

    y_preds = torch.cat(y_preds, dim=0).detach().cpu().numpy()
    y_labels = torch.cat(y_labels, dim=0).numpy()
    y_preds_prob = copy.deepcopy(y_preds)
    y_preds_hard = copy.deepcopy(y_preds)
    y_preds_hard[y_preds > 0.5] = 1
    y_preds_hard[y_preds < 0.5] = 0
    y_preds_hard = y_preds_hard.astype(int)
    recall = metrics.recall_score(y_labels, y_preds_hard)
    precision = metrics.precision_score(y_labels, y_preds_hard)
    fpr, tpr, thresholds = metrics.roc_curve(y_labels, y_preds_prob)
    auc = metrics.auc(fpr, tpr)
    print("ep: {}, recall: {:.3f}, precision: {:.3f}, auc: {:.3f}".format(ep, recall, precision, auc))
    model.train()
    return matrix, auc


def train(model, dataloader, test_loader, epoch, lr, save_dir, clip_num=5, pi=0.0):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    os.makedirs(save_dir, exist_ok=True)
    criterion = nn.BCELoss()
    model = model.cuda()
    best_auc = 0.0
    for ep in range(epoch):
        for step, (token_s, token_o, token_r, y_batch) in enumerate(dataloader):
            opt.zero_grad()
            token_s, token_o, token_r = token_s.cuda(), token_o.cuda(), token_r.cuda()
            y_batch = y_batch.cuda()
            y_pred = model(token_s, token_o, token_r)
            pos_mask = y_batch == 1
            neg_mask = y_batch == 0
            pf_label = y_batch[pos_mask]
            pf_label[pf_label == 1] = 0
            if torch.sum(pos_mask) != 0:
                pt_loss = criterion(y_pred[pos_mask].reshape(-1), y_batch[pos_mask])
                pf_loss = criterion(y_pred[pos_mask].reshape(-1), pf_label)
            else:
                pt_loss = pf_loss = 0
            if torch.sum(neg_mask) != 0:
                ng_loss = criterion(y_pred[neg_mask].reshape(-1), y_batch[neg_mask])
            else:
                ng_loss = 0
            item1 = pi * pt_loss
            item2 = ng_loss - pi * pf_loss
            item2[item2 < 0] = 0
            loss = item1 + item2
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_num)
            opt.step()
            if step % 30 == 0:
                print("Ep: {}, step: {}, loss: {:.5f}".format(ep, step, loss.item()))
        mat, auc = run_test(model, test_loader, ep)
        if auc > best_auc:
            best_auc = auc
            torch.save(mat, os.path.join(save_dir, "prune_mat.pkl"))
            torch.save(model.state_dict(), os.path.join(save_dir, "prune_model.pth"))

# utils
def convert_obj_class(obj_classes, rel):
    for index, (i_gt_class, i_relationships) in enumerate(zip(obj_classes, rel)):
        for index_rel in range(len(i_relationships)):
            i_relationships[index_rel, 0] = i_gt_class[i_relationships[index_rel, 0]]
            i_relationships[index_rel, 1] = i_gt_class[i_relationships[index_rel, 1]]
        rel[index] = i_relationships
    return rel


def main():
    path = "/irip/lijiankai_2020/dataset" # vg data path
    zs_fp = "./zeroshot_triplet_new.pytorch"
    roidb_file = os.path.join(path, "vg/VG-SGG-with-attri.h5")
    dict_file = os.path.join(path, "vg/VG-SGG-dicts-with-attri.json")
    glove_dir = os.path.join(path, "glove")
    embed_dim = 200
    batch_size = 8192
    epoch = 20
    save_dir = "./prune_ckpt/"
    zeroshot_triplet = torch.load(zs_fp).long().numpy()
    print("load info......")
    ind_to_classes, ind_to_predicates, ind_to_attributes = load_info(dict_file)
    print("load train......")
    split_mask, gt_boxes, gt_classes, gt_attributes, relationships = load_graphs(
        roidb_file, "train", num_im=-1, num_val_im=5000,
        filter_empty_rels=True, filter_non_overlap=False, zs_file=zs_fp
    )
    print("load test......")
    _, _, test_class, _, test_relations = load_graphs(
        roidb_file, "test", num_im=-1, num_val_im=5000,
        filter_empty_rels=True, filter_non_overlap=False
    )

    seen_relationships = convert_obj_class(gt_classes, relationships)
    test_relations = convert_obj_class(test_class, test_relations)

    seen_triplets = np.concatenate(np.array(seen_relationships), axis=0)
    seen_set = set()
    unseen_set = set()
    for i in range(len(seen_triplets)):
        item = seen_triplets[i]
        seen_set.add((item[0], item[1], item[2]))

    for i in range(len(zeroshot_triplet)):
        item = zeroshot_triplet[i]
        unseen_set.add((item[0], item[1], item[2]))

    obj_embed_vecs = obj_edge_vectors(ind_to_classes, wv_dir=glove_dir, wv_dim=embed_dim)
    rel_embed_vecs = rel_edge_vectors(ind_to_predicates, wv_dir=glove_dir, wv_dim=embed_dim)
    print("load dataset......")
    dataset = RelationData(seen_set)
    test_dataset = RelationData(unseen_set)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size//64, shuffle=False)
    model = RelationPrune(obj_embed_vecs, rel_embed_vecs)
    train(model, dataloader, test_loader, epoch, lr=0.001, save_dir=save_dir, pi=0.03)

if __name__ == "__main__":
    main()
