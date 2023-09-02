# importing the required packages and modules
from neo4j import GraphDatabase
import itertools
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.data
import dgl.nn
from dgl.nn import SAGEConv
import dgl.function as fn
import sklearn
from sklearn.metrics import roc_auc_score

# created class to perform node recommendation based on link prediction
class NodeRecommendationModel(nn.Module):
    def __init__(self, in_feats, h_feats):
        super(NodeRecommendationModel, self).__init__()
        self.conv1 = SAGEConv(in_feats, h_feats, 'mean')
        self.conv2 = SAGEConv(h_feats, h_feats, 'mean')
        self.fc = nn.Linear(h_feats, in_feats)

    def forward(self, g, in_feat):
        h = self.conv1(g, in_feat)
        h = F.relu(h)
        h = self.conv2(g, h)
        h = self.fc(h)
        return h

# created class to perform dot product between node features and uses that for prediction
class DotPredictor(nn.Module):
    def forward(self, g, h):
        with g.local_scope():
            g.ndata['h'] = h
            # Compute a new edge feature named 'score' by a dot-product between the
            # source node feature 'h' and destination node feature 'h'.
            g.apply_edges(fn.u_dot_v('h', 'h', 'score'))
            # u_dot_v returns a 1-element vector for each edge so you need to squeeze it.
            return g.edata['score'][:, 0]


# setting up neo4j graph database connection
uri = "bolt://127.0.0.1:7687"
username = "neo4j"
password = "Sunny1758#"

driver = GraphDatabase.driver(uri, auth=(username, password), encrypted=False)

features_dict = {}

# this function allows us to load the nodes, edges, features of each node using cypher commands
def load_neo4j_data(driver):
    with driver.session() as session:
        query = "MATCH (n) RETURN id(n) AS node_id"
        result = session.run(query)
        nodes = [record['node_id'] for record in result]

        query = "MATCH (n)-[r]->(m) RETURN id(n) AS src, id(m) AS dst"
        result = session.run(query)
        edges = [(record['src'], record['dst']) for record in result]

        query = (
            "MATCH (n) RETURN id(n) AS node_id, "
            "n.Feature1 as Feature1, n.Feature2 as Feature2, "
            "n.Feature3 as Feature3, n.Feature4 as Feature4, "
            "n.Feature5 as Feature5, n.Feature6 as Feature6, "
            "n.Feature7 as Feature7, n.Feature8 as Feature8, "
            "n.Feature9 as Feature9, n.Feature10 as Feature10, "
            "n.Feature11 as Feature11, n.Feature12 as Feature12, "
            "n.Feature13 as Feature13, n.Feature14 as Feature14, "
            "n.Feature15 as Feature15, n.Feature16 as Feature16, "
            "n.Feature17 as Feature17, n.Feature18 as Feature18, "
            "n.Feature19 as Feature19, n.Feature20 as Feature20, "
            "n.Feature21 as Feature21, n.Feature22 as Feature22, "
            "n.Feature23 as Feature23, n.Feature24 as Feature24, "
            "n.Feature25 as Feature25, n.Feature26 as Feature26, "
            "n.Feature27 as Feature27, n.Feature28 as Feature28, "
            "n.Feature29 as Feature29, n.Feature30 as Feature30, "
            "n.Feature31 as Feature31, n.Feature32 as Feature32, "
            "n.Feature33 as Feature33, n.Feature34 as Feature34, "
            "n.Feature35 as Feature35, n.Feature36 as Feature36, "
            "n.Feature37 as Feature37, n.Feature38 as Feature38, "
            "n.Feature39 as Feature39, n.Feature40 as Feature40, "
            "n.Feature41 as Feature41, n.Feature42 as Feature42, "
            "n.Feature43 as Feature43, n.Feature44 as Feature44, "
            "n.Feature45 as Feature45, n.Feature46 as Feature46, "
            "n.Feature47 as Feature47, n.Feature48 as Feature48, "
            "n.Feature49 as Feature49, n.Feature50 as Feature50, "
            "n.Feature51 as Feature51, n.Feature52 as Feature52, "
            "n.Feature53 as Feature53, n.Feature54 as Feature54, "
            "n.Feature55 as Feature55, n.Feature56 as Feature56, "
            "n.Feature57 as Feature57, n.Feature58 as Feature58, "
            "n.Feature59 as Feature59, n.Feature60 as Feature60, "
            "n.Feature61 as Feature61, n.Feature62 as Feature62, "
            "n.Feature63 as Feature63, n.Feature64 as Feature64, "
            "n.Feature65 as Feature65, n.Feature66 as Feature66, "
            "n.Feature67 as Feature67, n.Feature68 as Feature68, "
            "n.Feature69 as Feature69, n.Feature70 as Feature70, "
            "n.Feature71 as Feature71, n.Feature72 as Feature72, "
            "n.Feature73 as Feature73, n.Feature74 as Feature74, "
            "n.Feature75 as Feature75, n.Feature76 as Feature76, "
            "n.Feature77 as Feature77, n.Feature78 as Feature78, "
            "n.Feature79 as Feature79, n.Feature80 as Feature80, "
            "n.Feature81 as Feature81, n.Feature82 as Feature82, "
            "n.Feature83 as Feature83, n.Feature84 as Feature84, "
            "n.Feature85 as Feature85, n.Feature86 as Feature86, "
            "n.Feature87 as Feature87, n.Feature88 as Feature88, "
            "n.Feature89 as Feature89, n.Feature90 as Feature90, "
            "n.Feature91 as Feature91, n.Feature92 as Feature92, "
            "n.Feature93 as Feature93, n.Feature94 as Feature94, "
            "n.Feature95 as Feature95, n.Feature96 as Feature96, "
            "n.Feature97 as Feature97, n.Feature98 as Feature98, "
            "n.Feature99 as Feature99, n.Feature100 as Feature100, "
            "n.Feature101 as Feature101, n.Feature102 as Feature102, "
            "n.Feature103 as Feature103, n.Feature104 as Feature104, "
            "n.Feature105 as Feature105, n.Feature106 as Feature106, "
            "n.Feature107 as Feature107, n.Feature108 as Feature108, "
            "n.Feature109 as Feature109, n.Feature110 as Feature110, "
            "n.Feature111 as Feature111, n.Feature112 as Feature112, "
            "n.Feature113 as Feature113, n.Feature114 as Feature114, "
            "n.Feature115 as Feature115, n.Feature116 as Feature116, "
            "n.Feature117 as Feature117, n.Feature118 as Feature118, "
            "n.Feature119 as Feature119, n.Feature120 as Feature120, "
            "n.Feature121 as Feature121, n.Feature122 as Feature122, "
            "n.Feature123 as Feature123, n.Feature124 as Feature124, "
            "n.Feature125 as Feature125, n.Feature126 as Feature126, "
            "n.Feature127 as Feature127, n.Feature128 as Feature128, "
            "n.Feature129 as Feature129, n.Feature130 as Feature130, "
            "n.Feature131 as Feature131, n.Feature132 as Feature132, "
            "n.Feature133 as Feature133, n.Feature134 as Feature134, "
            "n.Feature135 as Feature135, n.Feature136 as Feature136, "
            "n.Feature137 as Feature137, n.Feature138 as Feature138, "
            "n.Feature139 as Feature139, n.Feature140 as Feature140, "
            "n.Feature141 as Feature141, n.Feature142 as Feature142, "
            "n.Feature143 as Feature143, n.Feature144 as Feature144, "
            "n.Feature145 as Feature145, n.Feature146 as Feature146, "
            "n.Feature147 as Feature147, n.Feature148 as Feature148, "
            "n.Feature149 as Feature149, n.Feature150 as Feature150, "
            "n.Feature151 as Feature151, n.Feature152 as Feature152, "
            "n.Feature153 as Feature153, n.Feature154 as Feature154, "
            "n.Feature155 as Feature155, n.Feature156 as Feature156, "
            "n.Feature157 as Feature157, n.Feature158 as Feature158, "
            "n.Feature159 as Feature159, n.Feature160 as Feature160, "
            "n.Feature161 as Feature161, n.Feature162 as Feature162, "
            "n.Feature163 as Feature163, n.Feature164 as Feature164, "
            "n.Feature165 as Feature165, n.Feature166 as Feature166, "
            "n.Feature167 as Feature167, n.Feature168 as Feature168, "
            "n.Feature169 as Feature169, n.Feature170 as Feature170, "
            "n.Feature171 as Feature171, n.Feature172 as Feature172, "
            "n.Feature173 as Feature173, n.Feature174 as Feature174, "
            "n.Feature175 as Feature175, n.Feature176 as Feature176, "
            "n.Feature177 as Feature177, n.Feature178 as Feature178, "
            "n.Feature179 as Feature179, n.Feature180 as Feature180, "
            "n.Feature181 as Feature181, n.Feature182 as Feature182, "
            "n.Feature183 as Feature183, n.Feature184 as Feature184, "
            "n.Feature185 as Feature185, n.Feature186 as Feature186, "
            "n.Feature188 as Feature187, n.Feature188 as Feature188, "
            "n.Feature189 as Feature189, n.Feature190 as Feature190, "
            "n.Feature191 as Feature191, n.Feature192 as Feature192, "
            "n.Feature193 as Feature193, n.Feature194 as Feature194, "
            "n.Feature195 as Feature195, n.Feature196 as Feature196, "
            "n.Feature197 as Feature197, n.Feature198 as Feature198, "
            "n.Feature199 as Feature199, n.Feature200 as Feature200, "
            "n.Feature201 as Feature201, n.Feature202 as Feature202, "
            "n.Feature203 as Feature203, n.Feature204 as Feature204, "
            "n.Feature205 as Feature205, n.Feature206 as Feature206, "
            "n.Feature207 as Feature207, n.Feature208 as Feature208, "
            "n.Feature209 as Feature209, n.Feature210 as Feature210, "
            "n.Feature211 as Feature211, n.Feature212 as Feature212, "
            "n.Feature213 as Feature213, n.Feature214 as Feature214, "
            "n.Feature215 as Feature215, n.Feature216 as Feature216, "
            "n.Feature217 as Feature217, n.Feature218 as Feature218, "
            "n.Feature219 as Feature219, n.Feature220 as Feature220, "
            "n.Feature221 as Feature221, n.Feature222 as Feature222, "
            "n.Feature223 as Feature223, n.Feature224 as Feature224"
        )
        result = session.run(query)
        for record in result:
            node_id = record["node_id"]
            # Create a list of all 224 features
            features = [float(record[f"Feature{i}"]) for i in range(1, 225)]
            features_dict[node_id] = features
    return nodes, edges, features_dict


nodes, edges, features_dict = load_neo4j_data(driver)
src, dst = zip(*edges)
src = np.array(src)
dst = np.array(dst)
# Create a mapping from Neo4j node IDs to DGL node IDs
node_id_to_dgl_id = {node_id: dgl_id for dgl_id, node_id in enumerate(nodes)}
# Create a list of all nodes (including those without edges)
all_nodes = list(node_id_to_dgl_id.values())
# Construct the DGL graph
g = dgl.graph((src, dst), num_nodes=len(all_nodes))
# Assign features to nodes
features_list = [features_dict[node_id] for node_id in all_nodes]
g.ndata['features'] = torch.tensor(features_list, dtype=torch.float32)
# splitting edge set for training and testing
u, v = g.edges()
eids = np.arange(g.number_of_edges())
eids = np.random.permutation(eids)
test_size = int(len(eids) * 0.1)
train_size = g.number_of_edges() - test_size
test_pos_u, test_pos_v = u[eids[:test_size]], v[eids[:test_size]]
train_pos_u, train_pos_v = u[eids[test_size:]], v[eids[test_size:]]
# finding all negative edges and split them for training and testing
neg_u, neg_v = np.where(np.logical_not(g.adjacency_matrix().to_dense()))
neg_eids = np.random.choice(len(neg_u), g.number_of_edges())
test_neg_u, test_neg_v = neg_u[neg_eids[:test_size]], neg_v[neg_eids[:test_size]]
train_neg_u, train_neg_v = neg_u[neg_eids[test_size:]], neg_v[neg_eids[test_size:]]

train_g = dgl.remove_edges(g, eids[:test_size])

train_pos_g = dgl.graph((train_pos_u, train_pos_v), num_nodes=g.number_of_nodes())
train_neg_g = dgl.graph((train_neg_u, train_neg_v), num_nodes=g.number_of_nodes())
test_pos_g = dgl.graph((test_pos_u, test_pos_v), num_nodes=g.number_of_nodes())
test_neg_g = dgl.graph((test_neg_u, test_neg_v), num_nodes=g.number_of_nodes())

# setting up model, pred, and optimizer required for training the model
num_classes = 1
num_recommendations = 5
model = NodeRecommendationModel(train_g.ndata['features'].shape[1], 16)
pred = DotPredictor()

# function to compute loss,we use binary cross entropy with logits because it involves binary classification
# as we need to pairwise ranking for recommendation systems
def compute_loss(pos_score, neg_score):
    scores = torch.cat([pos_score, neg_score])
    labels = torch.cat([torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])])
    return F.binary_cross_entropy_with_logits(scores, labels)

# function to compute accuracy
def compute_auc(pos_score, neg_score):
    scores = torch.cat([pos_score, neg_score]).numpy()
    labels = torch.cat(
        [torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])]).numpy()
    return roc_auc_score(labels, scores)

# using Adam optimization
optimizer = torch.optim.Adam(itertools.chain(model.parameters(), pred.parameters()), lr=0.001)
all_logits = []

# training the GNN model
for e in range(1000):
    # Forward
    h = model(train_g, train_g.ndata['features'])
    pos_score = pred(train_pos_g, h)
    neg_score = pred(train_neg_g, h)
    loss = compute_loss(pos_score, neg_score)

    # Backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if e % 5 == 0:
        print('In epoch {}, loss: {}'.format(e, loss))


with torch.no_grad():
    h = model(train_g, train_g.ndata['features'])
    all_scores = F.softmax(h, dim=1)

    # Select the top k nodes as recommendations for each target node
    _, top_indices = torch.topk(all_scores, num_recommendations, dim=1)

    correct_hits = 0
    total_targets = 0

    for idx, target_node in enumerate(all_nodes):
        top_recommended_nodes = top_indices[idx].tolist()
        print(f"Top {num_recommendations} recommendations for node {target_node}: {top_recommended_nodes}")

        # Likelihood scores for the top recommended nodes
        likelihood_scores = all_scores[idx][top_recommended_nodes].tolist()
        print(f"Likelihood scores: {likelihood_scores}")

        # Check if the target node is among the top recommended nodes
        if target_node in top_recommended_nodes:
            correct_hits += 1

        total_targets += 1

    # Compute HR@k (Hit Rate at k)
    hit_rate = correct_hits / total_targets
    print(f"Hit Rate at {num_recommendations}: {hit_rate:.4f}")



