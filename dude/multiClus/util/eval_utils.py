import numpy as np
from sklearn import metrics
from munkres import Munkres
import numpy as np
import torch
import os
from torch import nn
import faiss
from contextlib import redirect_stdout
from scipy.optimize import linear_sum_assignment
from torch.utils.data import DataLoader, TensorDataset

def cluster_metric(label, pred,state_logger=None):
    nmi = metrics.normalized_mutual_info_score(label, pred)
    ari = metrics.adjusted_rand_score(label, pred)
    try:
        pred_adjusted = get_y_preds(label, pred, len(set(label)))
        acc = metrics.accuracy_score(pred_adjusted, label)
    except:
        acc = 0
    purity = purity_score(label, pred)
    if state_logger!=None:
        state_logger.write(
            "[Clustering Result]: ACC = {:.2f}, NMI = {:.2f}, ARI = {:.2f}, Purity = {:.2f}".format(
                acc * 100, nmi * 100, ari * 100, purity * 100
            )
        )
    else:
        print(
            "[Clustering Result]: ACC = {:.2f}, NMI = {:.2f}, ARI = {:.2f}, Purity = {:.2f}".format(
                acc * 100, nmi * 100, ari * 100, purity * 100
            )
        )
    return acc


def purity_score(y_true, y_pred):
    # compute contingency matrix (also called confusion matrix)
    contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
    # return purity
    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)


def calculate_cost_matrix(C, n_clusters):
    cost_matrix = np.zeros((n_clusters, n_clusters))
    # cost_matrix[i,j] will be the cost of assigning cluster i to label j
    for j in range(n_clusters):
        s = np.sum(C[:, j])  # number of examples in cluster i
        for i in range(n_clusters):
            t = C[i, j]
            cost_matrix[j, i] = s - t
    return cost_matrix


def get_cluster_labels_from_indices(indices):
    n_clusters = len(indices)
    cluster_labels = np.zeros(n_clusters)
    for i in range(n_clusters):
        cluster_labels[i] = indices[i][1]
    return cluster_labels


def get_y_preds(y_true, cluster_assignments, n_clusters):
    """
    Computes the predicted labels, where label assignments now
    correspond to the actual labels in y_true (as estimated by Munkres)
    cluster_assignments:    array of labels, outputted by kmeans
    y_true:                 true labels
    n_clusters:             number of clusters in the dataset
    returns:    a tuple containing the accuracy and confusion matrix,
                in that order
    """
    confusion_matrix = metrics.confusion_matrix(
        y_true, cluster_assignments, labels=None
    )
    # compute accuracy based on optimal 1:1 assignment of clusters to labels
    cost_matrix = calculate_cost_matrix(confusion_matrix, n_clusters)
    indices = Munkres().compute(cost_matrix)
    kmeans_to_true_cluster_labels = get_cluster_labels_from_indices(indices)

    if np.min(cluster_assignments) != 0:
        cluster_assignments = cluster_assignments - np.min(cluster_assignments)
    y_pred = kmeans_to_true_cluster_labels[cluster_assignments]
    return y_pred

def kmeans(X, cluster_num):
    print("Perform K-means clustering...")
    d = X.shape[1]
    X = X.astype(np.float32)
    kmeans = faiss.Kmeans(d, cluster_num, gpu=True, spherical=True, niter=300, nredo=10)
    kmeans.train(X)
    D, I = kmeans.index.search(X, 1)
    I = I.reshape(-1)
    print("K-means clustering done.")
    return I

def cal_kmeans_loop(mlp, output_dir, max_train_steps, dataset):
    model_prefix = "mlp_model_"
    model_suffix = ".pth"
    if dataset=='dog':
        labels = np.load('stfdog_train_labels.npy')
        print(len(labels))
        data = np.load('stfdog_train_rand.npy')
    elif dataset=='cub':
        labels = np.load('cub_train_labels.npy')
        print(len(labels))
        data = np.load('cub_train_rand.npy')
    elif dataset=='flower':
        labels = np.load('flower_train_labels.npy')
        print(len(labels))
        data = np.load('flower_train_rand.npy')
    data_tensor = torch.tensor(data, dtype=torch.float32).cuda()
    # 循环遍历 mlp_model_0 到 mlp_model_500
    with open(f'{output_dir}.out', 'w') as f:
        for i in range(0,max_train_steps):
            # 定义文件夹路径列表
            save_mlp = os.path.join(output_dir, f"{model_prefix}{i}{model_suffix}")
            mlp.load_state_dict(torch.load(save_mlp))
            mlp.eval()
            #main_folders = [f"experiments/1000_{i}" for i in range(1, 7)]
            with torch.no_grad():
                con_embedds = mlp.get_mlpout(data_tensor)
            all_embeds = con_embedds.detach().cpu().numpy()
            all_embeds = np.array(all_embeds)
            #labels = np.load('dift1000_labels.npy')
            # Apply KMeans
            num_clusters = len(np.unique(labels))
            preds = kmeans(all_embeds, num_clusters)
            with redirect_stdout(f):
                print(f"i= {i} ")
                cluster_metric(labels,preds)
    
def infer(model, dataloader):
    model.eval()
    preds = []
    logits_image = []
    with torch.no_grad():
        for iter, (image) in enumerate(dataloader):
            image = image[0].cuda()
            logit_image = model(image)
            pred = torch.argmax(logit_image, dim=1).cpu().numpy()
            preds.append(pred)
            logits_image.append(logit_image.cpu().numpy())
    preds = np.concatenate(preds, axis=0)
    logits_image = np.concatenate(logits_image, axis=0)
    return preds, logits_image

def get_stat(gen_label, num_gt_lab, gt_lab):
    c_stat = np.zeros([num_gt_lab,num_gt_lab])
    for i in range(len(gt_lab)):
        gt_idx = int(gt_lab[i])
        c_stat[gt_idx][gen_label[i]] += 1
    return c_stat

def get_match(stat):
    _, col_ind = linear_sum_assignment(stat.max()-stat)
    return col_ind

def get_nmi(stat):
    n,m = stat.shape
    pij = stat/np.sum(stat)
    pi = np.sum(pij, 1)
    pj = np.sum(pij, 0)
    enti = sum([-pi[i]*np.log2(pi[i]+1e-6) for i in range(n)])
    entj = sum([-pj[i]*np.log2(pj[i]+1e-6) for i in range(m)])
    mi = 0
    for i in range(n):
        for j in range(m):
            mi += pij[i][j]*(np.log2(pij[i][j]/(pi[i]*pj[j]+1e-6)+1e-6))
    return mi/max(enti, entj)

def get_acc(stat, col_ind):
    tot = 0
    for i in range(stat.shape[0]):
        tot += stat[i][col_ind[i]]
    return tot/(np.sum(stat))

def eval_clusterhead(test_features, labels_test, model, cluster_head, cluster_num, targets_k, args):
    model.eval()
    with torch.no_grad():
        batch_size = 512
        images_embedding_test = []
        for i in range(0, len(test_features), batch_size):
            batch_features = test_features[i : i + batch_size]
            batch_tensor = torch.tensor(batch_features).cuda()
            batch_embedding = (
                model(batch_tensor).detach().cpu().numpy()
            )
            images_embedding_test.append(batch_embedding)
        images_embedding_test = np.concatenate(images_embedding_test, axis=0)
        # images_embedding_test = model.module.get_cnnout(torch.tensor(order_features).cuda()).detach().cpu().numpy()
        images_embedding_test = images_embedding_test / np.linalg.norm(
            images_embedding_test, axis=1, keepdims=True
        )
        dataset_image_test = TensorDataset(
            torch.from_numpy(images_embedding_test).float()
        )
        dataloader_test = DataLoader(
            dataset_image_test,
            batch_size=args.train_batch_size,
            shuffle=False,
            drop_last=False,
        )
        preds, confidences_image = infer(cluster_head, dataloader_test)
        c_table = get_stat(preds, cluster_num, labels_test)
        idx_map = get_match(c_table)
        cur_acc = get_acc(c_table, idx_map)
        cur_nmi = get_nmi(c_table[:targets_k,:])
        print("acc: ", cur_acc, " nmi: ",cur_nmi) 
        
        return confidences_image
    
def eval_cnn(test_features, labels_test, model, cluster_num, targets_k, args):
    model.eval()
    with torch.no_grad():
        batch_size = 512
        images_embedding_test = []
        for i in range(0, len(test_features), batch_size):
            batch_features = test_features[i : i + batch_size]
            batch_tensor = torch.tensor(batch_features).cuda()
            batch_embedding = (
                model(batch_tensor).detach().cpu().numpy()
            )
            images_embedding_test.append(batch_embedding)
        images_embedding_test = np.concatenate(images_embedding_test, axis=0)
        # images_embedding_test = model.module.get_cnnout(torch.tensor(order_features).cuda()).detach().cpu().numpy()
        images_embedding_test = images_embedding_test / np.linalg.norm(
            images_embedding_test, axis=1, keepdims=True
        )
        dataset_image_test = TensorDataset(
            torch.from_numpy(images_embedding_test).float()
        )
        dataloader_test = DataLoader(
            dataset_image_test,
            batch_size=args.train_batch_size,
            shuffle=False,
            drop_last=False,
        )
        preds = kmeans(images_embedding_test, cluster_num)
        c_table = get_stat(preds, cluster_num, labels_test)
        idx_map = get_match(c_table)
        cur_acc = get_acc(c_table, idx_map)
        cur_nmi = get_nmi(c_table[:targets_k,:])
        print("acc: ", cur_acc, " nmi: ",cur_nmi) 
