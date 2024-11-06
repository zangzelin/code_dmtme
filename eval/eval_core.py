from symtable import Symbol
from tkinter.messagebox import NO
import uuid
import scipy

# import sklearn
import torch
from sklearn.cluster import SpectralClustering
import sklearn
from sklearn.preprocessing import MinMaxScaler
import scipy.spatial as spt

# from scipy.spatial.distance import squareform
# from scipy.stats import spearmanr
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from sklearn import metrics
from sklearn.svm import SVC
import numpy as np
import plotly.figure_factory as ff
# from transformers import RagRetriever
import wandb

# import random
from sklearn.metrics import pairwise_distances

from sklearn.linear_model import LinearRegression

# from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from munkres import Munkres
from sklearn.neighbors import kneighbors_graph
import matplotlib.pyplot as plt
# import umap
import plotly.express as px
from sklearn.ensemble import ExtraTreesClassifier
import plotly.graph_objects as go

    # from sklearn.preprocessing import MinMaxScaler

class Eval:
    def __init__(
        self, input, latent, label, train_input, train_latent, train_label, mask
    ) -> None:

        self.input = input
        self.latent = latent
        self.label = label
        self.train_input = train_input
        self.train_latent = train_latent
        self.train_label = train_label
        self.mask = mask

    def GraphMatch(self, K=10):

        self.K = min(K, self.input.shape[0])
        input_knn_graph = kneighbors_graph(
            self.input, n_neighbors=self.K, mode="connectivity", include_self=True
        )
        mask_knn_graph = kneighbors_graph(
            self.input[:, self.mask],
            n_neighbors=self.K,
            mode="connectivity",
            include_self=True,
        )

        self.input_knn_graph = input_knn_graph
        self.mask_knn_graph = mask_knn_graph

        mask_input_is_con = input_knn_graph > 0.5
        rate = (mask_knn_graph[mask_input_is_con] > 0.5).sum() / (
            mask_input_is_con.sum()
        )
        return rate

    def VisSelectUMAP(self, data, label):

        # print(data.shape)
        data_masked = data[:, self.mask]
        emb_selected_data = umap.UMAP().fit_transform(data_masked)
        # plt.scatter(emb_selected_data[:,0], emb_selected_data[:,1], c=label)

        return px.scatter(
            x=emb_selected_data[:, 0],
            y=emb_selected_data[:, 1],
            color=[str(i) for i in label.tolist()],
            # color=label,
        )

    def VisAllUMAP(self, data, label):

        data_masked = data  # [:, self.mask]

        emb_selected_data = umap.UMAP().fit_transform(data_masked)
        # plt.scatter(emb_selected_data[:,0], emb_selected_data[:,1], c=label)
        return px.scatter(
            x=emb_selected_data[:, 0],
            y=emb_selected_data[:, 1],
            # color=label,
            color=[str(i) for i in label.tolist()],
        )

    def GetGraphMatchHist(self, epoch=0, txt=""):
        mask_input_is_con = self.input_knn_graph.todense() > 0.5
        rate_hist = (self.mask_knn_graph.todense()[mask_input_is_con] > 0.5).reshape(
            (-1, self.K)
        ).sum(axis=1) / self.K
        # import pdb; pdb.set_trace()
        # print(self.input_knn_graph.shape)
        # print(rate_hist.shape)
        rate_hist_array = np.array(rate_hist.view()).reshape((-1))
        # plt.figure()
        plt.hist(x=rate_hist_array, bins=10, density=True, alpha=0.75)
        savepath = "./tem/" + txt + "epoch{}_{}.npy".format(epoch, str(uuid.uuid1()))
        np.save(savepath, rate_hist_array)
        # plt.savefig('test_hist.png')
        # plt.close()
        return savepath

    def GraphMatchLatent(self, K=10):

        input_knn_graph = kneighbors_graph(
            self.input, n_neighbors=K, mode="connectivity", include_self=True
        )
        mask_knn_graph = kneighbors_graph(
            self.latent, n_neighbors=K, mode="connectivity", include_self=True
        )

        mask_input_is_con = input_knn_graph > 0.5
        rate = (mask_knn_graph[mask_input_is_con] > 0.5).sum() / (
            mask_input_is_con.sum()
        )
        return rate

    def _neighbours_and_ranks(self, distances):
        """
        Inputs:
        - distances,        distance matrix [n times n],
        - k,                number of nearest neighbours to consider
        Returns:
        - neighbourhood,    contains the sample indices (from 0 to n-1) of kth nearest neighbor of current sample [n times k]
        - ranks,            contains the rank of each sample to each sample [n times n], whereas entry (i,j) gives the rank that sample j has to i (the how many 'closest' neighbour j is to i)
        """
        k = self.k
        # Warning: this is only the ordering of neighbours that we need to
        # extract neighbourhoods below. The ranking comes later!
        indices = np.argsort(distances, axis=-1, kind="stable")

        # Extract neighbourhoods.
        neighbourhood = indices[:, 1 : k + 1]

        # Convert this into ranks (finally)
        ranks = indices.argsort(axis=-1, kind="stable")
        # print(ranks)

        return neighbourhood, ranks

    def _Distance_squared_GPU(self, x, y, cuda=7):

        x = torch.tensor(x).cuda()
        y = torch.tensor(y).cuda()
        m, n = x.size(0), y.size(0)
        xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
        yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
        dist = xx + yy
        dist = torch.addmm(dist, mat1=x, mat2=y.t(), beta=1, alpha=-2)

        d = dist.clamp(min=1e-36)
        return np.sqrt(d.detach().cpu().numpy())

    def _Distance_squared_CPU(self, x, y):

        x = torch.tensor(x)
        y = torch.tensor(y)
        m, n = x.size(0), y.size(0)
        xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
        yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
        dist = xx + yy
        # dist.addmm_(1, -2, x, y.t())
        dist = torch.addmm(dist, mat1=x, mat2=y.t(), beta=1, alpha=-2)
        d = dist.clamp(min=1e-36)
        return d.detach().cpu().numpy()

    def _trustworthiness(
        self, X_neighbourhood, X_ranks, Z_neighbourhood, Z_ranks, n, k
    ):
        """
        Calculates the trustworthiness measure between the data space `X`
        and the latent space `Z`, given a neighbourhood parameter `k` for
        defining the extent of neighbourhoods.
        """

        result = 0.0

        # Calculate number of neighbours that are in the $k$-neighbourhood
        # of the latent space but not in the $k$-neighbourhood of the data
        # space.
        for row in range(X_ranks.shape[0]):
            missing_neighbours = np.setdiff1d(
                Z_neighbourhood[row], X_neighbourhood[row]
            )

            for neighbour in missing_neighbours:
                result += X_ranks[row, neighbour] - k

        return 1 - 2 / (n * k * (2 * n - 3 * k - 1)) * result

    def E_SVC_ACC(self):

        method = SVC(kernel="linear", max_iter=90000)
        method.fit(self.train_input[:, self.mask], self.train_label)
        return metrics.accuracy_score(
            self.label, method.predict(self.input[:, self.mask])
        )

    def E_SVC_ACC_Latent(self):

        method = SVC(kernel="linear", max_iter=90000)
        method.fit(self.train_latent, self.train_label)
        return metrics.accuracy_score(self.label, method.predict(self.latent))

    def E_ExtraTrees_ACC(self):
        from sklearn.ensemble import ExtraTreesClassifier

        method = ExtraTreesClassifier(n_estimators=50, random_state=0)
        method.fit(self.train_input[:, self.mask], self.train_label)
        return metrics.accuracy_score(
            self.label, method.predict(self.input[:, self.mask])
        )

    def E_ExtraTrees_ACC_Latent(self):

        from sklearn.ensemble import ExtraTreesClassifier

        method = ExtraTreesClassifier(n_estimators=50, random_state=0)
        method.fit(self.train_latent, self.train_label)
        return metrics.accuracy_score(self.label, method.predict(self.latent))

    def E_ExtraTrees_ACC_valtest(self, seed=0):

        from sklearn.ensemble import ExtraTreesClassifier

        method = ExtraTreesClassifier(n_estimators=50, random_state=seed)
        method.fit(self.train_input[:, self.mask], self.train_label)
        valtest_fea = self.input[:, self.mask]
        valtest_lab = self.label.astype(np.int32)

        feature_val, feature_test, label_val, label_test = train_test_split(
            valtest_fea, valtest_lab, test_size=0.5, random_state=0
        )
        val_acc = metrics.accuracy_score(label_val, method.predict(feature_val))
        tes_acc = metrics.accuracy_score(label_test, method.predict(feature_test))

        return val_acc, tes_acc

    def E_ExtraTrees_ACC_Latent_valtest(self, seed=0):

        from sklearn.ensemble import ExtraTreesClassifier

        method = ExtraTreesClassifier(n_estimators=50, random_state=seed)
        method.fit(self.train_latent, self.train_label)
        valtest_fea = self.latent
        valtest_lab = self.label

        feature_val, feature_test, label_val, label_test = train_test_split(
            valtest_fea, valtest_lab, test_size=0.5, random_state=0
        )
        val_acc = metrics.accuracy_score(label_val, method.predict(feature_val))
        tes_acc = metrics.accuracy_score(label_test, method.predict(feature_test))

        return val_acc, tes_acc

    def E_Kmeans_ACC_Latent(self, seed=0):

        acc, nmi, f1_macro, precision_macro, adjscore = self.TestClassifacationKMeans(
            self.train_latent, self.train_label, seed=seed
        )
        return acc

    def E_Kmeans_ACC(self, seed=0):

        acc, nmi, f1_macro, precision_macro, adjscore = self.TestClassifacationKMeans(
            self.train_input[:, self.mask], self.train_label, seed=seed
        )
        return acc

    def E_Kmeans_ACC_Norm(self):

        acc, nmi, f1_macro, precision_macro, adjscore = self.TestClassifacationKMeans(
            StandardScaler().fit_transform(self.train_input[:, self.mask]),
            self.train_label,
        )
        return acc

    def E_Kmeans_ACC_Latent_TEST(self):

        acc, nmi, f1_macro, precision_macro, adjscore = self.TestClassifacationKMeans(
            self.latent, self.label
        )
        return acc

    def E_Kmeans_ACC_TEST(self):

        acc, nmi, f1_macro, precision_macro, adjscore = self.TestClassifacationKMeans(
            self.input[:, self.mask], self.label
        )
        return acc

    def E_Kmeans_ACC_valtest(self, seed=0):

        valtest_fea = StandardScaler().fit_transform(self.input[:, self.mask])
        valtest_lab = self.label
        feature_val, feature_test, label_val, label_test = train_test_split(
            valtest_fea, valtest_lab, test_size=0.5, random_state=0
        )
        (
            acc_val,
            nmi,
            f1_macro,
            precision_macro,
            adjscore,
        ) = self.TestClassifacationKMeans(feature_val, label_val, seed=seed)
        (
            acc_test,
            nmi,
            f1_macro,
            precision_macro,
            adjscore,
        ) = self.TestClassifacationKMeans(feature_test, label_test, seed=seed)

        return acc_val, acc_test

    def E_Kmeans_ACC_Latent_valtest(self, seed=0):

        valtest_fea = self.latent
        valtest_lab = self.label
        feature_val, feature_test, label_val, label_test = train_test_split(
            valtest_fea, valtest_lab, test_size=0.5, random_state=0
        )
        (
            acc_val,
            nmi,
            f1_macro,
            precision_macro,
            adjscore,
        ) = self.TestClassifacationKMeans(feature_val, label_val, seed=seed)
        (
            acc_test,
            nmi,
            f1_macro,
            precision_macro,
            adjscore,
        ) = self.TestClassifacationKMeans(feature_test, label_test, seed=seed)

        return acc_val, acc_test

    def E_Classifacation_SVC_Mask(self, mask):

        from sklearn.preprocessing import StandardScaler

        method = SVC(kernel="linear", max_iter=90000)
        cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=1, random_state=1)
        # if
        n_scores = cross_val_score(
            method,
            StandardScaler().fit_transform(self.input[:, mask]),
            self.label.astype(np.int32),
            scoring="accuracy",
            cv=cv,
            n_jobs=-1,
        )

        return n_scores.mean()

    def E_Classifacation_KNN(self):

        from sklearn.neighbors import KNeighborsClassifier

        method = KNeighborsClassifier(n_neighbors=3)
        cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=1, random_state=1)
        # if
        n_scores = cross_val_score(
            method,
            self.latent,
            self.label.astype(np.int32),
            scoring="accuracy",
            cv=cv,
            n_jobs=-1,
        )

        return n_scores.mean()

    def E_NNACC(self):

        indexNN = self.neighbour_latent[:, 0].reshape(-1)
        labelNN = self.label[indexNN]
        acc = (self.label == labelNN).sum() / self.label.shape[0]

        return acc

    def E_mrre(
        self,
    ):
        """
        Calculates the mean relative rank error quality metric of the data
        space `X` with respect to the latent space `Z`, subject to its $k$
        nearest neighbours.
        """
        k = self.k

        X_neighbourhood, X_ranks = self.neighbour_input, self.rank_input
        Z_neighbourhood, Z_ranks = self.neighbour_latent, self.rank_latent

        n = self.distance_input.shape[0]

        # First component goes from the latent space to the data space, i.e.
        # the relative quality of neighbours in `Z`.

        mrre_ZX = 0.0
        for row in range(n):
            for neighbour in Z_neighbourhood[row]:
                rx = X_ranks[row, neighbour]
                rz = Z_ranks[row, neighbour]

                mrre_ZX += abs(rx - rz) / rz

        # Second component goes from the data space to the latent space,
        # i.e. the relative quality of neighbours in `X`.

        mrre_XZ = 0.0
        for row in range(n):
            # Note that this uses a different neighbourhood definition!
            for neighbour in X_neighbourhood[row]:
                rx = X_ranks[row, neighbour]
                rz = Z_ranks[row, neighbour]

                # Note that this uses a different normalisation factor
                mrre_XZ += abs(rx - rz) / rx

        # Normalisation constant
        C = n * sum([abs(2 * j - n - 1) / j for j in range(1, k + 1)])
        return mrre_ZX / C, mrre_XZ / C

    def E_distanceAUC(
        self,
    ):

        disZN = (self.distance_latnet - self.distance_latnet.min()) / (
            self.distance_latnet.max() - self.distance_latnet.min()
        )
        LRepeat = self.label.reshape(1, -1).repeat(
            self.distance_latnet.shape[0], axis=0
        )
        L = (LRepeat == LRepeat.T).reshape(-1)
        auc = metrics.roc_auc_score(1 - L, disZN.reshape(-1))

        return auc

    def E_trustworthiness(self):
        X_neighbourhood, X_ranks = self.neighbour_input, self.rank_input
        Z_neighbourhood, Z_ranks = self.neighbour_latent, self.rank_latent
        n = self.distance_input.shape[0]
        return self._trustworthiness(
            X_neighbourhood, X_ranks, Z_neighbourhood, Z_ranks, n, self.k
        )

    def E_continuity(self):
        """
        Calculates the continuity measure between the data space `X` and the
        latent space `Z`, given a neighbourhood parameter `k` for setting up
        the extent of neighbourhoods.

        This is just the 'flipped' variant of the 'trustworthiness' measure.
        """

        X_neighbourhood, X_ranks = self.neighbour_input, self.rank_input
        Z_neighbourhood, Z_ranks = self.neighbour_latent, self.rank_latent
        n = self.distance_input.shape[0]
        # Notice that the parameters have to be flipped here.
        return self._trustworthiness(
            Z_neighbourhood, Z_ranks, X_neighbourhood, X_ranks, n, self.k
        )

    def E_Rscore(self):
        # n = self.distance_input.shape[0]
        import scipy

        r = scipy.stats.pearsonr(
            self.distance_input.reshape(-1), self.distance_latnet.reshape(-1)
        )
        # print(r)
        return r[0]

    def E_Dismatcher(self):
        emb, label = self.latent, self.label
        list_dis = []
        for i in list(set(label)):
            p = emb[label == i]
            m = p.mean(axis=0)[None, :]
            list_dis.append(pairwise_distances(p, m).mean())
        list_dis = np.array(list_dis)
        list_dis_norm = list_dis / list_dis.max()
        sort1 = np.argsort(list_dis_norm)
        # print('latent std:', list_dis_norm)
        # print('latent sort:', sort1)

        emb, label = self.input, self.label
        emb = emb.reshape(emb.shape[0], -1)
        list_dis = []
        for i in list(set(label)):
            p = emb[label == i]
            m = p.mean(axis=0)[None, :]
            list_dis.append(pairwise_distances(p, m).mean())
        list_dis = np.array(list_dis)
        list_dis_norm = list_dis / list_dis.max()
        sort2 = np.argsort(list_dis_norm)
        # print('latent std:', list_dis_norm)
        # print('latent sort:', sort2)

        v, s, t = 0, sort2.tolist(), sort1.tolist()
        for i in range(len(t)):
            if t[i] != s[i]:
                v = v + abs(t.index(s[i]) - i)
        s_constant = 2.0 / len(s) ** 2

        return v * s_constant

    def TestClassifacationKMeans(self, embedding, label, seed=0):

        label = np.array(label).reshape(-1)
        l1 = list(set(label.tolist()))
        numclass1 = len(l1)
        embedding = StandardScaler().fit_transform(embedding)
        predict_labels = KMeans(n_clusters=numclass1, random_state=seed).fit_predict(
            embedding
        )
        # predict_labels = SpectralClustering(n_clusters=numclass1, random_state=0).fit_predict(embedding)

        l2 = list(set(predict_labels))
        numclass2 = len(l2)
        cost = np.zeros((numclass1, numclass2), dtype=int)
        for i, c1 in enumerate(l1):
            mps = [i1 for i1, e1 in enumerate(label) if e1 == c1]
            for j, c2 in enumerate(l2):
                mps_d = [i1 for i1 in mps if predict_labels[i1] == c2]
                cost[i][j] = len(mps_d)

        # match two clustering results by Munkres algorithm
        m = Munkres()
        cost = cost.__neg__().tolist()
        indexes = m.compute(cost)
        # get the match results
        new_predict = np.zeros(len(predict_labels))
        for i, c in enumerate(l1):
            # correponding label in l2:
            c2 = l2[indexes[i][1]]
            # ai is the index with label==c2 in the pred_label list
            ai = [ind for ind, elm in enumerate(predict_labels) if elm == c2]
            new_predict[ai] = c

        acc = metrics.accuracy_score(label, new_predict)
        f1_macro = metrics.f1_score(label, new_predict, average="macro")
        precision_macro = metrics.precision_score(label, new_predict, average="macro")
        nmi = metrics.normalized_mutual_info_score(label, predict_labels)
        adjscore = metrics.adjusted_rand_score(label, predict_labels)
        return acc, nmi, f1_macro, precision_macro, adjscore

    def E_ExtraTreesClassifier(self, train_data, train_label):

        method = ExtraTreesClassifier(n_estimators=5, random_state=0)
        method.fit(train_data, train_label)
        n_scores = metrics.accuracy_score(self.label, method.predict(self.latent))

        return n_scores.mean()

    def E_LinearRegressionLoss(self):

        reg = LinearRegression().fit(self.train_input[:, self.mask], self.train_input)
        return np.mean((reg.predict(self.input[:, self.mask]) - self.input) ** 2)

    def E_LinearRegressionLossLatent(self):
        reg = LinearRegression().fit(self.train_latent, self.train_input)
        return np.mean((reg.predict(self.latent) - self.input) ** 2)


def showMask(MaskWeight, t=0.1):
    # def showMaskHeatMap(self, t=0.1):
    plt.figure()
    fig, ax = plt.subplots(figsize=(5, 5))
    data = MaskWeight.detach().cpu().numpy()  # .reshape(dim_i, dim_j)
    N_allF = len(MaskWeight.detach().cpu().numpy())
    N_c = int(np.sqrt(N_allF))
    N_r = N_allF // N_c
    if N_c * N_r < N_allF:
        N_r += 1
    data = np.concatenate([data, np.array([0] * (N_c * N_r - N_allF))]).reshape(
        N_c, N_r
    )
    data[data < t] = 0
    im = plt.imshow(data)
    plt.colorbar(im)
    plt.close()
    return fig


def Test_ET_CV(e, wandb_logs, bestval):

    his_r = np.zeros(shape=(10, 2))
    for seed in range(10):
        acc_val, acc_test = e.E_ExtraTrees_ACC_valtest(seed=seed)
        # acc_val_l, acc_test_l = e.E_ExtraTrees_ACC_Latent_valtest(seed=seed)
        # acc_val_km, acc_test_km = e.E_Kmeans_ACC_valtest(seed=seed)
        # acc_val_km_l, acc_test_km_l = e.E_Kmeans_ACC_Latent_valtest(seed=seed)

        his_r[seed, :] = np.array(
            [
                acc_val,
                acc_test,
                # acc_val_l,
                # acc_test_l,
            ]
        )

        wandb_logs.update(
            {
                "m/ET_ACC_val_{}".format(seed): acc_val,
                "m/ET_ACC_test_{}".format(seed): acc_test,
                # "m/ET_ACC_Latent_val_{}".format(seed): acc_val_l,
                # "m/ET_ACC_Latent_test_{}".format(seed): acc_test_l,
            }
        )
    wandb_logs.update(
        {
            "mmean/ET_ACC_val_mean": np.mean(his_r[:, 0]),
            "mmean/ET_ACC_test_mean": np.mean(his_r[:, 1]),
            # "mmean/ET_ACC_Latent_val_mean": np.mean(his_r[:, 2]),
            # "mmean/ET_ACC_Latent_test_mean": np.mean(his_r[:, 3]),
        }
    )
    wandb_logs.update(
        {
            "mstd/ET_ACC_val_std": np.std(his_r[:, 0]),
            "mstd/ET_ACC_test_std": np.std(his_r[:, 1]),
            # "mstd/ET_ACC_Latent_val_std": np.std(his_r[:, 2]),
            # "mstd/ET_ACC_Latent_test_std": np.std(his_r[:, 3]),
        }
    )
    if bestval < np.mean(his_r[:, 0]):
        bestval = np.mean(his_r[:, 0])
        wandb_logs.update(
            {
                "mbest/ET_ACC_val_mean": np.mean(his_r[:, 0]),
                "mbest/ET_ACC_test_mean": np.mean(his_r[:, 1]),
                # "mbest/ET_ACC_Latent_val_mean": np.mean(his_r[:, 2]),
                # "mbest/ET_ACC_Latent_test_mean": np.mean(his_r[:, 3]),
            }
        )
    return bestval


def Test_KM_CV(e, wandb_logs, bestval):

    his_r = np.zeros(shape=(10, 2))
    for seed in range(10):
        # acc_val, acc_test = e.E_ExtraTrees_ACC_valtest(seed=seed)
        # acc_val_l, acc_test_l = e.E_ExtraTrees_ACC_Latent_valtest(seed=seed)
        acc_val, acc_test = e.E_Kmeans_ACC_valtest(seed=seed)
        # acc_val_km_l, acc_test_km_l = e.E_Kmeans_ACC_Latent_valtest(seed=seed)

        his_r[seed, :] = np.array(
            [
                acc_val,
                acc_test,
            ]
        )

        wandb_logs.update(
            {
                "m/KM_ACC_val_{}".format(seed): acc_val,
                "m/KM_ACC_test_{}".format(seed): acc_test,
                # "m/KM_ACC_Latent_val_{}".format(seed): acc_val_l,
                # "m/KM_ACC_Latent_test_{}".format(seed): acc_test_l,
            }
        )
    wandb_logs.update(
        {
            "mmean/KM_ACC_val_mean": np.mean(his_r[:, 0]),
            "mmean/KM_ACC_test_mean": np.mean(his_r[:, 1]),
            # "mmean/KM_ACC_Latent_val_mean": np.mean(his_r[:, 2]),
            # "mmean/KM_ACC_Latent_test_mean": np.mean(his_r[:, 3]),
        }
    )
    wandb_logs.update(
        {
            "mstd/KM_ACC_val_std": np.std(his_r[:, 0]),
            "mstd/KM_ACC_test_std": np.std(his_r[:, 1]),
            # "mstd/KM_ACC_Latent_val_std": np.std(his_r[:, 2]),
            # "mstd/KM_ACC_Latent_test_std": np.std(his_r[:, 3]),
        }
    )
    if bestval < np.mean(his_r[:, 0]):
        bestval = np.mean(his_r[:, 0])
        wandb_logs.update(
            {
                "mbest/KM_ACC_val_mean": np.mean(his_r[:, 0]),
                "mbest/KM_ACC_test_mean": np.mean(his_r[:, 1]),
                # "mbest/KM_ACC_Latent_val_mean": np.mean(his_r[:, 2]),
                # "mbest/KM_ACC_Latent_test_mean": np.mean(his_r[:, 3]),
            }
        )
    return bestval


def ShowEmb(latent, labelstr, index):

    wandb_logs = {}
    for i in range(len(labelstr)):

        if latent.shape[1] > 2:
            latent2d = umap.UMAP().fit_transform(latent)
        else:
            latent2d = latent

        color = np.array(labelstr[i])[index]
        latent2d = np.concatenate([latent2d, latent2d[-2:-1]])
        color = np.concatenate([color, color[-2:-1]])

        # wandb_logs["vis_meta/0emb{}".format(str(i))] = px.scatter(
        #     x=latent2d[:, 0],
        #     y=latent2d[:, 1],
        #     size=[0.1]*(latent2d.shape[0]-1)+[1],
        #     # symbol=['asterisk']*latent2d.shape[0],
        #     size_max=15,
        #     width=500,
        #     color=color,
        # )
        fig = plt.figure(figsize=(10, 10))
        plt.scatter(
            x=latent2d[:, 0],
            y=latent2d[:, 1],
            c=np.array(color).astype(np.int32),
            s=5,
        )

        wandb_logs["vis_meta/0emb{}".format(str(i))] = fig

    return wandb_logs


def Plot_subfig_1_1(
    latent,
    label,
    index,
    row=1,
    col=1,
    label_name_list=None,
):

    # wandb_logs = {}
    # for i in range(len(labelstr)):

    if latent.shape[1] > 2:
        latent2d = umap.UMAP().fit_transform(latent)
    else:
        latent2d = latent

    color = np.array(label)
    # latent2d = np.concatenate([latent2d, latent2d[-2:-1]])
    # color = np.concatenate([color, color[-2:-1]])
    color_set = list(set(color.tolist()))

    print(color_set)
    fig_list = []
    for c in color_set:
        mask = label == c

        if label_name_list is not None:
            name_c = label_name_list[c]
        else:
            name_c = c

        fig = go.Scatter(
            x=latent2d[mask, 0],
            y=latent2d[mask, 1],
            mode="markers",
            name=name_c,
            # text=['index:{} class:{}'.format(index[i],
            #  int(color[i]) ) for i in range(latent2d.shape[0])],
            marker_line_width=0,
            marker=dict(
                size=[5] * color.shape[0],
                # color=color[mask].astype(np.int32),
            ),
        )
        fig_list.append(fig)

    return fig_list


def Plot_subfig_1_1_easy(
    latent,
    labelstr,
    index,
    row=1,
    col=1,
):

    # wandb_logs = {}
    for i in range(len(labelstr)):

        if latent.shape[1] > 2:
            latent2d = umap.UMAP().fit_transform(latent)
        else:
            latent2d = latent

        color = np.array(labelstr[i])[index]
        # latent2d = np.concatenate([latent2d, latent2d[-2:-1]])
        # color = np.concatenate([color, color[-2:-1]])

        fig = go.Scatter(
            x=latent2d[:, 0],
            y=latent2d[:, 1],
            mode="markers",
            text=[
                "index:{} class:{}".format(index[i], int(color[i]))
                for i in range(latent2d.shape[0])
            ],
            marker_line_width=0,
            marker=dict(
                size=[5] * color.shape[0],
                color=color.astype(np.int32),
            ),
        )

    return [fig]


def Plot_subfig_1_2(
    ins_emb,
    label_pesodu,
    cluster_centers,
    # shap_values,
):

    n_clusters = cluster_centers.shape[0]

    cluster_list = []
    for i in range(n_clusters):
        if (label_pesodu == i).sum() > 15:
            cluster_list.append(i)

    fig_list = []
    for i in cluster_list:
        kmeans_mask = label_pesodu == i
        hull = spt.ConvexHull(points=ins_emb[kmeans_mask])
        fig_list.append(
            # go.Scatter3d(
            go.Scatter(
                x=ins_emb[kmeans_mask][hull.vertices][:, 0],
                y=ins_emb[kmeans_mask][hull.vertices][:, 1],
                fill="toself",
                opacity=0.2,
            )
        )

    # import_fea_every_clu = shap_values.mean(axis=1).argsort(axis=1)[:,::-1]
    str_import_fea_every_clu = [
        'C{}'.format(i) for i in range(cluster_centers.shape[0])
        ]

    # ins emb
    fig_list.append(
        go.Scatter(
            x=ins_emb[:, 0],
            y=ins_emb[:, 1],
            mode="markers",
            name="",
            text=label_pesodu,
            marker_line_width=0,
            marker=dict(
                size=[5] * ins_emb.shape[0],
                color=label_pesodu + 1,
            ),
        )
    )

    fig_list.append(
        go.Scatter(
            x=cluster_centers[:, 0],
            y=cluster_centers[:, 1],
            # z=np.zeros(shape=(cluster_centers.shape[0])),
            mode="markers+text",
            name="",
            text=str_import_fea_every_clu,
            # marker_line_width=0,
            marker=dict(
                symbol=["square"] * cluster_centers.shape[0],
                size=[25] * cluster_centers.shape[0],
                color="green",
            ),
        )
    )

    return fig_list


def Plot_subfig_2_1(global_importance, fea_list):
    importance = np.copy(global_importance)
    fea_name = fea_list

    list_xx = []
    list_yy = []
    for i in range(0, importance.shape[0]):
        index_x = np.argmax(importance)
        list_xx.append(importance[index_x])
        list_yy.append(fea_name[index_x])
        importance[index_x] = -1
    list_xx = np.array(list_xx)
    list_yy = np.array(list_yy)

    print(list_xx)
    print(list_yy)

    bar = go.Bar(
        x=list_xx,
        y=list_yy,
        text=["imp:{}".format(str(i)[:6]) for i in list_xx],
        orientation="h",
        marker=dict(
            color=[
                "rgb(250,{},{})".format(250 - int(a * 200), 250 - int(a * 200))
                for a in list_xx.tolist()
            ],
        ),
    )
    return [bar]


def Plot_subfig_2_2(
    shap_values,
    fea_list_all,
    # global_importance_raw,
    # fea_list,
    line_every_clu_to_fea=15,
):
    # from sklearn.preprocessing import MinMaxScaler
    def get_index_clu(clu_index):
        return clu_index + num_fea

    # def get_index_ins(ins_index):
    #     return ins_index+num_clu
    def get_index_fea(fea_index):
        return fea_index

    num_clu, num_ins, num_fea = shap_values.shape
    # num_fea = global_importance_raw.shape[0]
    # global_importance = np.copy(global_importance_raw)
    # global_importance[global_importance<0.1] = global_importance[global_importance>0.1].min()
    # global_importance = MinMaxScaler().fit_transform(global_importance[:, None])[:,0]

    label = []
    label += fea_list_all
    label += ["C{}".format(i) for i in range(num_clu)]
    # label += ['ins{}'.format(i) for i in range(num_ins)]
    nodecolor = []
    # nodecolor += ['rgb(255,0,255)' for i in range(num_ins)]
    nodecolor += ["blue" for i in range(num_fea)]
    nodecolor += ["green" for i in range(num_clu)]

    clu_fea_matrix = shap_values.mean(axis=1)
    clu_fea_matrix_index = clu_fea_matrix.argsort(axis=1)[:, ::-1]
    clu_fea_matrix_norm = (
        MinMaxScaler()
        .fit_transform(clu_fea_matrix.reshape(-1, 1))
        .reshape(clu_fea_matrix.shape)
    )

    source = {}
    target = {}
    value = {}
    linkcolor = {}

    for i in range(num_clu):
        source[i] = [get_index_clu(i)] * line_every_clu_to_fea
        target[i] = [
            get_index_fea(ins)
            for ins in clu_fea_matrix_index[i, :line_every_clu_to_fea].tolist()
        ]
        value[i] = [
            clu_fea_matrix_norm[i, ins]
            for ins in clu_fea_matrix_index[i, :line_every_clu_to_fea].tolist()
        ]
        linkcolor[i] = [
            "rgb({},{},255)".format(
                250 - int(150 * clu_fea_matrix_norm[i, ins]),
                50 + int(150 * clu_fea_matrix_norm[i, ins]),
            )
            for ins in clu_fea_matrix_index[i, :line_every_clu_to_fea].tolist()
        ]

    source_all = []
    target_all = []
    value_all = []
    linkcolor_all = []
    for i in range(num_clu):
        source_all += source[i]
        target_all += target[i]
        value_all += value[i]
        linkcolor_all += linkcolor[i]

    sankey_list = [
        go.Sankey(
            valueformat=".1f",
            valuesuffix=" ",
            node=dict(
                pad=15,
                thickness=20,
                line=dict(color="black", width=0.5),
                label=label,
                color=nodecolor,
            ),
            link=dict(
                source=source_all,
                target=target_all,
                value=value_all,
                color=linkcolor_all,
            ),
            visible=True,
        )
    ]
    for i in range(num_clu):
        sankey_list.append(
            go.Sankey(
                valueformat=".1f",
                valuesuffix=" ",
                node=dict(
                    pad=15,
                    thickness=20,
                    line=dict(color="black", width=0.5),
                    label=label,
                    color=nodecolor,
                ),
                link=dict(
                    source=source[i],
                    target=target[i],
                    value=value[i],
                    color=linkcolor[i],
                ),
                visible=False,
            )
        )
    return sankey_list


def Plot_case_study(
    data,
    mask,
    pix,
    num_s_shap,
    shap_values_abs,
    shap_values,
    label_pesodu,
    ):

    # top_k_import = 150
    label_pesodu_c = label_pesodu[:shap_values.shape[1]]
    # data_after_mask = data[:, mask]
    if pix > 0:
        dict_fig = {}
        from plotly.subplots import make_subplots
        print('label:', label_pesodu)
        for j in range(10):
            fig_img2 = make_subplots(rows=1, cols=10)
            for i in range(10):

                mask_cluster_eq = (label_pesodu_c == j)
                importance_c = shap_values[i, mask_cluster_eq, :].mean(axis=0)
                importance_c[~mask] = None
                importance_c[importance_c > 0.0009] = 0.0009
                importance_c[importance_c < -0.0009] = -0.0009
                f = px.imshow(
                    importance_c.reshape((pix, pix))[::-1],
                    color_continuous_midpoint=0.0,
                )

                fig_img2.add_traces(
                        f.data,
                        rows=[1] * len(f.data),
                        cols=[i+1] * len(f.data),
                    )
            dict_fig["b/imshow{}".format(j)] = fig_img2

        fig_img2 = make_subplots(rows=1, cols=10)
        for i in range(10):
            # mask_cluster_eq = (label_pesodu_c == j)
            importance_c = shap_values[i, :, :].mean(axis=0)
            importance_c[~mask] = None
            f = px.imshow(
                importance_c.reshape((pix, pix))[::-1],
            )

            fig_img2.add_traces(
                    f.data,
                    rows=[1] * len(f.data),
                    cols=[i+1] * len(f.data),
                )
        dict_fig["b/imshowall"] = fig_img2
        wandb.log(dict_fig)


def ShowEmbIns(ins_emb, pat_emb, pat_emb_neg, label, index=None, str_1="fea"):

    wandb_logs = {}
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=ins_emb[:, 0],
            y=ins_emb[:, 1],
            mode="markers",
            name="",
            text=index,
            marker=dict(
                size=[10] * ins_emb.shape[0],
                color=label + 1,
            )
            # color=np.array(labelstr[i])[index]
        )
    )
    fig.add_trace(
        go.Scatter(
            x=pat_emb[:, 0],
            y=pat_emb[:, 1],
            mode="markers+text",
            name="feature emb",
            text=["p{}".format(i) for i in range(pat_emb.shape[0])]
            if str_1 == "pat"
            else ["f{}".format(i) for i in range(pat_emb.shape[0])],
            marker=dict(
                size=[20] * pat_emb.shape[0],
                color=["red"] * pat_emb.shape[0],
            ),
            textfont=dict(
                # family="sans serif",
                # size=18,
                color="#ffffff"
            ),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=pat_emb_neg[:, 0],
            y=pat_emb_neg[:, 1],
            mode="markers+text",
            name="feature emb",
            text=["p{}".format(i) for i in range(pat_emb.shape[0])],
            marker=dict(
                size=[20] * pat_emb.shape[0],
                color=["blue"] * pat_emb_neg.shape[0],
            ),
            textfont=dict(
                # family="sans serif",
                # size=18,
                color="#ffffff"
            ),
        )
    )
    wandb_logs["vis_meta/emb_" + str_1] = fig

    return wandb_logs


def ShowEmbInsN(ins_emb, pat_emb, label):

    wandb_logs = {}
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=ins_emb[:, 0],
            y=ins_emb[:, 1],
            mode="markers",
            name="",
            marker=dict(
                size=[10] * ins_emb.shape[0],
                color=label + 1,
            )
            # color=np.array(labelstr[i])[index]
        )
    )
    fig.add_trace(
        go.Scatter(
            x=pat_emb[:, 0],
            y=pat_emb[:, 1],
            mode="markers",
            name="feature emb",
            marker=dict(size=[20] * pat_emb.shape[0], color=["red"] * pat_emb.shape[0]),
        )
    )
    wandb_logs["vis_meta/emb_neg"] = fig

    return wandb_logs


def ShowEmbInsColored_dence_fea(ins_emb, fea_or_pat, mask_feature, str_1="fea"):

    wandb_logs = {}
    index_feaute_list = np.where(mask_feature.cpu().numpy() > 0)[0]
    for ii in range(index_feaute_list.shape[0]):
        # if ii > 16:
        #     break
        index_f = index_feaute_list[ii]

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=ins_emb[:, 0],
                y=ins_emb[:, 1],
                mode="markers",
                name="Ins emb",
                marker=dict(
                    size=[10] * ins_emb.shape[0],
                    color=fea_or_pat[:, index_f],
                    colorscale="bluered",
                ),
            )
        )

        wandb_logs["vis_emb{}/{}_{}".format(str_1, str_1, str(index_f))] = fig

    return wandb_logs


def ShowEmbInsColored_dence_pat(ins_emb, fea_or_pat, str_1="fea"):

    wandb_logs = {}
    # index_feaute_list = np.where(mask_feature.cpu().numpy()>0)[0]
    for ii in range(fea_or_pat.shape[1]):
        # if ii > 16:
        #     break
        index_f = ii

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=ins_emb[:, 0],
                y=ins_emb[:, 1],
                mode="markers",
                name="Ins emb",
                marker=dict(
                    size=[10] * ins_emb.shape[0],
                    color=fea_or_pat[:, index_f],
                    colorscale="bluered",
                ),
            )
        )

        wandb_logs["vis_emb{}/{}_{}".format(str_1, str_1, str(index_f))] = fig

    return wandb_logs


def ShowEmbIns_WithTrack(
    ins_emb, fea_package, pat_package, label, mask, index=None, str_1="fea", line=False
):
    from plotly.subplots import make_subplots

    fea_emb, fea_emb_neg, fea_track_list = fea_package
    pat_emb, pat_emb_neg, pat_track_list = pat_package
    assert ins_emb.shape[1] == 2

    wandb_logs = {}
    # for i in range(pat_emb.shape[0]):

    for i in range(mask.shape[1]):
        fig = make_subplots(rows=1, cols=2)
        # fig_pat = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=ins_emb[:, 0],
                y=ins_emb[:, 1],
                mode="markers",
                name="",
                text=index,
                marker=dict(
                    size=[10] * ins_emb.shape[0],
                    color=label + 1,
                )
                # color=np.array(labelstr[i])[index]
            ),
            row=1,
            col=1,
        )
        mask_item = mask[:, i].cpu().numpy()
        fig.add_trace(
            go.Scatter(
                x=fea_emb[mask_item, 0],
                y=fea_emb[mask_item, 1],
                mode="markers+text",
                name="feature emb",
                text=["f{}".format(int(i)) for i in np.where(mask_item > 0)[0]],
                marker=dict(
                    size=[20] * fea_emb.shape[0],
                    color=["red"] * fea_emb.shape[0],
                ),
                textfont=dict(color="#ffffff"),
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=fea_emb_neg[mask_item, 0],
                y=fea_emb_neg[mask_item, 1],
                mode="markers+text",
                name="feature emb",
                # text=['f{}'.format(i) for i in range(fea_emb.shape[0])],
                text=["f{}".format(int(i)) for i in np.where(mask_item > 0)[0]],
                marker=dict(
                    size=[20] * fea_emb.shape[0],
                    color=["blue"] * fea_emb_neg.shape[0],
                ),
                textfont=dict(color="#ffffff"),
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=ins_emb[:, 0],
                y=ins_emb[:, 1],
                mode="markers",
                name="",
                text=index,
                marker=dict(
                    size=[10] * ins_emb.shape[0],
                    color=label + 1,
                ),
            ),
            row=1,
            col=2,
        )
        fig.add_trace(
            go.Scatter(
                x=pat_emb[:, 0],
                y=pat_emb[:, 1],
                mode="markers+text",
                name="feature emb",
                text=["p{}".format(i) for i in range(pat_emb.shape[0])],
                marker=dict(
                    size=[20] * pat_emb.shape[0],
                    color=["red"] * pat_emb.shape[0],
                ),
                textfont=dict(color="#ffffff"),
            ),
            row=1,
            col=2,
        )
        fig.add_trace(
            go.Scatter(
                x=pat_emb_neg[:, 0],
                y=pat_emb_neg[:, 1],
                mode="markers+text",
                name="feature emb",
                text=["p{}".format(i) for i in range(pat_emb.shape[0])],
                marker=dict(
                    size=[20] * pat_emb.shape[0],
                    color=["blue"] * pat_emb_neg.shape[0],
                ),
                textfont=dict(color="#ffffff"),
            ),
            row=1,
            col=2,
        )
        if line:
            fig.add_trace(
                go.Scatter(
                    x=pat_track_list[i][:, 0],
                    y=pat_track_list[i][:, 1],
                    mode="lines",
                    line=go.scatter.Line(),
                    showlegend=False,
                    marker=dict(size=[100] * len(pat_track_list[0])),
                ),
                row=1,
                col=2,
            )
            # for i in range(len(pat_track_list)):
            for j in range(mask.shape[0]):
                if mask[j, i]:
                    fig.add_trace(
                        go.Scatter(
                            x=fea_track_list[j][:, 0],
                            y=fea_track_list[j][:, 1],
                            mode="lines",
                            line=go.scatter.Line(),
                            showlegend=False,
                            marker=dict(size=[100] * len(fea_track_list[0])),
                        ),
                        row=1,
                        col=1,
                    )

        wandb_logs["vis_meta/emb_trac_pat_" + str(i)] = fig
        # wandb_logs["vis_meta/emb_trac_pat_"+str(i)] = fig_pat

    return wandb_logs


def ShowSankey(target_, pat_num, drop_feature=True):
    if drop_feature:
        label_ = ["f" + str(i) for i in range(len(target_))]
        label_.extend(["p" + str(i) for i in range(pat_num)])
        source = np.arange(len(target_))
        source = [
            source[i]
            for i in range(len(source))
            if target_[i] != len(target_) + pat_num
        ]
        target = [t for t in target_ if t != len(target_) + pat_num]
        wandb_logs = {}
        fig = go.Figure(
            data=[
                go.Sankey(
                    node=dict(
                        pad=15,
                        thickness=20,
                        line=dict(color="black", width=0.5),
                        label=[],
                        color="blue",
                    ),
                    link=dict(
                        source=source,
                        target=target,
                        value=[1 for i in range(len(label_))],
                    ),
                )
            ]
        )
        wandb_logs["Sankey"] = fig
        return wandb_logs
    else:
        label_ = ["f" + str(i) for i in range(len(target_))]
        label_.extend(["p" + str(i) for i in range(pat_num)])
        label_.append("sorted")
        wandb_logs = {}
        fig = go.Figure(
            data=[
                go.Sankey(
                    node=dict(
                        pad=15,
                        thickness=20,
                        line=dict(color="black", width=0.5),
                        label=[],
                        color="blue",
                    ),
                    link=dict(
                        source=np.arange(len(target_)),
                        target=target_,
                        value=[1 for i in range(len(label_))],
                    ),
                )
            ]
        )
        wandb_logs["Sankey"] = fig
        return wandb_logs


def ShowSankey_Zelin(mask):

    mask_np = mask.detach().cpu().numpy()
    index_x, index_y = np.where(mask_np > 0.5)
    index_x_nodup = np.unique(index_x)
    index_y_nodup = np.unique(index_y)
    index_x_no_dup_sort = np.argsort(index_x_nodup).tolist()
    index_y_no_dup_sort = np.argsort(index_y_nodup).tolist()
    dict_absolute_index_to_relative_index_x = dict(
        zip(index_x_nodup.tolist(), index_x_no_dup_sort)
    )
    dict_absolute_index_to_relative_index_y = dict(
        zip(index_y_nodup.tolist(), index_y_no_dup_sort)
    )

    fig = go.Figure(
        data=[
            go.Sankey(
                node=dict(
                    pad=15,
                    thickness=20,
                    line=dict(color="black", width=0.5),
                    label=["f" + str(i) for i in index_x_nodup]
                    + ["p" + str(i) for i in index_y_nodup],
                    color="blue",
                ),
                link=dict(
                    source=[
                        dict_absolute_index_to_relative_index_x[i] for i in index_x
                    ],  # indices correspond to labels, eg A1, A2, A1, B1, ...
                    target=[
                        dict_absolute_index_to_relative_index_y[i]
                        + len(index_x_no_dup_sort)
                        for i in index_y
                    ],
                    value=[1] * index_x.shape[0],
                ),
            )
        ]
    )
    return {"SankeyPlot": fig}


def ShowSankey_Zelin_return_fig(
    shap_values,
    label_pesodu,
    global_importance_raw,
    line_every_clu_to_fea=15,
):
    from sklearn.preprocessing import MinMaxScaler

    def get_index_clu(clu_index):
        return clu_index + num_fea

    # def get_index_ins(ins_index):
    #     return ins_index+num_clu
    def get_index_fea(fea_index):
        return fea_index

    num_clu, num_ins, num_fea = shap_values.shape
    global_importance = np.copy(global_importance_raw)
    global_importance[global_importance < 0.1] = global_importance[
        global_importance > 0.1
    ].min()
    global_importance = MinMaxScaler().fit_transform(global_importance[:, None])[:, 0]

    label = []
    label += ["f{}".format(i) for i in range(num_fea)]
    label += ["Cluster {}".format(i) for i in range(num_clu)]
    # label += ['ins{}'.format(i) for i in range(num_ins)]
    nodecolor = []
    # nodecolor += ['rgb(255,0,255)' for i in range(num_ins)]
    nodecolor += [
        "rgb(250,{},{})".format(250 - int(a * 200), 250 - int(a * 200))
        for a in global_importance.tolist()
    ]
    nodecolor += ["green" for i in range(num_clu)]

    clu_fea_matrix = shap_values.mean(axis=1)
    clu_fea_matrix_index = clu_fea_matrix.argsort(axis=1)[:, ::-1]
    clu_fea_matrix_norm = MinMaxScaler().fit_transform(clu_fea_matrix)

    source = {}
    target = {}
    value = {}
    linkcolor = {}

    for i in range(num_clu):
        source[i] = [get_index_clu(i)] * line_every_clu_to_fea
        target[i] = [
            get_index_fea(ins)
            for ins in clu_fea_matrix_index[i, :line_every_clu_to_fea].tolist()
        ]
        value[i] = [
            1000 * clu_fea_matrix[i, ins]
            for ins in clu_fea_matrix_index[i, :line_every_clu_to_fea].tolist()
        ]
        linkcolor[i] = [
            "rgb({},{},255)".format(
                250 - int(150 * clu_fea_matrix_norm[i, ins]),
                50 + int(150 * clu_fea_matrix_norm[i, ins]),
            )
            for ins in clu_fea_matrix_index[i, :line_every_clu_to_fea].tolist()
        ]

    source_all = []
    target_all = []
    value_all = []
    linkcolor_all = []
    for i in range(num_clu):
        source_all += source[i]
        target_all += target[i]
        value_all += value[i]
        linkcolor_all += linkcolor[i]

    sankey_list = [
        go.Sankey(
            valueformat=".1f",
            valuesuffix=" ",
            node=dict(
                pad=15,
                thickness=20,
                line=dict(color="black", width=0.5),
                label=label,
                color=nodecolor,
            ),
            link=dict(
                source=source_all,
                target=target_all,
                value=value_all,
                color=linkcolor_all,
            ),
            visible=True,
        )
    ]
    for i in range(num_clu):
        sankey_list.append(
            go.Sankey(
                valueformat=".1f",
                valuesuffix=" ",
                node=dict(
                    pad=15,
                    thickness=20,
                    line=dict(color="black", width=0.5),
                    label=label,
                    color=nodecolor,
                ),
                link=dict(
                    source=source[i],
                    target=target[i],
                    value=value[i],
                    color=linkcolor[i],
                ),
                visible=False,
            )
        )
    return sankey_list


def Show_global_importance_Zelin_return_fig(p_m):

    imp_save = np.copy(p_m)

    if len(p_m.shape) <= 1:
        y = p_m
    else:
        y = p_m.sum(axis=1)

    p_m = MinMaxScaler().fit_transform(p_m[:, None])[:, 0]

    list_x = []
    list_y = []
    # color_y = []
    for i in range(y.shape[0]):
        if y[i] > 0.1:
            list_y.append("f{}".format(i))
            list_x.append(y[i])
    list_x = np.array(list_x)
    list_y = np.array(list_y)

    list_x = MinMaxScaler().fit_transform(list_x[:, None])[:, 0]

    bar = go.Bar(
        x=list_x,
        y=list_y,
        text=["imp:{}".format(str(i)[:6]) for i in imp_save],
        orientation="h",
        marker=dict(
            color=[
                "rgb(250,{},{})".format(250 - int(a * 200), 250 - int(a * 200))
                for a in list_x.tolist()
            ],
        ),
    )

    return bar


def ShowEmbInsFeaPat(
    ins_emb,
    pat_emb,
    pat_emb_neg,
    fea_emb,
    fea_emb_neg,
    label,
    mask=None,
    index=None,
    str_1="fea",
):

    wandb_logs = {}
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=ins_emb[:, 0],
            y=ins_emb[:, 1],
            mode="markers",
            name="",
            text=index,
            marker=dict(
                size=[10] * ins_emb.shape[0],
                color=label + 1,
            )
            # color=np.array(labelstr[i])[index]
        )
    )

    fig.add_trace(
        go.Scatter(
            x=pat_emb[:, 0],
            y=pat_emb[:, 1],
            mode="markers+text",
            name="",
            text=["p{}".format(i) for i in range(pat_emb.shape[0])],
            marker=dict(
                size=[20] * pat_emb.shape[0],
                color=["red"] * pat_emb.shape[0],
            ),
            textfont=dict(color="#ffffff"),
        )
    )
    # fig.add_trace(
    #     go.Scatter(
    #         x=pat_emb_neg[:, 0], y=pat_emb_neg[:, 1],
    #         mode='markers+text',
    #         name='',
    #         text=['p{}'.format(i) for i in range(pat_emb.shape[0])],
    #         marker=dict(
    #             size=[20]*pat_emb.shape[0],
    #             color=['blue']*pat_emb_neg.shape[0],
    #         ),
    #         textfont=dict(
    #             # family="sans serif",
    #             # size=18,
    #             color="#ffffff"
    #         )
    #     )
    #     )

    fea_mask = (mask.sum(dim=1) > 0.5).detach().cpu().numpy()
    # print(fea_mask)
    fea_index_str = ["f{}".format(i) for i in range(fea_emb.shape[0])]
    fea_index_str_use = []
    for i in range(len(fea_index_str)):
        if fea_mask[i]:
            fea_index_str_use.append(fea_index_str[i])

    fig.add_trace(
        go.Scatter(
            x=fea_emb[:, 0][fea_mask],
            y=fea_emb[:, 1][fea_mask],
            mode="markers+text",
            name="fea emb",
            text=fea_index_str_use,
            marker=dict(
                size=[15] * fea_emb.shape[0],
                color=["blue"] * fea_emb.shape[0],
                symbol="square",
            ),
            textfont=dict(
                # family="sans serif",
                # size=18,
                color="#ffffff"
            ),
        )
    )

    line_x_list = []
    line_y_list = []
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if mask[i, j]:
                x_c = [pat_emb[j][0], fea_emb[i][0], None]
                y_c = [pat_emb[j][1], fea_emb[i][1], None]
                line_x_list += x_c
                line_y_list += y_c

    fig.add_trace(
        go.Scatter(
            x=line_x_list,
            y=line_y_list,
            mode="lines",
            opacity=0.5,
            textfont=dict(color="#ffffff"),
        )
    )
    wandb_logs["vis_meta/emb_ins_fea_pat" + str_1] = fig

    return wandb_logs


def gpu2np(a):
    return a.cpu().detach().numpy()


def cluster_kmeans(data, n_clusters):
    KMeans_model = KMeans(
        n_clusters=n_clusters,
        # assign_labels="discretize",
        # random_state=0,
        # eigen_solver="arpack",
        # affinity="nearest_neighbors",
        n_jobs=-1,
    ).fit(data)
    return KMeans_model.labels_


def rotate_fea(fea, ins, mask):

    from scipy.spatial.transform import Rotation as R

    dis_list = []
    for i in range(360):
        theta = np.radians(i)
        c, s = np.cos(theta), np.sin(theta)
        r = np.array(((c, -s), (s, c)))
        dis_list.append(pairwise_distances(np.dot(fea, r), ins)[mask].sum())
    dis_list = np.array(dis_list)

    best_i = np.argmax(dis_list)
    theta = np.radians(best_i)
    c, s = np.cos(theta), np.sin(theta)
    best_r = np.array(((c, -s), (s, c)))
    print("best_i")
    print(best_i)
    return np.dot(fea, best_r), best_r


def feag_cluster_mask(
    shap_values, mask, fea_cluster_centers, fea_label_pesodu, n_feverycluset
):
    shap_values_fea_pat = shap_values.mean(axis=1).T
    shap_values_fea_pat = shap_values_fea_pat[gpu2np(mask)]
    shap_values_fea_group = []
    for i in range(fea_cluster_centers.shape[0]):
        shap_values_fea_group.append(
            shap_values_fea_pat[fea_label_pesodu == i].mean(axis=0, keepdims=1)
        )
    shap_values_fea_group = np.concatenate(shap_values_fea_group)
    shap_values_fea_group_topk = np.sort(shap_values_fea_group, axis=0)[::-1][
        n_feverycluset
    ]
    mask_fea_pat = shap_values_fea_group > shap_values_fea_group_topk
    return mask_fea_pat, shap_values_fea_group


def show_local_expl(
    data,
    ins_emb,
    model,
    global_importance_raw,
    n_feverycluset=15,
    n_clusters=10,
    row=1,
    col=2,
    num_s_shap=5,
    pix=8,
):

    from sklearn.cluster import KMeans
    import shap

    fig_list = []

    KMeans_model = KMeans(
        n_clusters=n_clusters,
        n_jobs=-1,
    ).fit(ins_emb)
    label_pesodu = KMeans_model.labels_
    cluster_centers = KMeans_model.cluster_centers_

    fea_emb_all = umap.UMAP().fit_transform(data.T[gpu2np(model.mask)])
    fea_KMeans_model = KMeans(
        n_clusters=n_clusters,
        n_jobs=-1,
    ).fit(fea_emb_all)
    fea_label_pesodu = fea_KMeans_model.labels_
    fea_cluster_centers = fea_KMeans_model.cluster_centers_

    model.cluster_rescale = np.zeros(shape=(cluster_centers.shape[0]))
    model.cluster_centers = torch.tensor(cluster_centers).to(model.mask.device)
    data_after_mask = data[:, gpu2np(model.mask)]

    model.forward = model.predict_lime_g

    explainer = shap.GradientExplainer(
        model,
        torch.tensor(data).to(model.mask.device),
    )

    shap_values = explainer.shap_values(
        torch.tensor(data).to(model.mask.device)[0:num_s_shap]
    )
    shap_values = np.abs(np.array(shap_values))

    shap_values_fea_ins = shap_values.mean(axis=0).T[gpu2np(model.mask)]
    fea_most_import_ins_index = np.argsort(
        shap_values_fea_ins,
        axis=1,
    )[:, -2:]
    fake_ins_for_fea = [
        data[fea_most_import_ins_index[i][0:1]] * 0.2
        + data[fea_most_import_ins_index[i][1:2]] * 0.8
        for i in range(fea_most_import_ins_index.shape[0])
    ]
    fake_ins_for_fea = np.concatenate(fake_ins_for_fea)

    fea_emb = model.forward_fea(torch.tensor(fake_ins_for_fea))[2]
    fea_emb = gpu2np(fea_emb)

    mask_fea_pat, shap_values_fea_group = feag_cluster_mask(
        shap_values, model.mask, fea_cluster_centers, fea_label_pesodu, n_feverycluset
    )
    print("finish local exp")

    for i in range(n_clusters):
        kmeans_mask = label_pesodu == i
        hull = spt.ConvexHull(points=ins_emb[kmeans_mask])
        fig_list.append(
            # go.Scatter3d(
            go.Scatter(
                x=ins_emb[kmeans_mask][hull.vertices][:, 0],
                y=ins_emb[kmeans_mask][hull.vertices][:, 1],
                fill="toself",
                opacity=0.3,
            )
        )

    import_fea_every_clu = shap_values.mean(axis=1).argsort(axis=1)[:, ::-1]
    str_import_fea_every_clu = [
        "Cluster {}:".format(i)
        # + ','.join(
        # import_fea_every_clu[i].astype(np.str).tolist()
        # )[:50]
        for i in range(import_fea_every_clu.shape[0])
    ]
    fig_list.append(
        go.Scatter(
            x=cluster_centers[:, 0],
            y=cluster_centers[:, 1],
            # z=np.zeros(shape=(cluster_centers.shape[0])),
            mode="markers+text",
            name="",
            text=str_import_fea_every_clu,
            # marker_line_width=0,
            marker=dict(
                symbol=["x-dot"] * cluster_centers.shape[0],
                size=[25] * cluster_centers.shape[0],
                color="green",
            ),
            # color=np.array(labelstr[i])[index]
        )
        # row=1, col=2,
    )

    # ins emb
    fig_list.append(
        go.Scatter(
            x=ins_emb[:, 0],
            y=ins_emb[:, 1],
            # z=np.zeros(shape=(ins_emb.shape[0])),
            mode="markers",
            name="",
            text=label_pesodu,
            marker_line_width=0,
            marker=dict(
                size=[5] * ins_emb.shape[0],
                color=label_pesodu + 1,
            ),
            # color=np.array(labelstr[i])[index]
        )
        # row=1, col=2,
    )
    # cluster_centers emb

    fea_imort_color = []
    global_importance = np.copy(global_importance_raw)
    global_importance[global_importance < 0.1] = global_importance[
        global_importance > 0.1
    ].min()
    global_importance = MinMaxScaler().fit_transform(global_importance[:, None])[:, 0]
    fea_imort_color += [
        "rgb(250,{},{})".format(250 - int(a * 200), 250 - int(a * 200))
        for a in global_importance.tolist()
    ]

    # fea emb
    fea_name = np.array(["f{}".format(i) for i in range(gpu2np(model.mask).shape[0])])
    fig_list.append(
        # go.Scatter3d(
        go.Scatter(
            x=fea_emb[:, 0],
            y=fea_emb[:, 1],
            # z=[0]*fea_emb.shape[0],
            mode="markers+text",
            name="",
            # text=fea_emb_name,
            text=fea_name[gpu2np(model.mask)],
            marker_line_width=0,
            marker=dict(
                size=[15] * fea_emb.shape[0],
                symbol=["square"] * fea_emb.shape[0],
                color=fea_imort_color,
            ),
        )
    )

    if pix > 1000:
        dict_fig = {}
        for s in range(min(num_s_shap, 2)):
            data_exp = data_after_mask[s]
            data_exp_show = np.zeros(data.shape[1])
            data_exp_show[gpu2np(model.mask)] = data_exp
            data_exp_show[~gpu2np(model.mask)] = None
            fig_img = go.Figure()
            fig_img.add_trace(go.Heatmap(z=data_exp_show.reshape(pix, pix)[::-1]))

            # importance_abs = np.zeros(data.shape[1])
            # importance = np.zeros(data.shape[1])
            importance = shap_values[:, s, :].mean(axis=0)
            importance_abs = np.abs(shap_values[:, s, :]).mean(axis=0)

            m_high_import = importance_abs > np.sort(importance_abs)[::-1][15]
            importance_mask = importance[m_high_import]
            importance_abs_mask = importance_abs[m_high_import]
            imp_index = np.where(m_high_import)[0]
            important_feature_list_x = imp_index % pix
            important_feature_list_y = (pix - 1) - imp_index // pix
            fig_img.add_trace(
                go.Scatter(
                    x=important_feature_list_x,
                    y=important_feature_list_y,
                    mode="markers",
                    text=[
                        "f" + str(int(a)) + "_" + str(b)[:6]
                        for (a, b) in zip(imp_index, imp_index)
                    ],
                    marker=dict(
                        size=15
                        * np.abs(importance_abs_mask)
                        / np.abs(importance_abs_mask).min(),
                        color=[
                            "red" if importance_mask[i] > 0 else "green"
                            for i in range(importance_mask.shape[0])
                        ],
                    ),
                )
            )
            dict_fig["imshow{}".format(s)] = fig_img
        wandb.log(dict_fig)
    return fig_list, mask_fea_pat, shap_values, label_pesodu


def ShowEmbInsFeaPat_returen_fig(
    ins_emb,
    pat_emb,
    pat_emb_neg,
    fea_emb,
    fea_emb_neg,
    label,
    fig,
    row=1,
    col=2,
    mask=None,
    index=None,
    str_1="fea",
):

    fig.add_trace(
        go.Scatter(
            x=ins_emb[:, 0],
            y=ins_emb[:, 1],
            mode="markers",
            name="",
            text=index,
            marker_line_width=0,
            marker=dict(
                size=[5] * ins_emb.shape[0],
                color=label + 1,
            ),
            # color=np.array(labelstr[i])[index]
        ),
        row=row,
        col=col,
    )

    fig.add_trace(
        go.Scatter(
            x=pat_emb[:, 0],
            y=pat_emb[:, 1],
            mode="markers+text",
            name="",
            text=["p{}".format(i) for i in range(pat_emb.shape[0])],
            marker_line_width=0,
            marker=dict(
                size=[20] * pat_emb.shape[0],
                color=["red"] * pat_emb.shape[0],
            ),
            textfont=dict(color="#ffffff"),
        ),
        row=row,
        col=col,
    )

    fea_mask = (mask.sum(dim=1) > 0.5).detach().cpu().numpy()
    # print(fea_mask)
    fea_index_str = ["f{}".format(i) for i in range(fea_emb.shape[0])]
    fea_index_str_use = []
    for i in range(len(fea_index_str)):
        if fea_mask[i]:
            fea_index_str_use.append(fea_index_str[i])

    fig.add_trace(
        go.Scatter(
            x=fea_emb[:, 0][fea_mask],
            y=fea_emb[:, 1][fea_mask],
            mode="markers+text",
            name="fea emb",
            text=fea_index_str_use,
            marker_line_width=0,
            marker=dict(
                size=[20] * fea_emb.shape[0],
                color=["green"] * fea_emb.shape[0],
                symbol="square",
            ),
            textfont=dict(
                # family="sans serif",
                # size=18,
                color="#ffffff"
            ),
        ),
        row=row,
        col=col,
    )

    line_x_list = []
    line_y_list = []
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if mask[i, j]:
                x_c = [pat_emb[j][0], fea_emb[i][0], None]
                y_c = [pat_emb[j][1], fea_emb[i][1], None]
                line_x_list += x_c
                line_y_list += y_c

    fig.add_trace(
        go.Scatter(
            x=line_x_list,
            y=line_y_list,
            mode="lines",
            opacity=0.5,
            textfont=dict(color="#ffffff"),
        ),
        row=row,
        col=col,
    )
    # wandb_logs["vis_meta/emb_ins_fea_pat"+str_1] = fig

    return fig


def subfig_12_arrow_flows(
    cf, img_from, cf_emb, model, img_from_emb, text="", show_bool=False
):

    fig_flow = []
    flow_end = []
    change_dict_list = []
    loop_emb_his_np_for_cf_0 = []
    loop_emb_his_np_for_cf_1 = []
    for cf_index in range(cf.shape[0]):
        img_from_c = torch.tensor(img_from)[cf_index : cf_index + 1]
        cf_c = torch.tensor(cf)[cf_index : cf_index + 1]
        cf_c_emb = torch.tensor(cf_emb[cf_index : cf_index + 1])
        change_bool = (((img_from_c - cf_c) != 0).int() + model.mask.int()) > 1
        loop_emb_his_0 = [torch.tensor(img_from_emb)]
        loop_emb_his_1 = []
        change_dict = {}
        for loop in range(change_bool.sum()):
            cf_tem = []
            for i in range(change_bool.shape[1]):
                img_from_tem = torch.clone(img_from_c)
                img_from_tem[:, i] = cf_c[:, i]
                cf_tem.append(img_from_tem)
            cf_tem = torch.cat(cf_tem)
            cf_tem_emb = model.forward_exp(cf_tem)[2]
            dis_tem_to_cf_c = model.pdist2(cf_tem_emb, cf_c_emb)
            dis_tem_to_cf_c[~change_bool[0]] = dis_tem_to_cf_c.max() + 1
            if dis_tem_to_cf_c.min() < 0.01:
                break
            best_index = dis_tem_to_cf_c.argmin()
            change_dict[int(best_index)] = [
                float(img_from_c[:, best_index]),
                float(cf_c[:, best_index]),
            ]

            img_from_c = cf_tem[best_index : (best_index + 1)]
            change_bool[0, best_index] = False
            loop_emb_his_0.append(cf_tem_emb[best_index : (best_index + 1)])
            loop_emb_his_1.append(loop_emb_his_0[loop + 1] - loop_emb_his_0[loop])
        loop_emb_his_1.append(cf_c_emb - loop_emb_his_0[-1])
        loop_emb_his_0_np = gpu2np(torch.cat(loop_emb_his_0))[:-1]
        loop_emb_his_1_np = gpu2np(torch.cat(loop_emb_his_1))[:-1]
        loop_emb_his_np_for_cf_0.append(loop_emb_his_0_np)
        loop_emb_his_np_for_cf_1.append(loop_emb_his_1_np)
        change_dict_list.append(change_dict)
        flow_end.append(img_from_c)
    loop_emb_his_np_for_cf_0 = np.concatenate(loop_emb_his_np_for_cf_0)
    loop_emb_his_np_for_cf_1 = np.concatenate(loop_emb_his_np_for_cf_1)

    figff = ff.create_quiver(
        x=loop_emb_his_np_for_cf_0[:, 0],
        y=loop_emb_his_np_for_cf_0[:, 1],
        u=loop_emb_his_np_for_cf_1[:, 0],
        v=loop_emb_his_np_for_cf_1[:, 1],
        scale=1,
        arrow_scale=0.1,
        visible=show_bool,
        name=text,
    )
    fig_flow.append(figff.data[0])

    fig_flow.append(
        go.Scatter(
            mode="markers+text",
            name=text,
            x=img_from_emb[:, 0],
            y=img_from_emb[:, 1],
            # marker_line_width=0,
            text=["origin {}".format(i) for i in range(img_from_emb.shape[0])],
            marker=dict(
                size=[15] * img_from_emb.shape[0],
                color=["red"] * img_from_emb.shape[0],
            ),
            visible=show_bool,
        )
    )
    fig_flow.append(
        go.Scatter(
            mode="markers+text",
            name=text,
            x=cf_emb[:, 0],
            y=cf_emb[:, 1],
            text=["cf {}".format(i) for i in range(cf_emb.shape[0])],
            marker=dict(
                size=[15] * cf_emb.shape[0],
                color=["yellow"] * cf_emb.shape[0],
            ),
            visible=show_bool,
        )
    )
    return fig_flow, change_dict_list, np.concatenate(flow_end)


# import patemb_main_XAI


def Plot_subfig_1_3(model, ins_emb, cf_list, img_from_list, pix):
    fig_1 = []
    fig_1.append(
        go.Scatter(
            mode="markers",
            name="",
            x=ins_emb[:, 0],
            y=ins_emb[:, 1],
            marker_line_width=0,
            marker=dict(
                size=[5] * ins_emb.shape[0],
            ),
        )
    )

    change_dict_for_ij_list = []
    cf_flow_end_list = []
    show_bool = True
    for i in range(len(img_from_list)):
        cf_flow_end_list_c = []
        for j in range(len(img_from_list[0])):
            if i != j:
                cf = cf_list[i][j]
                img_from = img_from_list[i][j]
                img_from_emb = gpu2np(
                    model.forward_exp(
                        img_from.to(model.mask.device))[2]
                )
                cf_emb_ = gpu2np(
                    model.forward_exp(
                        torch.tensor(cf).to(model.mask.device))[2]
                )
                culster_index_list = gpu2np(
                    torch.argmin(
                        model.pdist2(
                            model.cluster_centers, torch.tensor(cf_emb_)),
                        dim=0,
                    )
                )
                culster_index = np.argmax(np.bincount(culster_index_list))
                cf_emb = np.concatenate(
                    img_from.shape[0]
                    * [gpu2np(model.cluster_centers[culster_index])[None, :]]
                )

                fig_flows, change_dict_for_ij, flow_end = subfig_12_arrow_flows(
                    cf,
                    img_from,
                    cf_emb,
                    model,
                    img_from_emb,
                    text="{}_{}".format(i, j),
                    show_bool=show_bool,
                )
                cf_flow_end_list_c.append(flow_end)
                if pix > 0:
                    patemb_main.plot_cf_figure(
                        flow_end, img_from, model, pix=pix, text="{}->{}".format(i, j)
                    )

                show_bool = False
                fig_1 += fig_flows
                change_dict_for_ij_list.append(change_dict_for_ij)
        cf_flow_end_list.append(cf_flow_end_list_c)

    return fig_1, change_dict_for_ij_list, cf_flow_end_list


def Plot_subfig_2_3(
    model, change_dict_for_ij_list, img_from_list, cf_list, n_clusters, fea_list_all
):

    fig_list = []
    c = 0
    for i in range(len(img_from_list)):
        for j in range(len(img_from_list[0])):
            if i != j:
                name = "{}_{}".format(i, j)
                img_from = img_from_list[i][j]
                change_dict_list = change_dict_for_ij_list[c]
                cf = cf_list[i][j]
                f = Plot_subfig_2_3_c(
                    model,
                    img_from,
                    change_dict_list,
                    cf,
                    name,
                    visible=name == "0_1",
                    fea_list_all=fea_list_all,
                )
                fig_list += f
                c += 1
    return fig_list


def Plot_subfig_2_3_c(
    model, img_from, change_dict_list, cf, name, visible, fea_list_all
):
    # fig_1 = []
    fig_2 = []

    str_use = []
    o_list = []
    feature_use_bool = gpu2np(model.mask) > 0
    cf_list = []
    for i in range(cf.shape[1]):
        if cf[0, i] != img_from[0, i] and feature_use_bool[i]:
            str_use.append("f{}".format(i))
            o_list.append(cf[0, i])
            cf_list.append(img_from[0, i])

    all_keys_index = []
    for i in range(len(change_dict_list)):
        all_keys_index += list(change_dict_list[i].keys())
    all_keys_index = list(set(all_keys_index))
    all_keys_str = [fea_list_all[i] for i in all_keys_index]

    import_abs_value = np.abs(img_from - cf).T.mean(axis=1)[all_keys_index]
    import_score = import_abs_value / import_abs_value.max()

    list_xx = []
    list_yy = []
    for i in range(0, import_score.shape[0]):
        index_x = np.argmax(import_score)
        list_xx.append(float(import_score[index_x]))
        list_yy.append(all_keys_str[index_x])
        import_score[index_x] = -1
    list_xx = np.array(list_xx)
    list_yy = np.array(list_yy)

    fig_2.append(
        go.Bar(
            x=list_xx.tolist(),
            y=list_yy.tolist(),
            visible=visible,
            text=["imp:{}".format(str(i)[:6]) for i in list_xx.tolist()],
            orientation="h",
            name=name,
            marker=dict(
                color=[
                    "rgb(250,{},{})".format(250 - int(a * 200), 250 - int(a * 200))
                    for a in list_xx
                ]
            ),
        )
    )

    return fig_2


def add_button_local(fig, sankey_index_sart, sankey_index_end, n_clusters):
    all_vis_list = np.array([True] * len(fig.data))
    all_vis_list[sankey_index_sart + 1 : sankey_index_end] = False
    buttons22 = [
        dict(label="ALl", method="update", args=[{"visible": all_vis_list}]),
    ]
    for i in range(n_clusters):
        all_vis_list_c = np.array(all_vis_list)
        all_vis_list_c[sankey_index_sart] = False
        all_vis_list_c[sankey_index_sart + 1 + i] = True
        buttons22.append(
            {
                "label": "Cluster {}".format(i),
                "method": "update",
                "args": [
                    {
                        "visible": all_vis_list_c,
                    }
                ],
            }
        )

    fig.update_layout(
        updatemenus=[
            dict(
                buttons=buttons22,
                x=0.4,
                y=0.35,
                xanchor="left",
                yanchor="top",
            )
        ]
    )
    return fig


def add_button_swich(fig, sankey_index_sart, sankey_index_end, n_clusters):
    all_vis_list = [fig.data[i].visible for i in range(len(fig.data))]
    # all_vis_list[sankey_index_sart:sankey_index_end] = False
    buttons1 = []
    # buttons2 = []

    index_fig = 0
    for i in range(n_clusters):
        for j in range(n_clusters):
            if i != j:
                all_vis_list_c = np.array(all_vis_list)
                all_vis_list_c[sankey_index_sart:sankey_index_end] = False
                index_s = index_fig
                all_vis_list_c[index_s : index_s + 3] = True
                buttons1.append(
                    {
                        "label": "C{}->C{}".format(i, j),
                        "method": "update",
                        "args": [
                            {
                                "visible": all_vis_list_c,
                            }
                        ],
                    }
                )
                index_fig += 3

    fig.update_layout(
        updatemenus=[
            dict(
                buttons=buttons1,
                x=0.8,
                y=0.35,
                xanchor="left",
                yanchor="top",
            )
        ]
    )
    return fig


def add_button(
    fig, sankey_index_sart, sankey_index_end, flow_sart, flow_end, n_clusters
):
    all_vis_list = [fig.data[i].visible for i in range(len(fig.data))]
    name_list = [fig.data[i].name for i in range(len(fig.data))]
    name_list = list(set(name_list))
    name_list.remove("")
    name_list.remove(None)
    # all_vis_list[sankey_index_sart:sankey_index_end] = False
    buttons1 = []
    buttons2 = []
    # buttons2 = []

    buttons1 = [
        dict(label="ALl", method="update", args=[{"visible": all_vis_list}]),
    ]
    for i in range(n_clusters):
        all_vis_list_c = np.array(all_vis_list)
        all_vis_list_c[sankey_index_sart] = False
        all_vis_list_c[sankey_index_sart + 1 + i] = True
        buttons1.append(
            {
                "label": "Cluster {}".format(i),
                "method": "update",
                "args": [
                    {
                        "visible": all_vis_list_c,
                    }
                ],
            }
        )

    index_fig = 0
    for name in name_list:
        if len(name.split("_")) == 2:
            all_vis_list_c = np.array(all_vis_list)
            all_vis_list_c[flow_sart : flow_sart + 3] = False
            m_maks_false = (
                np.array([fig.data[i].name for i in range(len(fig.data))]) == "0_1"
            )
            m_maks_true = (
                np.array([fig.data[i].name for i in range(len(fig.data))]) == name
            )
            all_vis_list_c[m_maks_false] = False
            all_vis_list_c[m_maks_true] = True

            buttons2.append(
                {
                    "label": "C{}->C{}".format(name.split("_")[0], name.split("_")[1]),
                    "method": "update",
                    "args": [
                        {
                            "visible": all_vis_list_c,
                        }
                    ],
                }
            )
            index_fig += 3

    fig.update_layout(
        updatemenus=[
            dict(
                buttons=buttons1,
                x=0.3,
                y=0.35,
                xanchor="left",
                yanchor="top",
            ),
            dict(
                buttons=buttons2,
                x=0.75,
                y=0.95,
                xanchor="left",
                yanchor="top",
            ),
        ]
    )
    return fig
