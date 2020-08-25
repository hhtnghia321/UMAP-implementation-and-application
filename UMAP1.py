import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import euclidean_distances
from scipy import optimize
from sklearn.manifold import SpectralEmbedding
from tqdm import tqdm

class UMAP():
    def __init__(self, min_dist, n_dim, k, sym_method = 1, spec_neighbor = 50, epoch = 200, learning_rate = 1, ran_seed = None, k_tolerant = 20,Q_method = 1, X = [[1],[1]]):
        self.data = X
        self.min_dist = min_dist
        self.n_dim = n_dim
        self.k = k
        self.dist = euclidean_distances(X, X)
        self.rho = [sorted(self.dist[i])[1] for i in range(self.dist.shape[0])]
        self.sym_method = sym_method 
        self.spec_neighbor = spec_neighbor
        self.a, self.b = self.get_a_b()
        self.epoch = epoch
        self.learning_rate = learning_rate
        self.seed = ran_seed
        self.k_tolerant = k_tolerant
        self.Q_method = Q_method

    def high_top_rep_i(self,row, sigma_i):
        d = self.dist[row]- self.rho[row]
        d[d < 0 ] = 0
        P_ij = np.exp(-d/sigma_i)
        return P_ij

    def high_top_rep(self):
        d= self.dist- np.repeat(np.array(self.rho).reshape((1,-1)),150,axis = 0)
        d[d < 0 ] = 0
        P = np.exp(-d/np.repeat(np.array(self.sigma()).reshape((-1,1)),150,axis = 1))
        return P

    def k_func(self, P_ij):
        k =  np.power(2, P_ij.sum())
        return k 

    def bi_search_sigma(self, prob_k):
        lower_sigma = 1e-100
        upper_sigma = 1000
        appro_sigma = (upper_sigma + lower_sigma)/2
        for _ in range(self.k_tolerant):
        #while abs(prob_k(appro_sigma) - self.k) > self.k_tolerant:
            #print(abs(prob_k(appro_sigma) - self.k))
            appro_sigma = (upper_sigma + lower_sigma)/2
            if prob_k(appro_sigma) < self.k:
                lower_sigma = appro_sigma
            else:
                upper_sigma = appro_sigma
            if np.abs(prob_k(appro_sigma) - self.k) <= 1e-5:
                break
        return appro_sigma

    def sigma(self):
        sigma = []
        for i in range(self.data.shape[0]):
            prob_k = lambda sigma: self.k_func(self.high_top_rep_i(row = i, sigma_i = sigma))
            #prob_k return k with particular P_i
            sigma_i = self.bi_search_sigma(prob_k)
            sigma.append(sigma_i)
        return sigma

    def approx_curve(self, dist):
            q = []
            for i in range(len(dist)):
                if dist[i] <= self.min_dist:
                    q.append(1)
                else:
                    q.append(np.exp(-dist[i]+self.min_dist))
            return q

    def low_top_rep(self, dist_i,a,b):
        Q_ij = np.power((1 + a*dist_i**(2*b)),-1)
        return Q_ij

    def get_a_b(self):
        dist = np.linspace(0, 10, 300)
        para, _ = optimize.curve_fit(self.low_top_rep, dist, self.approx_curve(dist))
        return para[0], para[1]

    def sym(self):
        P = self.high_top_rep()
        if self.sym_method == 1:
            P = P + np.transpose(P) - np.multiply(P, np.transpose(P))
        else:
            P = (P + np.transpose(P))/2
        return P

    def init_Y(self):
        Y = SpectralEmbedding(n_components = self.n_dim, n_neighbors = self.spec_neighbor).fit_transform(np.log(self.data +1))
        return Y

    def Cross_En(self, Y, P):
        dist_y = np.square(euclidean_distances(Y, Y))
        Q = self.low_top_rep(dist_y, self.a, self.b)
        CE = - P * np.log(Q + 0.01) - (1 - P) * np.log(1 - Q + 0.01)
        return np.sum(CE)

    def Cross_En_grad(self, Y, P):
        y_diff = np.expand_dims(Y, 1) - np.expand_dims(Y, 0)
        inv_dist = np.power(1 + self.a * np.square(euclidean_distances(Y, Y))**self.b, -1)
        Q = np.dot(1 - P, np.power(0.001 + np.square(euclidean_distances(Y, Y)), -1))
        np.fill_diagonal(Q, 0)
        if self.Q_method == 1:
            Q = Q / np.sum(Q, axis = 1, keepdims = True)
        fact=np.expand_dims(self.a*P*(1e-8 + np.square(euclidean_distances(Y, Y)))**(self.b-1) - Q, 2)
        return 2 * self.b * np.sum(fact * y_diff * np.expand_dims(inv_dist, 2), axis = 1)

    def fit_transform(self, X):
        self.__init__(X = X, min_dist=self.min_dist, n_dim = self.n_dim, k = self.k)
        np.random.seed(self.seed)
        Y = self.init_Y()
        P = self.sym()
        CE = []
        for i in tqdm(range(self.epoch)):
            #print(self.Cross_En_grad(Y, P))
            Y = Y - self.learning_rate*self.Cross_En_grad(Y, P)
            CE.append(self.Cross_En(Y,P)/1e+5)
        return Y , CE





    
