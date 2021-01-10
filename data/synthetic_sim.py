# Built off of NRI's code: https://github.com/ethanfetaya/NRI

import numpy as np
import matplotlib.pyplot as plt
import time
import networkx as nx

class SpringSim(object):
    def __init__(self, n_balls=5, box_size=5., loc_std=.5, vel_norm=.5,
                 interaction_strength=.1, noise_var=0., structure_config='00'):
        self.n_balls = n_balls
        self.box_size = box_size
        self.loc_std = loc_std
        self.vel_norm = vel_norm
        self.interaction_strength = interaction_strength
        self.noise_var = noise_var

        self._spring_types = np.array([0., 0.5, 1.])
        self._delta_T = 0.001
        self._max_F = 0.1 / self._delta_T

        self.structure_config = structure_config

        if self.structure_config == '00':
            # Default config, as in NRI
            self.center_first_l_levels = 0
            self.spring_type = 'ideal'
            self.per_level_speed = np.array([1., 1., 1.])

            int_s = self.interaction_strength
            self.per_level_l2a_a2d_force = np.array([int_s, int_s])
            self.per_level_ws_force = np.array([int_s, int_s, int_s])

            self.per_level_l2a_a2d_length = [0., 0.]
            self.per_level_ws_length = [0., 0., 0.]
        elif self.structure_config == '01':
            # ideal_center
            # Default config, as in NRI, but center init nodes (in hierarchy)
            self.center_first_l_levels = 3
            self.spring_type = 'ideal'
            self.per_level_speed = np.array([1., 1., 1.])

            int_s = self.interaction_strength
            self.per_level_l2a_a2d_force = np.array([int_s, int_s])
            self.per_level_ws_force = np.array([int_s, int_s, int_s])

            self.per_level_l2a_a2d_length = [0., 0.]
            self.per_level_ws_length = [0., 0., 0.]
        elif self.structure_config == '02':
            self.box_size = 1.
            # rigid_non-random hierarchy config
            # balls move only slightly, and have centered cluster pos initially
            self.center_first_l_levels = 3
            self.spring_type = 'finite'
            self.per_level_speed = np.array([.5, .5, .5]) / 8

            self.per_level_l2a_a2d_force = np.array([10, 10])*5
            self.per_level_ws_force = np.array([1.1, 1.0, 50.0]) * 5

            self.per_level_l2a_a2d_length = [0.4, 0.15]
            self.per_level_ws_length = [0.65, 0.2, 0.1]
        elif self.structure_config == '03':
            self.box_size = 1.
            # rigid_random config
            # balls move only slightly, but have random pos initially
            self.center_first_l_levels = 2
            self.spring_type = 'finite'
            self.per_level_speed = np.array([.5, .5, .5]) / 8

            self.per_level_l2a_a2d_force = np.array([10, 10])*5
            self.per_level_ws_force = np.array([1.1, 1.0, 50.0]) * 5

            self.per_level_l2a_a2d_length = [0.4, .15]
            self.per_level_ws_length = [0.65, 0.2, .1]
        elif self.structure_config == '04':
            self.box_size = 1.
            self.interaction_strength = .5
            self._max_F = 0.01 / self._delta_T
            # fast_non-random
            # balls move much faster, and have centered cluster pos initially
            self.center_first_l_levels = 3
            self.spring_type = 'finite'
            self.per_level_speed = np.array([2, 2, 2]) / 8

            self.per_level_l2a_a2d_force = np.array([100, 100])*5
            self.per_level_ws_force = np.array([1.1, 1.0, 50.0]) * 5

            self.per_level_l2a_a2d_length = [0.6, 0.03]
            self.per_level_ws_length = [0.65, 0.2, .04]
        elif self.structure_config == '05':
            self.box_size = 1.
            self.interaction_strength = .5
            self._max_F = 0.01 / self._delta_T
            # fast_random
            # balls move much faster, but have random pos initially
            self.center_first_l_levels = 2
            self.spring_type = 'finite'
            self.per_level_speed = np.array([1, 1, 1]) / 16

            self.per_level_l2a_a2d_force = np.array([100, 100])*5
            self.per_level_ws_force = np.array([1.1, 1.0, 50.0]) * 5

            self.per_level_l2a_a2d_length = [0.4, 0.1]
            self.per_level_ws_length = [0.65, 0.2, .1]

        # Generate graph properties
        self.hierarchy_nodes_list = self.create_hierarchy_nodes_list()
        self.Qc, self.Qstd, self.speed = self.generate_quadrant_coordinates()
        self.full_adj, self.adj_int_strength, self.adj_edge_length = self.generate_full_adj()
        self.adj_edge_length = self.adj_edge_length.reshape(1, n_balls, n_balls)

    def _energy(self, loc, vel, edges):
        # disables division by zero warning, since I fix it with fill_diagonal
        with np.errstate(divide='ignore'):

            K = 0.5 * (vel ** 2).sum()
            U = 0
            for i in range(loc.shape[1]):
                for j in range(loc.shape[1]):
                    if i != j:
                        r = loc[:, i] - loc[:, j]
                        dist = np.sqrt((r ** 2).sum())
                        U += 0.5 * self.interaction_strength * edges[
                            i, j] * (dist ** 2) / 2
            return U + K

    def _clamp(self, loc, vel):
        '''
        :param loc: 2xN location at one time stamp
        :param vel: 2xN velocity at one time stamp
        :return: location and velocity after hiting walls and returning after
            elastically colliding with walls
        '''
        assert (np.all(loc < self.box_size * 3))
        assert (np.all(loc > -self.box_size * 3))

        over = loc > self.box_size
        loc[over] = 2 * self.box_size - loc[over]
        assert (np.all(loc <= self.box_size))

        # assert(np.all(vel[over]>0))
        vel[over] = -np.abs(vel[over])

        under = loc < -self.box_size
        loc[under] = -2 * self.box_size - loc[under]
        # assert (np.all(vel[under] < 0))
        assert (np.all(loc >= -self.box_size))
        vel[under] = np.abs(vel[under])

        return loc, vel

    def _l2(self, A, B):
        """
        Input: A is a Nxd matrix
               B is a Mxd matirx
        Output: dist is a NxM matrix where dist[i,j] is the square norm
            between A[i,:] and B[j,:]
        i.e. dist[i,j] = ||A[i,:]-B[j,:]||^2
        """
        A_norm = (A ** 2).sum(axis=1).reshape(A.shape[0], 1)
        B_norm = (B ** 2).sum(axis=1).reshape(1, B.shape[0])
        dist = A_norm + B_norm - 2 * A.dot(B.transpose())
        return dist

    def sample_edges(self, spring_prob):
        edges = np.random.choice(self._spring_types,
                                 size=(self.n_balls, self.n_balls),
                                 p=spring_prob)
        edges = np.tril(edges) + np.tril(edges, -1).T
        np.fill_diagonal(edges, 0)
        return edges

    def sample_trajectory(self, T=10000, sample_freq=10,
                          spring_prob=[1. / 2, 0, 1. / 2]):
        n = self.n_balls
        assert (T % sample_freq == 0)
        T_save = int(T / sample_freq - 1)
        diag_mask = np.ones((n, n), dtype=bool)
        np.fill_diagonal(diag_mask, 0)
        counter = 0
        # Sample edges
        edges = self.sample_edges(spring_prob)
        # Initialize location and velocity
        loc = np.zeros((T_save, 2, n))
        vel = np.zeros((T_save, 2, n))
        loc_next = np.random.randn(2, n)
        vel_next = np.random.randn(2, n)

        # Put the particles to appropriate quadrants initially
        loc_next = np.multiply(loc_next, self.Qstd)
        loc_next += self.Qc

        v_norm = np.sqrt((vel_next ** 2).sum(axis=0)).reshape(1, -1)
        vel_next = vel_next * self.vel_norm / v_norm
        loc[0, :, :], vel[0, :, :] = self._clamp(loc_next, vel_next)

        # disables division by zero warning, since I fix it with fill_diagonal
        with np.errstate(divide='ignore'):
            forces = - self.adj_int_strength * edges
            np.fill_diagonal(forces, 0)  # self forces are zero
            forces = forces.reshape(1, n, n)
            F = (forces *
                 np.concatenate((
                     np.subtract.outer(loc_next[0, :],
                                       loc_next[0, :]).reshape(1, n, n),
                     np.subtract.outer(loc_next[1, :],
                                       loc_next[1, :]).reshape(1, n, n)))).sum(
                axis=-1)
            F[F > self._max_F] = self._max_F
            F[F < -self._max_F] = -self._max_F

            vel_next += self._delta_T * F
            # run leapfrog
            for i in range(1, T):
                loc_next += self._delta_T * (vel_next * self.speed)
                loc_next, vel_next = self._clamp(loc_next, vel_next)

                if i % sample_freq == 0:
                    loc[counter, :, :], vel[counter, :, :] = loc_next, vel_next
                    counter += 1

                F = self.calculate_forces(forces, loc_next)

                F[F > self._max_F] = self._max_F
                F[F < -self._max_F] = -self._max_F
                vel_next += self._delta_T * F
            # Add noise to observations
            loc += np.random.randn(T_save, 2, self.n_balls) * self.noise_var
            vel += np.random.randn(T_save, 2, self.n_balls) * self.noise_var

            return loc, vel, edges

    def generate_full_adj(self):
        """
        To implement in derived classes.
        :return: adjacency matrix
        """
        adj_mat = np.ones([self.n_balls, self.n_balls])
        adj_mat -= np.eye(self.n_balls)

        int_st = self.interaction_strength * adj_mat
        lengths = adj_mat.copy()
        return adj_mat, int_st, lengths

    def create_hierarchy_nodes_list(self):
        # not hierarchical dataset - all nodes inside 1 level
        hns = [list(range(self.n_balls))]
        return hns

    def generate_quadrant_coordinates(self):
        Qc = np.array([[0, 0] for _ in range(self.n_balls)])
        Qstd = np.array([self.loc_std for _ in range(self.n_balls)])
        s = np.array([[1, 1] for _ in range(self.n_balls)])
        return Qc.transpose(), Qstd.transpose(), s.transpose()

    def calculate_forces(self, F, loc):
        n = self.n_balls
        if self.spring_type == 'ideal':
            loc0s = np.subtract.outer(loc[0, :], loc[0, :]).reshape(1, n, n)
            loc1s = np.subtract.outer(loc[1, :], loc[1, :]).reshape(1, n, n)
            res = F * np.concatenate((loc0s, loc1s))
            res = res.sum(axis=-1)
        elif self.spring_type == 'finite':
            loc0s = np.subtract.outer(loc[0, :], loc[0, :]).reshape(1, n, n)
            loc1s = np.subtract.outer(loc[1, :], loc[1, :]).reshape(1, n, n)
            f0 = loc0s - self.adj_edge_length * np.sign(loc0s)
            f1 = loc1s - self.adj_edge_length * np.sign(loc1s)
            res = F * np.concatenate((f0, f1))
            res = res.sum(axis=-1)
        elif type == 'charged':
            raise NotImplementedError

        return res


class Hsprings(SpringSim):
    def __init__(self, n_children='4-4', **kwargs):
        """
        :param n_children: number of children nodes in each level have: '3-2'
        For other params see SpringSim class.
        """
        self.nc = list(map(int, n_children.split('-')))
        self.nl = len(self.nc)
        # number of nodes in each level of the graph that have children
        self.nn = [1]  # root node
        for l in range(1, self.nl + 1):
            self.nn.append(self.nn[-1] * self.nc[l-1])
        # total number of nodes in the graph
        n_balls = np.sum(self.nn)
        super(Hsprings, self).__init__(n_balls, **kwargs)

    def visualize(self, A):
        """
        Visualize graph determined by the adjacency matrix A
        :param A: adjacency matrix
        """
        G = nx.from_numpy_matrix(np.array(A))
        nx.draw(G, with_labels=True)
        plt.show()
        plt.clf()
        exit(0)

    def generate_full_adj(self):
        """
        To implement in derived classes.
        :return: adjacency matrix
        """
        raise NotImplementedError

    def create_hierarchy_nodes_list(self):
        # Create node list (1 for each level of hierarchy)
        i = 0
        hns = [[i]]  # root node
        i += 1
        for l in range(1, self.nl + 1):
            hns.append(list(range(i,i+np.prod(self.nc[:l]))))
            i += np.prod(self.nc[:l])
        return hns


class HspringsV1TreeRandomDrop(Hsprings):

    def __init__(self, n_children='4-4', **kwargs):
        """
        For params see Hsprings class.
        """
        super(HspringsV1TreeRandomDrop, self).__init__(
            n_children, **kwargs)

    def sample_edges(self, spring_prob):
        # create upper triangular part of the adjacency matrix
        edges = np.triu(self.generate_full_adj(), 1)

        edges_mask = np.random.choice(self._spring_types,
                                      size=(self.n_balls, self.n_balls),
                                      p=spring_prob)

        edges *= edges_mask
        # complete the lower triangular part
        edges = edges + edges.T

        # self.visualize(edges)

        return edges

    def generate_full_adj(self):
        """
        Generates the full adjacency matrix for this class. (no random drops)
        :return: adjacency matrix
        """
        edges = np.zeros(shape=(self.n_balls, self.n_balls))
        row_idx = 0  # start filling adjacency mat from root node
        col_idx = 1  # skip the root node and start from 2nd node
        for l in range(self.nl):
            for n in range(self.nn[l]):
                edges[row_idx, col_idx:col_idx + self.nc[l]] = 1
                # Increase counters after filling connections for a parent node
                col_idx += self.nc[l]
                row_idx += 1
        return edges


class HspringsV2TreeRandomClusters(Hsprings):
    def __init__(self, n_children='4-4', randomize_all_clusters=False,
                 **kwargs):
        """
        For params see Hsprings class.
        """
        # This is the key difference between V2 and V3
        # If False, it makes (random) sibling connections only at the last
        # level. Otherwise, it makes them at all levels of the graph.
        self.randomize_all_clusters = randomize_all_clusters
        super(HspringsV2TreeRandomClusters, self).__init__(
            n_children, **kwargs)

    def sample_cluster_adj_mat(self, nc):
        prod = 0
        spring_prob = [.75, 0, .25]
        # make sure that graph is not disconnected
        while prod == 0:
            cluster_adj = np.random.choice(
                self._spring_types, size=(nc, nc),
                p=spring_prob)
            cluster_adj = (cluster_adj + cluster_adj.T) / 2 + np.eye(nc)
            # compute A^n, and then check if any of the elements is == 0
            # this means there are no n-th order connections between nodes
            # https://math.stackexchange.com/questions/864604/checking-connectivity-of-adjacency-matrix
            prod = np.prod(np.linalg.matrix_power(cluster_adj, nc))

        cluster_adj = (cluster_adj > 0).astype(np.float32) - np.eye(nc)
        return cluster_adj

    def generate_full_adj(self):
        """
        Generates the full adjacency matrix for this class. (no random drops)
        :return: adjacency matrix
        """
        edges = np.zeros(shape=(self.n_balls, self.n_balls))
        row_idx = 0  # start filling adjacency mat from root node
        col_idx = 1  # skip the root node and start from 2nd node

        # generate hierarchical interaction strength matrix
        int_strngth = np.zeros(shape=(self.n_balls, self.n_balls))
        # generate hierarchical edge lengths matrix
        lengths = np.zeros(shape=(self.n_balls, self.n_balls))
        for l in range(self.nl):
            for n in range(self.nn[l]):
                edges[row_idx, col_idx:col_idx + self.nc[l]] = 1
                # Fill symmetric the lower triangular (undirected graph)
                edges[col_idx:col_idx + self.nc[l], row_idx] = 1

                # same for interaction strength
                int_strngth[row_idx, col_idx:col_idx + self.nc[l]] = self.per_level_l2a_a2d_force[l]
                int_strngth[col_idx:col_idx + self.nc[l], row_idx] = self.per_level_l2a_a2d_force[l]

                # same for lengths
                lengths[row_idx, col_idx:col_idx + self.nc[l]] = self.per_level_l2a_a2d_length[l]
                lengths[col_idx:col_idx + self.nc[l], row_idx] = self.per_level_l2a_a2d_length[l]

                if l == self.nl - 1 or self.randomize_all_clusters:
                    # create full cluster adjacency matrix
                    edges[col_idx:col_idx + self.nc[l],
                          col_idx:col_idx + self.nc[l]] = \
                        np.ones((self.nc[l], self.nc[l])) - np.eye(self.nc[l])
                    # same for interaction strength
                    int_strngth[col_idx:col_idx + self.nc[l],
                          col_idx:col_idx + self.nc[l]] = \
                        np.ones((self.nc[l], self.nc[l])) * self.per_level_ws_force[l]
                    # same for lengths
                    lengths[col_idx:col_idx + self.nc[l],
                          col_idx:col_idx + self.nc[l]] = \
                        np.ones((self.nc[l], self.nc[l])) * self.per_level_ws_length[l]

                # Increase counters after filling connections for a parent node
                col_idx += self.nc[l]
                row_idx += 1

        return edges, int_strngth, lengths

    def generate_quadrant_coordinates(self):
        """
        For each particle it generates quadrant center coordinates inside which
        the initial particle position can be sampled.
        Structure passing through the tree is similar to `generate_full_adj` fn
        How to make sure particles are assigned into apropriate quadrants:
        1) generate Quadrant center coordiates (n_atoms, 2)
        2) generate Quadrant width (initial = self.loc_std) (n_atoms,)
        3) for all atoms generate initial positions at random
        4) multiply by each atoms coordinates by loc_std (to shrink to Q width)
        5)
        :return: dictionary of quadrant centers and std_loc
        """
        qw = 1.  # quadrant width
        Qc = np.array([[0, 0]])
        Qo = np.array([[-qw, -qw], [qw, qw], [qw, -qw], [-qw, qw]])
        Qstd = np.array([qw])
        if self.center_first_l_levels > 0:
            Qstd *= 0.

        ls = self.per_level_speed
        s = np.array([[ls[0], ls[0]]])

        row_idx = 0  # start filling adjacency mat from root node
        col_idx = 1  # skip the root node and start from 2nd node
        for l in range(self.nl):
            Qo /= 2
            qw /= 2
            for n in range(self.nn[l]):
                for c in range(self.nc[l]):
                    pc = Qc[row_idx]
                    cc = pc + Qo[c]
                    Qc = np.append(Qc, [cc], axis=0)
                    if l < self.center_first_l_levels - 1:
                        Qstd = np.append(Qstd, 0)
                    else:
                        Qstd = np.append(Qstd, qw)
                    col_idx += 1
                    s = np.append(s, [[ls[l+1], ls[l+1]]], axis=0)
                # Increase parent index after populating all its children nodes
                row_idx += 1

        return Qc.transpose(), Qstd.transpose(), s.transpose()

    def sample_edges(self, spring_prob):
        edges = np.zeros(shape=(self.n_balls, self.n_balls))
        row_idx = 0  # start filling adjacency mat from root node
        col_idx = 1  # skip the root node and start from 2nd node
        for l in range(self.nl):
            for n in range(self.nn[l]):
                if l < self.nl - 1 and not self.randomize_all_clusters:
                    # Note: we handle differently the first level in the hierarchy,
                    #       as we want clear cut to n_children branches here
                    edges[row_idx, col_idx:col_idx + self.nc[l]] = 1
                    # Fill symmetric the lower triangular (undirected graph)
                    edges[col_idx:col_idx + self.nc[l], row_idx] = 1
                else:
                    # create parent - cluster relations (at least 1 connection)
                    r_int = np.random.randint(low=1, high=2 ** self.nc[l])
                    binary_str = format(r_int, '0{}b'.format(self.nc[l]))
                    for j in range(self.nc[l]):
                        edges[row_idx, col_idx + j] = binary_str[j]
                        edges[col_idx + j, row_idx] = binary_str[j]

                    # create cluster adjacency matrix
                    edges[col_idx:col_idx + self.nc[l],
                          col_idx:col_idx + self.nc[l]] = \
                        self.sample_cluster_adj_mat(self.nc[l])
                # Increase counters after filling connections for a parent node
                col_idx += self.nc[l]
                row_idx += 1

        # self.visualize(edges)

        return edges

class HspringsV3TreeRandomAllLevels(HspringsV2TreeRandomClusters):
    def __init__(self, n_children='4-4', **kwargs):
        """
        For params see Hsprings class.
        """
        # This is the key difference to V2 - it will make siblings connections
        # at all graph levels, not only the last one
        randomize_all_clusters = True
        super(HspringsV3TreeRandomAllLevels, self).__init__(
            n_children, randomize_all_clusters, **kwargs)


class ChargedParticlesSim(object):
    def __init__(self, n_balls=5, box_size=5., loc_std=1., vel_norm=0.5,
                 interaction_strength=1., noise_var=0.):
        self.n_balls = n_balls
        self.box_size = box_size
        self.loc_std = loc_std
        self.vel_norm = vel_norm
        self.interaction_strength = interaction_strength
        self.noise_var = noise_var

        self._charge_types = np.array([-1., 0., 1.])
        self._delta_T = 0.001
        self._max_F = 0.1 / self._delta_T

    def _l2(self, A, B):
        """
        Input: A is a Nxd matrix
               B is a Mxd matirx
        Output: dist is a NxM matrix where dist[i,j] is the square norm
            between A[i,:] and B[j,:]
        i.e. dist[i,j] = ||A[i,:]-B[j,:]||^2
        """
        A_norm = (A ** 2).sum(axis=1).reshape(A.shape[0], 1)
        B_norm = (B ** 2).sum(axis=1).reshape(1, B.shape[0])
        dist = A_norm + B_norm - 2 * A.dot(B.transpose())
        return dist

    def _energy(self, loc, vel, edges):

        # disables division by zero warning, since I fix it with fill_diagonal
        with np.errstate(divide='ignore'):

            K = 0.5 * (vel ** 2).sum()
            U = 0
            for i in range(loc.shape[1]):
                for j in range(loc.shape[1]):
                    if i != j:
                        r = loc[:, i] - loc[:, j]
                        dist = np.sqrt((r ** 2).sum())
                        U += 0.5 * self.interaction_strength * edges[
                            i, j] / dist
            return U + K

    def _clamp(self, loc, vel):
        '''
        :param loc: 2xN location at one time stamp
        :param vel: 2xN velocity at one time stamp
        :return: location and velocity after hiting walls and returning after
            elastically colliding with walls
        '''
        assert (np.all(loc < self.box_size * 3))
        assert (np.all(loc > -self.box_size * 3))

        over = loc > self.box_size
        loc[over] = 2 * self.box_size - loc[over]
        assert (np.all(loc <= self.box_size))

        # assert(np.all(vel[over]>0))
        vel[over] = -np.abs(vel[over])

        under = loc < -self.box_size
        loc[under] = -2 * self.box_size - loc[under]
        # assert (np.all(vel[under] < 0))
        assert (np.all(loc >= -self.box_size))
        vel[under] = np.abs(vel[under])

        return loc, vel

    def sample_trajectory(self, T=10000, sample_freq=10,
                          charge_prob=[1. / 2, 0, 1. / 2]):
        n = self.n_balls
        assert (T % sample_freq == 0)
        T_save = int(T / sample_freq - 1)
        diag_mask = np.ones((n, n), dtype=bool)
        np.fill_diagonal(diag_mask, 0)
        counter = 0
        # Sample edges
        charges = np.random.choice(self._charge_types, size=(self.n_balls, 1),
                                   p=charge_prob)
        edges = charges.dot(charges.transpose())
        # Initialize location and velocity
        loc = np.zeros((T_save, 2, n))
        vel = np.zeros((T_save, 2, n))
        loc_next = np.random.randn(2, n) * self.loc_std
        vel_next = np.random.randn(2, n)
        v_norm = np.sqrt((vel_next ** 2).sum(axis=0)).reshape(1, -1)
        vel_next = vel_next * self.vel_norm / v_norm
        loc[0, :, :], vel[0, :, :] = self._clamp(loc_next, vel_next)

        # disables division by zero warning, since I fix it with fill_diagonal
        with np.errstate(divide='ignore'):
            # half step leapfrog
            l2_dist_power3 = np.power(
                self._l2(loc_next.transpose(), loc_next.transpose()), 3. / 2.)

            # size of forces up to a 1/|r| factor
            # since I later multiply by an unnormalized r vector
            forces_size = self.interaction_strength * edges / l2_dist_power3
            np.fill_diagonal(forces_size,
                             0)  # self forces are zero (fixes division by zero)
            assert (np.abs(forces_size[diag_mask]).min() > 1e-10)
            F = (forces_size.reshape(1, n, n) *
                 np.concatenate((
                     np.subtract.outer(loc_next[0, :],
                                       loc_next[0, :]).reshape(1, n, n),
                     np.subtract.outer(loc_next[1, :],
                                       loc_next[1, :]).reshape(1, n, n)))).sum(
                axis=-1)
            F[F > self._max_F] = self._max_F
            F[F < -self._max_F] = -self._max_F

            vel_next += self._delta_T * F
            # run leapfrog
            for i in range(1, T):
                loc_next += self._delta_T * vel_next
                loc_next, vel_next = self._clamp(loc_next, vel_next)

                if i % sample_freq == 0:
                    loc[counter, :, :], vel[counter, :, :] = loc_next, vel_next
                    counter += 1

                l2_dist_power3 = np.power(
                    self._l2(loc_next.transpose(), loc_next.transpose()),
                    3. / 2.)
                forces_size = self.interaction_strength * edges / l2_dist_power3
                np.fill_diagonal(forces_size, 0)
                # assert (np.abs(forces_size[diag_mask]).min() > 1e-10)

                F = (forces_size.reshape(1, n, n) *
                     np.concatenate((
                         np.subtract.outer(loc_next[0, :],
                                           loc_next[0, :]).reshape(1, n, n),
                         np.subtract.outer(loc_next[1, :],
                                           loc_next[1, :]).reshape(1, n,
                                                                   n)))).sum(
                    axis=-1)
                F[F > self._max_F] = self._max_F
                F[F < -self._max_F] = -self._max_F
                vel_next += self._delta_T * F
            # Add noise to observations
            loc += np.random.randn(T_save, 2, self.n_balls) * self.noise_var
            vel += np.random.randn(T_save, 2, self.n_balls) * self.noise_var
            return loc, vel, edges

    def generate_full_adj(self):
        """
        To implement in derived classes.
        :return: adjacency matrix
        """
        adj_mat = np.ones([self.n_balls, self.n_balls])
        adj_mat -= np.eye(self.n_balls)
        return adj_mat


if __name__ == '__main__':
    sim = SpringSim()
    # sim = ChargedParticlesSim()

    t = time.time()
    loc, vel, edges = sim.sample_trajectory(T=5000, sample_freq=100)

    print(edges)
    print("Simulation time: {}".format(time.time() - t))
    vel_norm = np.sqrt((vel ** 2).sum(axis=1))
    plt.figure()
    axes = plt.gca()
    axes.set_xlim([-5., 5.])
    axes.set_ylim([-5., 5.])
    for i in range(loc.shape[-1]):
        plt.plot(loc[:, 0, i], loc[:, 1, i])
        plt.plot(loc[0, 0, i], loc[0, 1, i], 'd')
    plt.figure()
    energies = [sim._energy(loc[i, :, :], vel[i, :, :], edges) for i in
                range(loc.shape[0])]
    plt.plot(energies)
    plt.show()
