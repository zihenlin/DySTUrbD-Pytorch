"""
This model simulate the dynamics of a City under Covid-19 pandemic.

DySTUrbD-epi: Model
This program is the implementation of DySTUrbD-Epi class.
"""

import torch
from scipy.sparse.csgraph import shortest_path as sp
from scipy.sparse import csr_matrix

import buildings
import agents
import networks


class DySTUrbD_Epi(object):
    """Simulate the world."""

    def __init__(self, args):
        """
        Initialize the world.

        Parameter
        ---------
        args : arguments
        """
        print("Init start!")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.device = "cpu"
        print("Current Device:", self.device)

        print("Init buildings")
        self.buildings = buildings.Buildings(args.buildings_dir, self.device)

        print("Init agents")
        self.agents = agents.Agents(args, self.buildings, self.device)

        print("Init networks")
        self.network = networks.Networks(self.agents, self.buildings, self.device)

        print("Init complete!")

    def __call__(self):
        """
        Run the model, Gogogo!
        """
        print("Get shortest path")
        dist = self._get_dist()
        prob_AA = self._prob_AA(dist)
        routine = self._get_routine(dist)
        print("DONE")

    def _get_dist(self):
        """
        Compute shortest paths among nodes

        Schematic illustration of getting (A+B,A+B) from (A,A), (A,B), (A,B)^T, (B,B)
        let A = 3, B = 2

            A              B
         ------           ----
         |= = =           |= =
        A|=   =  concat  A|= =
         |= = =           |= =

         concat          concat

         |= = =           |= =
        B|= = =  concat  B|= =
         ------           ----
            A               B

        Return
        -------
        res : torch.Tensor (A+B, A+B)
        """
        AA = self.network.AA
        BB = self.network.BB
        AB = self.network.AB["total"]
        BA = torch.zeros_like(AB.T, device=self.device)

        AAAB = torch.cat((AA, AB), 1)
        BABB = torch.cat((BA, BB), 1)
        res = torch.cat((AAAB, BABB), 0)

        # Construct csr matrix version
        dim = res.shape[0]
        res = res.fill_diagonal_(0)
        res = res.to_sparse()
        row_idx = res.indices()[0].detach().cpu().numpy()
        col_idx = res.indices()[1].detach().cpu().numpy()
        val = -torch.log(res.values()).detach().cpu().numpy()
        csr = csr_matrix((val, (row_idx, col_idx)), shape=(dim, dim))
        res = sp(csr, directed=True, return_predecessors=False)

        res = torch.from_numpy(res).to(self.device)
        res = res.fill_diagonal_(float("inf"))
        # TODO got reasonable outcome but different result

        return res

    def _prob_AA(self, dist):
        """
        Return agents-interaction matrix

        Return
        ------
        res : torch.Tensor
        """
        num_a = self.network.AA.shape[0]
        res = dist[:num_a, :num_a]
        res = torch.exp(-res)

        return res

    def _get_routine(self, dist):
        """
        Get a list of buildings visited by each agent.

        Return
        -------
        res : torch.Tensor (A, B)
        """
        num_aa = self.network.AA.shape[0]
        num_bb = self.network.BB.shape[0]
        dist_ab = 1
        dist_bb = 6

        nodes_ab = dist[:num_aa, -num_bb:] < dist_ab  # a_nodes (A,B)
        nodes_bb = dist[-num_bb:, -num_bb:] < dist_bb  # all buildings (A,B,B)

        res = self.network.AB["total"]
        regular = ["house", "anchor", "trivial"]

        for b in regular:
            for cnt in range(2):
                template = torch.zeros_like(self.network.AB["total"])

                if cnt < 1:
                    prev = nodes_ab
                    nodes = self.network.AB[b]
                    clone_bb = nodes_bb
                else:
                    prev = candidates
                    dummy_col = ~(choice.sum(0).bool())
                    nodes = torch.cat((choice, dummy_col.view(-1, 1)), 1)
                    dummy_row = torch.zeros((1, num_bb), device=self.device)
                    clone_bb = torch.cat((nodes_bb, dummy_row), 0)

                idx = nodes.long().argmax(
                    dim=1
                )  # get the building index for each agent
                candidates = clone_bb.index_select(0, idx).bool()
                candidates = (candidates | prev) & ~(self.network.AB["total"])
                idx = candidates.double().multinomial(
                    1
                )  # randomly pick one from each row
                choice = template.scatter(1, idx.view(-1, 1), candidates)
                res |= choice  # add new building to routine (some are zeros)

        return res
