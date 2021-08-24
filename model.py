"""
This model simulate the dynamics of a City under Covid-19 pandemic.

DySTUrbD-epi: Model
This program is the implementation of DySTUrbD-Epi class.
"""

import torch
from scipy.sparse.csgraph import shortest_path as sp

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
        path = self.__get_path()
        print("DONE")

    def __get_path(self):
        """
        Compute shortest paths among nodes

        Schematic illustration of getting (A+B,A+B) from (A,A), (A,B), (A,B)^T, (B,B)
        let A = 3, B = 2

            A              B
         ------           ----
         |= = =           |= =
        A|=   =  concat  A|= =
         |= = =           |= =

                 concat

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
        AB = self.network.AB
        BA = AB.transpose(0, 1)

        AAAB = torch.cat((AA, AB), 1)
        BABB = torch.cat((BB, BA), 1)
        res = torch.cat((AAAB, BABB), 0)

        # import scipy shortest path as sp
        res = sp(res.detach().cpu().numpy(), directed=True, return_predecessors=False)
        res = torch.from_numpy(res).to(self.device)
        res = res.fill_diagonal_(100)

        return res
