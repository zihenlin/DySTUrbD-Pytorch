"""
This model simulate the dynamics of a City under Covid-19 pandemic.

DySTUrbD-epi: Model
This program is the implementation of DySTUrbD-Epi class.
"""

import torch

import buildings
import agents


class DySTUrbD_Epi(object):
    """Simulate the world."""

    def __init__(self, args):
        """
        Initialize the world.

        Parameter
        ---------
        args : arguments
        """
        self.buildings = buildings.Buildings(args.buildings_dir)
        self.agents = agents.Agents(args, self.buildings)
        self.network = self.__create_network()

    def __create_network(self):
        """
        Create three networks.

        1. AA - agent - agent
        2. Aa - agent - anchor
        3. AH - agent - house
        4. BB - building - building

        Return
        --------
        res : dict
        """
        res = {
            "BB": self.__building_building(),
            "AA": self.__agent_agent(),
            "Aa": self.__agent_anchor(),
            "AH": self.__agent_house(),
        }

        return res

    def __building_building(self):
        """
        Initialize building building network.

        Nodes
        --------
        Buildings

        Edges
        --------
        Scores (floor volume / distance)

        Return
        --------
        res : torch.Tensor
        """
        beta = 2
        threshold = 0.5
        vol = self.buildings.floor["volume"]
        coor = torch.cat(
            (
                self.buildings.identity["X"].view(-1, 1),
                self.buildings.identity["Y"].view(-1, 1),
            ),
            dim=1,
        )
        distance = torch.cdist(coor, coor).pow(beta)
        distance[distance == 0] = 1e6
        scores = vol.div(distance)

        for i in range(coor.shape[0]):
            scores[i, i] = 0

        res = scores.div(torch.sum(scores, 1))
        res[res < threshold] = 0
        res = -torch.log(res)

        return res

    def __agent_agent(self):
        """
        Initialize Agent - Agent network.

        Nodes
        --------
        Agents

        Edges
        --------
        Composite similarity in temrs of age, income, and household distance.
        @TODO Learn the weights to reflect the importance of each factor.

        Return
        --------
        res : torch.Tensor
        """
        threshold = 0.5
        income = self.__income_sim()
        age = self.__age_sim()
        dist = self.__house_dist()
        w = {"income": 1, "age": 1, "dist": 1}
        res = w["income"] * income * w["age"] * age * w["dist"] * dist

        h = self.agents.identity["house"]
        res[h == h.view(-1, 1)] = 1  # same house gets max probability
        res[res < threshold] = 0
        res = -torch.log(res)

        return res

    def __income_sim(self):
        """
        Compute income pairwise similarity.

        Return
        --------
        res : torch.Tensor
        """
        h = self.agents.identity["house"]
        j = self.agents.job
        h_unique = torch.unique(h)
        agent_hh = (h_unique[:, None] == h).long().argmax(dim=0)
        h_income = torch.zeros((h_unique.shape[0],))

        for idx in range(h_unique.shape[0]):
            mask = (h == h_unique[idx]) & (j["status"] == 1)
            h_income[idx] = torch.sum(j["income"][mask])

        h_income = h_income[agent_hh]
        res = h_income.sub(h_income.view(-1, 1)).abs()
        res = 1 - (res.div(res.max(dim=1)[0]))

        return res

    def __age_sim(self):
        """
        Compute age pairwise similarity.

        Return
        -------
        res : torch.Tensor
        """
        age = self.agents.identity["age"]
        res = age.sub(age.view(-1, 1)).abs()
        res = 1 - (res.div(res.max(dim=1)[0]))

        return res

    def __house_dist(self):
        """
        Compute household pairwise distance

        Return
        --------
        res : torch.Tensor
        """
        agent_homes = self.agents.building["house"]
        buildings = self.buildings.identity["id"]
        h_idx = (buildings[:, None] == agent_homes).long().argmax(dim=0)
        coor = torch.cat(
            (
                self.buildings.identity["X"][h_idx].view(-1, 1),
                self.buildings.identity["Y"][h_idx].view(-1, 1),
            ),
            1,
        )
        res = torch.cdist(coor, coor)
        res = 1 - (res.div(res.max(dim=1)[0]))

        return res

    def __agent_anchor(self):
        """
        Create network of agent and anchor buildings.

        Nodes
        ------
        Agents, Buildings

        Edges
        ------
        1 between agent and their anchor building.

        Shape
        ------
        (Agent, Buildings)

        Return
        ------
        res : torch.Tensor
        """
        agents = self.agents.building["anchor"]
        anchors = torch.unique(agents)
        res = torch.zeros((agents.shape[0], anchors.shape[0]))

        for idx in anchors.shape[0]:
            mask = agents == anchors[idx]
            res[mask, idx] = 1

        return res

    def __agent_house(self):
        """
        Create network of agent and house.

        Nodes
        ------
        Agents, Buildings

        Edges
        ------
        1 between agent and their house.

        Shape
        ------
        (Agent, Buildings)

        Return
        ------
        res : torch.Tensor
        """
        agents = self.agents.building["house"]
        houses = torch.unique(agents)
        res = torch.zeros((agents.shape[0], houses.shape[0]))

        for idx in houses.shape[0]:
            mask = agents == houses[idx]
            res[mask, idx] = 1

        return res
