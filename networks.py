"""
This module initializes agents.

A subset of the attributes will be loaded directly from csv file.
While the remaining will be randomly initialized.

For the details of the attribute, please refer to readme.
"""

import torch


class Networks(object):
    def __init__(self, agents, buildings, device):
        """
        Create three networks.

        1. AA - agent - agent
        2. BB - building - building
        3. AB - agent - building
        """
        self.BB = self.__building_building(buildings)
        self.AA = self.__agent_agent(agents, buildings, device)
        self.AB = self.__agent_building(agents, buildings, device)

    def __building_building(self, buildings):
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
        threshold = 0.00161
        vol = buildings.floor["volume"]
        coor = torch.cat(
            (
                buildings.identity["X"].view(-1, 1),
                buildings.identity["Y"].view(-1, 1),
            ),
            dim=1,
        ).double()
        distance = torch.cdist(coor, coor, 2).pow(beta)
        distance[distance == 0] = 1e-5
        scores = vol.div(distance)
        scores = scores.fill_diagonal_(0)
        res = scores.div(torch.sum(scores, 1).view(-1, 1))
        res[res < threshold] = 0
        # res = -torch.log(res)
        # res = torch.nan_to_num(res, posinf=100)

        return res

    def __agent_agent(self, agents, buildings, device):
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
        threshold = 0.95
        income = self.__income_sim(agents, device)
        age = self.__age_sim(agents)
        dist = self.__house_dist(agents, buildings)
        w = {"income": 1, "age": 1, "dist": 1}
        res = income.mul(w["income"]).mul(age.mul(w["age"])).mul(dist.mul(w["dist"]))

        h = agents.identity["house"]
        res[h == h.view(-1, 1)] = 1  # same house gets max probability
        res[res < threshold] = 0

        # res = -torch.log(res)
        # res = torch.nan_to_num(res, posinf=100)

        return res

    def __income_sim(self, agents, device):
        """
        Compute income pairwise similarity.

        Return
        --------
        res : torch.Tensor
        """
        h = agents.identity["house"]
        j = agents.job
        h_unique = torch.unique(h)
        agent_hh = (h_unique[:, None] == h).long().argmax(dim=0)
        h_income = torch.zeros((h_unique.shape[0],), dtype=torch.float64, device=device)

        for idx in range(h_unique.shape[0]):
            mask = (h == h_unique[idx]) & (j["status"] == 1)
            h_income[idx] = torch.sum(j["income"][mask])

        h_income = h_income[agent_hh]
        res = h_income.sub(h_income.view(-1, 1)).abs()
        res = 1 - (res / torch.max(res))

        del h_unique, agent_hh, h_income, mask
        return res

    def __age_sim(self, agents):
        """
        Compute age pairwise similarity.

        Return
        -------
        res : torch.Tensor
        """
        age = agents.identity["age"].double()
        res = age.sub(age.view(-1, 1)).abs()
        res = 1 - (res / torch.max(res))

        return res

    def __house_dist(self, agents, buildings):
        """
        Compute household pairwise distance

        Return
        --------
        res : torch.Tensor
        """
        agent_homes = agents.building["house"]
        id_b = buildings.identity["idx"]
        h_idx = (id_b[:, None] == agent_homes).int().argmax(dim=0)
        coor = torch.cat(
            (
                buildings.identity["X"][h_idx].view(-1, 1),
                buildings.identity["Y"][h_idx].view(-1, 1),
            ),
            1,
        ).double()
        res = torch.cdist(coor, coor)
        res = 1 - (res / torch.max(res))

        del h_idx, coor
        return res

    def __agent_building(self, agents, buildings, device):
        """
        Create network of agent and buildings.

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
        res : dict
        """
        type_b = ["house", "anchor", "trivial"]
        num_a = agents.identity["id"].shape[0]
        num_b = buildings.identity["id"].shape[0]
        res = dict()
        res["total"] = torch.zeros((num_a, num_b), device=device, dtype=torch.bool)

        for val in type_b:
            res[val] = torch.nn.functional.one_hot(
                agents.building[val].long(), num_classes=num_b
            ).bool()
            res["total"] = res["total"] | res[val]

        # res["total"] = -torch.log(res["total"])
        # res["total"] = torch.nan_to_num(res["total"], posinf=100)

        return res
