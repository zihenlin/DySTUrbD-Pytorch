"""
This module initializes agents.

A subset of the attributes will be loaded directly from csv file.
While the remaining will be randomly initialized.

For the details of the attribute, please refer to readme.
"""

import gc

import torch


class Networks(object):
    def __init__(self, args, agents, buildings, device):
        """
        Create three networks.

        1. AA - agent - agent
        2. BB - building - building
        3. AB - agent - building
        4. AH - agent - household
        """
        print("BB")
        self.BB = self.__building_building(args, buildings)
        print("AA")
        self.AA = self.__agent_agent(args, agents, buildings, device)
        print("AB")
        self.AB = self.__agent_building(agents, buildings, device)
        print("AH")
        self.AH = self.__agent_household(agents)
        print("ASA")
        self.ASA = self.__agent_SA(agents, buildings)

        gc.collect()

    def __building_building(self, args, buildings):
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
        res : torch.coo_sparse
        """
        beta = args["BB"]["beta"]
        threshold = args["BB"]["threshold"]
        batch = args["BB"]["batch"]
        vol = buildings.floor["volume"].detach().clone()
        coor = torch.cat(
            (buildings.identity["X"].view(-1, 1), buildings.identity["Y"].view(-1, 1),),
            dim=1,
        ).double()

        # Batch Vectorize
        # If the number of rows from curr to the end is fewer than batch
        # break the loop and process the remaining.
        # Ensure the matrix size will never exceed the hardware limitations

        curr = 0
        res = None
        while coor[curr:].shape[0] > batch:
            batch_res = self.__get_building_affinity(
                curr, coor[curr : curr + batch], coor, beta, threshold, vol
            )
            res = batch_res if res is None else torch.cat((res, batch_res), 0)
            curr += batch
        # Process the remaining
        batch_res = self.__get_building_affinity(
            curr, coor[curr:], coor, beta, threshold, vol
        )
        res = batch_res if res is None else torch.cat((res, batch_res), 0)

        del beta, threshold, vol, coor, curr, batch_res
        return res

    def __get_building_affinity(self, curr, batch_coor, coor, beta, threshold, vol):
        """
        Parameters
        ----------
        coor : tensor (N, 2), xy-coordinate
        beta : float, hyperparameter 
        threshold: float, hyperparameter
        vol  : tensor (N,1), floorspace volume

        return
        ------
        res  : sparse tensor
        """
        dist = torch.cdist(batch_coor, coor, 2).pow(beta)
        dist[dist == 0] = 1e-5
        end = curr + batch_coor.shape[0]
        scores = vol.div(dist)
        scores[:, curr:end] = scores[:, curr:end].fill_diagonal_(
            0
        )  # offset the columns
        scores = scores.div(scores.sum(1).view(-1, 1))  # normalize
        scores[scores < threshold] = 0

        res = scores.to_sparse()
        del scores, dist
        return res

    def __agent_agent(self, args, agents, buildings, device):
        """
        Initialize Agent - Agent network.

        Nodes
        --------
        Agents

        Edges
        --------
        Composite similarity in temrs of age, income, and household distance.

        Return
        --------
        res : torch.Tensor
        """
        threshold = args["AA"]["threshold"]
        batch = args["AA"]["batch"]

        age_sparse = self.__age_sim(agents, batch, threshold)
        income_sparse = self.__income_sim(agents, age_sparse, device, threshold)
        res = self.__house_dist(agents, batch, income_sparse, buildings, threshold)

        # same house gets max probability
        house_a = agents.identity["house"]
        for uniq_h in house_a.unique():
            indices = (house_a == uniq_h).nonzero().view(-1)
            # create source-target-value
            sparse_small = (
                torch.ones((indices.shape[0], indices.shape[0]))
                .fill_diagonal_(0)
                .to_sparse()
                .coalesce()
            )
            sparse_big = torch.sparse_coo_tensor(
                indices[sparse_small.indices()], sparse_small.values(), res.size()
            )
            res = res.add(sparse_big)
        res = res.coalesce()

        capped_val = res.values()
        capped_val[capped_val > 1] = 1.0
        res = torch.sparse_coo_tensor(res.indices(), capped_val, res.size()).coalesce()

        del threshold, house_a, capped_val
        return res

    def __age_sim(self, agents, batch, threshold):
        """
        Compute age pairwise similarity.

        Return
        -------
        res : torch.Tensor
        """
        age = agents.identity["age"].double()

        curr = 0
        max_diff = torch.max(age) - torch.min(age)
        res = None
        while age[curr:].shape[0] > batch:
            batch_res = age.sub(age[curr : curr + batch].view(-1, 1)).abs()
            batch_res = 1 - (batch_res / max_diff)
            batch_res[batch_res < threshold] = 0
            batch_res = batch_res.to_sparse()
            res = batch_res if res is None else torch.cat((res, batch_res), 0)
            curr += batch
        # Process the remaining
        batch_res = age.sub(age[curr:].view(-1, 1)).abs()
        batch_res = 1 - (batch_res / max_diff)
        batch_res[batch_res < threshold] = 0
        batch_res = batch_res.to_sparse()
        res = batch_res if res is None else torch.cat((res, batch_res), 0).coalesce()

        del age, curr, max_diff, batch_res
        return res

    def __income_sim(self, agents, sparse, device, threshold):
        """
        Compute income pairwise similarity.

        Return
        --------
        res : torch.Tensor
        """
        h = agents.identity["house"]

        # Compute household net income
        j = agents.job
        h_unique = torch.unique(h)
        agent_hh = (h_unique[:, None] == h).long().argmax(dim=0)
        h_income = torch.zeros((h_unique.shape[0],), dtype=torch.float64, device=device)
        for idx in range(h_unique.shape[0]):
            mask = (h == h_unique[idx]) & (j["status"] == 1)
            h_income[idx] = torch.sum(j["income"][mask])
        h_income = h_income[agent_hh]

        sparse_idx = sparse.indices()
        max_diff = torch.max(h_income) - torch.min(h_income)
        income_sim = h_income[sparse_idx[0]].sub(sparse_idx[1]).abs()
        income_sim = 1 - (income_sim / max_diff)

        # Create new sparse matrix
        new_val = sparse.values().mul(income_sim)
        new_val[new_val < threshold] = 0
        mask = new_val.nonzero().view(-1)

        new_idx = torch.stack((sparse_idx[0][mask], sparse_idx[1][mask]))
        res = torch.sparse_coo_tensor(new_idx, new_val[mask], sparse.size())

        del (
            h_unique,
            agent_hh,
            h_income,
            mask,
            new_idx,
            sparse_idx,
            income_sim,
            max_diff,
            new_val,
        )
        return res

    def __house_dist(self, agents, batch, sparse, buildings, threshold):
        """
        Compute household pairwise distance

        Return
        --------
        res : torch.Tensor
        """
        sparse = sparse.coalesce()
        source = sparse.indices()[0]
        target = sparse.indices()[1]

        s_homes = agents.building["house"][source]
        t_homes = agents.building["house"][target]

        coor = torch.cat(
            (buildings.identity["X"].view(-1, 1), buildings.identity["Y"].view(-1, 1),),
            1,
        ).double()

        # compute approximate global max diff
        max_diff = self.__get_building_maxdiff(coor)
        id_b = buildings.identity["idx"]

        curr = 0
        house_dist = None
        while s_homes[curr:].shape[0] > batch:
            s_h_idx = (
                (id_b[:, None] == s_homes[curr : curr + batch]).int().argmax(dim=0)
            )
            t_h_idx = (
                (id_b[:, None] == t_homes[curr : curr + batch]).int().argmax(dim=0)
            )
            s_coor = coor[s_h_idx, :]
            t_coor = coor[t_h_idx, :]
            batch_res = torch.sqrt(torch.square(s_coor - t_coor).sum(1))  # l2 norm
            batch_res = batch_res.to_sparse().coalesce()
            house_dist = (
                batch_res
                if house_dist is None
                else torch.cat((house_dist, batch_res), 0)
            )
            curr += batch

        s_h_idx = (id_b[:, None] == s_homes[curr:]).int().argmax(dim=0)
        t_h_idx = (id_b[:, None] == t_homes[curr:]).int().argmax(dim=0)
        s_coor = coor[s_h_idx, :]
        t_coor = coor[t_h_idx, :]
        batch_res = torch.sqrt(torch.square(s_coor - t_coor).sum(1))  # l2 norm
        batch_res = batch_res.to_sparse().coalesce()
        house_dist = (
            batch_res
            if house_dist is None
            else torch.cat((house_dist, batch_res), 0).coalesce()
        )
        curr += batch

        # Check if any house_dist larger than approximate max diff
        dist_ratio = house_dist.values() / max_diff
        max_idx = dist_ratio.argmax(dim=0)
        if (
            dist_ratio[max_idx] > 1
        ):  # there exists a dist large than approximate global max
            dist_ratio = house_dist.values() / house_dist.values()[max_idx]  # replace

        # normalize
        new_val = 1 - dist_ratio
        mask = house_dist.indices().view(-1)

        # get intersect
        new_val = sparse.values()[mask].mul(new_val)
        new_val[new_val < threshold] = 0
        mask = new_val.nonzero().view(-1)

        # stack
        new_idx = torch.stack((sparse.indices()[0][mask], sparse.indices()[1][mask]))
        res = torch.sparse_coo_tensor(new_idx, new_val[mask], sparse.size())

        del (
            id_b,
            s_h_idx,
            t_h_idx,
            coor,
            s_coor,
            t_coor,
            house_dist,
            new_val,
            mask,
            new_idx,
        )
        return res

    def __get_building_maxdiff(self, coor):
        """
        Find 4 coordinate tuples:
            (global min X, local min Y)
            (global max X, local max Y)
            (local min X, global min Y)
            (local max X, global min Y)
        compute the dist between these 4 coordinates, and get the max

        Parameter
        ---------
        coor : tensor (B, 2) - xy coordinate

        Return
        ---------
        res : global max diff of building distance
        """
        global_X = coor[:, 0].aminmax()
        gminX = coor[:, 0] == global_X.min
        gmaxX = coor[:, 0] == global_X.max
        gminX_lminY = coor[gminX, 1].argmin()
        gmaxX_lmaxY = coor[gmaxX, 1].argmax()

        global_Y = coor[:, 1].aminmax()
        gminY = coor[:, 1] == global_Y.min
        gmaxY = coor[:, 1] == global_Y.max
        gminY_lminX = coor[gminY, 0].argmin()
        gmaxY_lmaxX = coor[gmaxY, 0].argmax()

        tuples = torch.stack(
            (
                coor[gminX, :][gminX_lminY],
                coor[gmaxX, :][gmaxX_lmaxY],
                coor[gminY, :][gminY_lminX],
                coor[gmaxY, :][gmaxY_lmaxX],
            )
        )

        res = torch.cdist(tuples, tuples).amax()
        del (
            tuples,
            global_X,
            gminX,
            gmaxX,
            gminX_lminY,
            gmaxX_lmaxY,
            gminY,
            gmaxY,
            gminY_lminX,
            gmaxY_lmaxX,
        )
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
        # directly construct sparse matrix using agents.identity with mask
        # sum those up and cap to 1 to obtain total
        #
        type_b = ["house", "anchor", "trivial"]
        num_a = agents.identity["id"].shape[0]
        num_b = buildings.identity["id"].shape[0]
        res = dict()
        res["total"] = torch.sparse_coo_tensor(
            size=(num_a, num_b), dtype=torch.long
        ).coalesce()

        for val in type_b:
            a = agents.building[val].long()
            a_nonempty = a != -1

            a_buildings = a[a_nonempty]
            a_involved = a_nonempty.nonzero().view(-1)
            res[val] = torch.sparse_coo_tensor(
                torch.stack((a_involved, a_buildings)),
                torch.ones(a_involved.shape[0], dtype=torch.long),
                size=(num_a, num_b),
                dtype=torch.long,
            ).coalesce()

            res["total"] = res["total"].add(res[val]).coalesce()

        val_capped = res["total"].values()
        val_capped[val_capped > 1] = 1
        res["total"] = torch.sparse_coo_tensor(
            res["total"].indices(), val_capped, res["total"].shape
        ).coalesce()
        del type_b, num_a, num_b
        return res

    def __agent_household(self, agents):
        """
        Create network of agents and households.

        Nodes
        ------
        Agents, Households

        Edges
        ------
        1 betweeen agents and their household

        Shape
        ------
        (Agent, Household)

        Return
        ------
        res : torch.Tensor
        """
        num_a = agents.identity["id"].shape[0]
        num_h = agents.identity["house"].unique().shape[0]

        sparse_val = torch.ones(num_a, dtype=torch.long)
        res = torch.sparse_coo_tensor(
            torch.stack((torch.arange(num_a), agents.identity["house"])),
            sparse_val,
            size=(num_a, num_h),
        ).coalesce()

        del num_a, num_h, sparse_val
        return res

    def __agent_SA(self, agents, buildings):
        """
        Create network of agents and SA.

        Nodes
        ------
        Agents, SA

        Edges
        ------
        1 betweeen agents and their SA

        Shape
        ------
        (Agent, SA)

        Return
        ------
        res : torch.Tensor
        """
        num_a = agents.identity["id"].shape[0]
        num_SA = buildings.identity["area"].unique().shape[0]

        sparse_val = torch.ones(num_a, dtype=torch.long)
        res = torch.sparse_coo_tensor(
            torch.stack((torch.arange(num_a), agents.identity["area"])),
            sparse_val,
            size=(num_a, num_SA),
        ).coalesce()

        del num_a, num_SA, sparse_val
        return res
