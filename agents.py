"""
This module initializes agents.

A subset of the attributes will be loaded directly from csv file.
While the remaining will be randomly initialized.

For the details of the attribute, please refer to readme.
"""

import torch

import util


class Agents(object):
    """
    Specify agents.

    To improve readability, the torch.Tensors will be wrapped using python
    native dictionary structure.

    Attributes
    ----------
    identity            : Identities of the agents.
    building            : Building IDs of the agents.
    job                 : Employment related information.
    risk                : Risks of exposure, contagion, admission, mortality.
    timestamp           : Relative start dates and period
    status              : Contagious status
    """

    def __init__(self, args, b, device):
        """
        Initialize attributes.

        Parameter
        ---------
        args : arguments
        b : initialized building objects
        """
        path = args.agents_dir
        self.device = device
        self.identity = self._init_identity(path)
        self.building, self.job = self._init_building_activity(path, b)
        self.risk = self._init_risk(args, self.identity["age"])
        self.timestamp = dict()  # add sick period
        self.status = self._init_status()

    def _init_identity(self, path):
        """
        Initialize identities.

        Load real data for agent, house, disability, age group.
        Randomly initalize specific age and religious status.

        Paramter
        --------
        path    : str

        Return
        --------
        res     : dict
        """
        data = util.load_csv(path, self.device, cols=[0, 11, 2, 6], nrows=100)
        keys = ["id", "house", "disable", "group"]
        res = dict()

        for idx in range(len(keys)):
            res[keys[idx]] = data[:, idx]
        res["age"] = self._group_to_age(res["group"])
        res["religious"] = self._assign_religious(res["house"])

        return res

    def _group_to_age(self, group):
        """
        Randomly assign specific age according to groups.

        Parameter
        ---------
        group   : torch.Tensor

        Return
        ---------
        res     : torch.Tensor
        """
        masks = {"child": group == 1, "adult": group == 2, "elderly": group == 3}
        lower = {"child": 1, "adult": 19, "elderly": 66}
        upper = {"child": 19, "adult": 66, "elderly": 91}
        res = torch.zeros((group.shape[0],), device=self.device).byte()

        for k, v in masks.items():
            res[v] = torch.randint(
                lower[k], upper[k], (v.count_nonzero().item(),), device=self.device
            ).byte()

        return res

    def _assign_religious(self, house):
        """
        Randomly assign religious status to family.

        Parameter
        ---------
        house   : torch.Tensor

        Return
        ---------
        res     : torch.Tensor
        """
        res = torch.zeros((house.shape[0],), device=self.device).bool()
        house_id = house.unique()
        house_num = house_id.shape[0]
        is_religious = torch.randint(0, 2, (house_num,), device=self.device).bool()

        for idx in range(house_num):
            mask = house == house_id[idx]
            res[mask] = is_religious[idx]

        return res

    def _load_house(self, path):
        """
        Initialize buildings.

        Load real data for house building id.

        Paramter
        --------
        path    : str

        Return
        --------
        res     : dict
        """
        res = dict()

        res["house"] = torch.round(
            util.load_csv(path, self.device, cols=[1], nrows=100).view(-1)
        )

        return res

    def _load_employ(self, path):
        """
        Initialize employments.

        Load real data for status, local, income.
        Nullify children's employment info.
        Align status for local worker.

        Paramter
        --------
        path    : str

        Return
        --------
        res     : dict
        """
        data = util.load_csv(path, self.device, cols=[3, 12, 10], nrows=100)
        keys = ["status", "local", "income"]
        mask = ~(self.identity["group"] == 1)
        res = dict()
        for idx in range(len(keys)):
            res[keys[idx]] = data[:, idx] * mask

        align = (res["status"] == 0) & (res["local"] == 1)
        res["status"][align] = 1
        houses = self.identity["house"].unique()

        for h in houses:  # adjust income
            mask_h = self.identity["house"] == h  # same house
            mask_i = res["status"] == 1  # employed
            mask = mask_h & mask_i
            num = mask.count_nonzero().item()
            if num > 0:
                res["income"][mask] /= num
            else:
                res["income"][mask] = 0

        return res

    def _init_building_activity(self, path, buildings):
        """
        Initialize buildings and activities.

        Parameter
        ---------
        buildings   : dict

        Return
        ---------
        b : dict (building)
        j : dict (jobs)
        """
        b = self._load_house(path)
        j = self._load_employ(path)

        jobs = self._create_job(buildings)
        mask = j["local"] == 1
        j["job"] = self._assign_job(jobs, mask, j)
        idx = self._anchor_index(
            self.identity["religious"], self.identity["age"], j["job"]
        )
        b["anchor"] = self._assign_anchor(idx, buildings.activity, jobs)
        b["trivial"] = self._assign_trivial(idx, buildings.activity)

        return b, j

    def _create_job(self, b):
        """
        Create jobs for downstream assignments.

        Paramter
        --------
        b : Buildings object

        Return
        --------
        res     : dict
        """
        res = dict()
        job_density = torch.Tensor([0, 0.0004733, 0, 0.0315060, 0.0479474]).to(
            self.device
        )
        avg = 7177.493  # average income
        std = 1624.297  # standard income

        job_floor = b.floor["volume"][:, None] * job_density
        land_use = b.land["initial"].view(-1, 1)
        land_use = torch.round(land_use).long()
        job_num = torch.gather(job_floor, 1, land_use)
        job_num = torch.round(job_num).int()
        total = torch.sum(job_num).item()

        res["income"] = torch.zeros((total,), device=self.device)
        res["building"] = torch.zeros((total,), device=self.device)
        res["vacant"] = torch.ones((total,), device=self.device)
        res["id"] = torch.arange(total, device=self.device) + 1

        cnt = 0
        for idx, val in enumerate(job_num):
            if val.item() == 0:
                continue
            else:
                num = val.item()
            res["income"][cnt : cnt + num] = torch.normal(
                avg, std, size=(num,), device=self.device
            )
            res["building"][cnt : cnt + num] = b.identity["id"][idx].expand(num)
            cnt += num

        return res

    def _assign_job(self, jobs, mask, j):
        """
        Assign created jobs to agents.

        Parameter
        ---------
        jobs    : dict
        mask    : torch.Tensor
        j       : dict (agent)

        Return
        ---------
        res     : torch.Tensor
        """
        num_j = jobs["id"].shape[0]  # number of jobs
        num_a = mask.count_nonzero()  # number of agents
        idx_a = torch.nonzero(mask)
        limit = num_j if num_j < num_a else num_a
        random = torch.randperm(num_a, device=self.device)
        res = torch.zeros((mask.shape[0],), device=self.device, dtype=torch.int64)

        while limit > 0:
            idx_j = torch.nonzero(jobs["vacant"])  # index of vacancies
            agent = idx_a[random[limit - 1]]  # randomly pick a work
            job = jobs["income"][idx_j].sub(j["income"][agent]).abs().argmin()
            res[agent] = jobs["id"][idx_j[job]]  # assign
            j["income"][agent] = jobs["income"][idx_j[job]]  # update income
            jobs["vacant"][idx_j[job]] = 0
            mask[agent] = 0
            limit -= 1

        # remove jobless agent from work force
        idx_a = torch.nonzero(mask)
        j["status"][idx_a] = 0
        j["local"][idx_a] = 0

        return res

    def _anchor_index(self, religious, age, job):
        """
        Return index with anchor activity.

        Parameter
        ---------
        job : torch.Tensor
        religious : torch.Tensor
        age : torch.Tensor

        Return
        ---------
        res : dict
        """
        r_idx = religious == 1
        kinder = age < 7
        elem = (age >= 7) & (age < 15)
        high = (age >= 15) & (age < 19)
        uni = (age >= 19) & (age < 25)

        res = {
            "job": job,
            "school0": ~r_idx & kinder,
            "school1": ~r_idx & elem,
            "school2": ~r_idx & high,
            "schoolR0": r_idx & kinder,
            "schoolR1": r_idx & elem,
            "schoolR2": r_idx & high,
            "schoolR3": r_idx & uni,
            "etc": ~r_idx,
            "etcR": r_idx,
        }

        return res

    def _assign_anchor(self, idx, a, jobs):
        """
        Initialize buildings for anchor activities.

        Parameter
        ---------
        idx     : dict
        a       : dict
        jobs    : dict

        Return
        ---------
        res     : torch.Tensor
        """
        res = torch.zeros((idx["job"].shape[0],), device=self.device)
        has_job = torch.nonzero(idx["job"])

        for agent in has_job:
            mask = jobs["id"] == idx["job"][agent]
            res[agent] = jobs["building"][torch.nonzero(mask)]

        "====================================================="

        anchor = [x for x in idx.keys() if "school" in x]
        anchor += [x for x in idx.keys() if "etc" in x]

        for key in anchor:

            val = (res == 0) & idx[key] if "etc" in key else idx[key]
            agents = torch.nonzero(val).view(-1)
            num_a = agents.shape[0]
            building = a[key]
            num_b = building.shape[0]
            if num_b != 0:
                random = torch.randint(num_b, size=(num_a,), device=self.device)
                for i in range(num_a):
                    res[agents[i]] = building[random[i]]
            if "etc" in key:
                mask = torch.randint(2, size=(num_a,), device=self.device)
                res[agents] *= mask

        return res

    def _assign_trivial(self, idx, a):
        """
        Initialize buildings for trivial activities.

        Parameter
        ---------
        idx     : dict
        a       : dict

        Return
        ---------
        res     : torch.Tensor
        """
        res = torch.zeros((idx["job"].shape[0],), device=self.device)
        etc = [x for x in idx.keys() if "etc" in x]
        for key in etc:
            val = idx[key]
            agents = torch.nonzero(val).view(-1)
            num_a = agents.shape[0]
            building = a[key]
            num_b = building.shape[0]
            if num_b != 0:
                random = torch.randint(num_b, size=(num_a,), device=self.device)
                for i in range(num_a):
                    res[agents[i]] = building[random[i]]
            mask = torch.randint(2, size=(num_a,), device=self.device)
            res[agents] *= mask

        return res

    def _init_risk(self, args, age):
        """
        Initialize the risks.

        Parameter
        ---------
        args : arguments

        Return
        ---------
        res : dict
        """
        res = dict()
        risks = {
            "infection": util.load_csv(args.infection_dir, self.device),
            "admission": util.load_csv(args.admission_dir, self.device),
            "mortality": util.load_csv(args.mortality_dir, self.device),
        }
        for key, val in risks.items():
            risk = torch.zeros((age.shape[0],), device=self.device)
            for row in val:
                mask = (age >= row[0]) & (age < row[1])
                risk[mask] = torch.normal(
                    row[2],
                    row[3],
                    (mask.count_nonzero().item(),),
                    device=self.device,
                )
            risk[risk < 0] = 0
            res[key] = risk
        res["exposure"] = torch.rand(age.shape[0], device=self.device)

        return res

    def _init_status(self):
        """
        Initialize the status.

        Parameter
        ---------
        num : int

        Return
        ---------
        res : torch.Tensor
        """
        num = self.identity["id"].shape[0]
        res = torch.ones((num), device=self.device)
        infected = torch.randperm(num, device=self.device)[:20]
        res[infected] = 2

        return res
