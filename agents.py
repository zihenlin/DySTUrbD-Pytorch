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
    risk                : Risks of exposure, infection, admission, mortality.
    timestamp           : Relative start dates and period
    status              : Contagious status
    routine             : The originally initiated routine
    daily_infection     : Daily infection matrix for downstream analysis
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
        self.start = self._init_start_date(self)
        self.period = self._init_period(self)
        self.status = self._init_status()
        self.daily_infection = []

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
        data = util.load_csv(path, self.device, cols=[0, 11, 2, 6], nrows=1000)
        keys = ["id", "house", "disable", "group"]
        res = dict()

        for idx in range(len(keys)):
            res[keys[idx]] = data[:, idx]

        h_unique = res["house"].unique()
        for idx in range(h_unique.shape[0]):
            mask = res["house"] == h_unique[idx]
            res["house"][mask] = idx

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

    def _load_house(self, path, b):
        """
        Initialize buildings.

        Load real data for house building id.

        Paramter
        --------
        path    : str
        b       : Buildings object

        Return
        --------
        res     : dict
        """
        res = dict()

        res["house"] = torch.round(
            util.load_csv(path, self.device, cols=[1], nrows=1000).view(-1)
        )
        self.identity["area"] = torch.zeros_like(res["house"])
        for idx in range(b.identity["idx"].shape[0]):
            mask = res["house"] == b.identity["id"][idx]
            res["house"][mask] = idx
            self.identity["area"][mask] = b.identity["area"][idx]

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
        data = util.load_csv(path, self.device, cols=[3, 12, 10], nrows=1000)
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
        b = self._load_house(path, buildings)
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
        res["building"] = torch.zeros((total,), device=self.device, dtype=torch.int64)
        res["vacant"] = torch.ones((total,), device=self.device)
        res["id"] = torch.arange(total, device=self.device, dtype=torch.int64) + 1

        cnt = 0
        for idx, val in enumerate(job_num):
            if val.item() == 0:
                continue
            else:
                num = val.item()
            res["income"][cnt : cnt + num] = torch.normal(
                avg, std, size=(num,), device=self.device
            )
            res["building"][cnt : cnt + num] = idx
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
        res = torch.zeros((idx["job"].shape[0],), dtype=torch.int64, device=self.device)
        has_job = torch.nonzero(idx["job"])

        for agent in has_job:
            mask = jobs["id"] == idx["job"][agent]
            res[agent] = jobs["building"][mask]

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
                res[agents] &= mask

        return re

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
        res = torch.zeros((idx["job"].shape[0],), dtype=torch.int64, device=self.device)
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

        return res

    def _init_start_date(self):
        """
        Initalize sick, quarantine, admission start dates

        Return
        -------
        res : dict
        """
        res = dict()
        keys = ["sick", "quarantine", "admission"]
        for key in keys:
            res[key] = torch.zeros((self.identity["id"].shape[0],), device=self.device)

        return res

    def _init_period(self):
        """
        Initalize sick, quarantine, admission period

        Return
        -------
        res : dict
        """
        res = dict()
        keys = ["sick", "quarantine", "admission"]
        for key in keys:
            res[key] = torch.zeros((self.identity["id"].shape[0],), device=self.device)

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

    def set_interaction(self, prob_AA):
        """
        Set the interaction matrix of agents obtained from shortest path.

        Parameter
        ----------
        prob_AA : torch.Tensor (A, A)
        """
        self.interaction = prob_AA

    def set_routine(self, routine):
        """
        Set the routine of agents obtained from shortest path and updates.

        Parameter
        ----------
        routine : torch.Tensor (A, A)
        """
        self.routine = routine

    def get_infected(self):
        """
        Return mask of infected agents.
        """
        return (
            (self.status == 2)
            | (self.status == 4)
            | (self.status == 5)
            | (self.status == 6)
        )

    def get_quarantined(self):
        """
        Return mask of quarantinead agents.
        """
        return (self.status == 3) | (self.status == 4) | (self.status == 5)

    def get_hospitalized(self):
        """
        Return mask of hospitalized agents.
        """
        return self.status == 6

    def get_recovered(self):
        """
        Return mask of recovered agents.
        """
        return self.status == 7

    def get_dead(self):
        """
        Return mask of dead agents.
        """
        return self.status == 8

    def get_no_admit(self):
        """
        Return mask of agents who are not suitable to be hospitalized.

        Status
        -------
        6. Hospitalized
        7. Recovered
        8. Dead
        """
        return (
            (self.period["sick"] < 4) | (self.period["sick"] > 14) | (self.status >= 6)
        )

    def get_no_death(self):
        """
        Return mask of agents who unlikely to die.
        """
        return self.period["admission"] < 3

    def update_routine(self, b_status, network_house):
        """
        Update routine of quaratine, hospitalized, and dead agents.
        """
        a_qua = self.get_quarantined()
        a_hos = self.get_hospitalized()
        a_dead = self.get_dead()
        res = self.routine
        res *= b_status  # building open or close
        res[a_qua] = network_house[a_qua]
        res[a_hos] &= False
        res[a_dead] &= False

        return res

    def update_period(self, day):
        """
        Update the periods of agents

        Parameter
        ---------
        day : int
        """
        a_qua = self.get_quarantined()
        a_hos = self.get_hospitalized()
        a_dead = self.get_dead()
        self.period["sick"][a_inf] = day - self.start["sick"]
        self.period["quarantine"][a_qua] = day - self.start["quarantine"]
        self.period["admission"][a_hos] = day - self.start["admission"]

    def reset_period(self, mask, key):
        """
        Reset period of agents.
        """
        self.period[key][mask] = 0

    def update_admission(self, day):
        """
        Update admission probability of agents and agent status if admitted.

        Parameter
        ---------
        day : int

        Return
        -------
        res : int

        #TODO: does quarantine period reset after admission
        """
        a_no_admit = self.get_no_admit()
        a_uninf = (self.status == 1) | (self.status == 3)
        tmp_risk = self.risk["admission"]
        tmp_risk *= ~(a_uninf | a_no_admit).int()  # remove admission risk
        rand_threshold = torch.zeros_like(tmp_risk).uniform_(0, 1)
        admission = tmp_risk > rand_threshold

        self.update_status(admission, 6)
        self.update_start(admission, "admission", day)
        self.reset_period(admission, "quarantine")
        self.update_start(admission, "quarantine", 0)

        res = admission.count_nonzero()

        return res

    def update_status(self, mask, status):
        """
        Update status of selected agents

        Parameter
        ----------
        mask : torch.Tensor (A, 1)
        status : int
        """
        self.status[mask] = status

    def update_start(self, mask, key, day):
        """
        Update start date of selected agents for the given key and day.

        Parameter
        ----------
        mask : torch.Tensor (A, 1)
        key : string
        day : int
        """
        self.start[key][mask] = day

    def update_death(self):
        """
        Update dead probability of agents and agent status if died.

        Return
        -------
        res : int
        """
        a_no_death = self.get_no_death()
        a_unhos = self.status != 6
        tmp_risk = self.risk["mortality"]
        tmp_risk *= ~(a_unhos | a_no_death).int()  # remove admission risk
        rand_risk = torch.zeros_like(tmp_risk).uniform_(0, 1)
        death = tmp_risk > rand_risk
        self.update_status(death, 8)
        res = death.count_nonzero()

        return res

    def get_exposed_risk(self, routine, gamma):
        """
        Compute the risk of exposed agents using interaction probabiltiy
        and official risk distribution.
        Return a sparse matrix with exposed agents and their interaction
        with infected agents.

        Parameter
        ---------
        routine : torch.Tensor

        Return
        ---------
        res : torch.Tensor (A,A)

        """
        a_inf = self.get_infected()
        b_inf = (
            routine[a_inf].sum(0).bool()
        )  # get a list of buildings visited by infected agents
        a_exposed = (routine.add(b_inf) == 2).sum(
            1
        ) & ~a_inf  # uninfected agents visited same buildings

        contagious_strength = gamma.log.prob(self.period["sick"]).exp()

        res = self.interaction * contagious_strength
        res *= a_inf  # only interaction with infected agent
        res *= self.risk["infection"].view(-1, 1)
        res *= a_exposed.view(-1, 1)  # only exposed agents
        res *= 0.08  # normalize factor

        return res

    def update_infection(self, day, routine, gamma):
        """
        Update infection probability of agents and agent status if infected.

        Parameter
        ---------
        day : int
        rountine : torch.Tensor

        Return
        -------
        res1 : number of newly infected (not quarantined) agent
        res2 : number of newly infected and quarantined agent
        infection : daily infection matrix

        #TODO: Shall we add a workplace or similar thematic network to detect any potential cluster?
        """
        a_qua = self.get_quarantined()
        sparse_risk = self.get_exposed_risk(self, routine, gamma)

        rand_threshold = torch.zeros_like(sparse_risk).uniform_(0, 1)
        infection = sparse_risk > rand_threshold

        inf_qua = infection & a_qua
        inf_no_qua = infection & ~a_qua

        self.update_status(inf_no_qua, 2)
        self.update_start(inf_no_qua, "sick", day)
        self.update_status(inf_qua, 5)
        self.update_start(inf_qua, "sick", day)

        res1 = infection.count_nonzero()
        res2 = inf_qua.count_nonzero()

        return res, res2, infection

    def end_quarantine(self):
        """
        Restore the status of agents who have sufficient days of quarantine.
        """
        threshold = 7  # hyperparam
        a_end_qua = self.period["quarantine"] == threshold
        a_healthy = self.status == 3
        a_undiagnosed = self.status == 4
        a_end_qua &= a_healthy | a_undiagnosed

        self.update_status(a_end_qua & a_healthy, 1)  # healthy agents
        self.update_status(a_end_qua & a_undiagnosed, 2)  # undiagnosed infected agents
        self.reset_period(a_end_qua, "quarantine")
        self.update_start(a_end_qua, "quarantine", 0)  # reset

    def update_diagnosis(self, day):
        """
        Turn the status of agents with sufficient days of infection into diagnosed.
        Send these free agents to quarantine.

        Parameter
        ---------
        day : int

        Return
        ---------
        res : int
        """
        threshold = 7  # hyperparam
        a_diagnosed = self.period["sick"] == threshold
        a_free = self.status == 2
        a_qua = self.status == 4
        new_qua = a_diagnosed & a_free
        a_diagnosed &= a_free | a_qua

        self.update_start(new_qua, "quarantine", day)
        self.update_status(a_diagnosed, 5)

        res = new_qua.count_nonzero()

        return res

    def update_diagnosed_family(self, day, network_house):
        """
        Detect newly diagnosed agents and send their family to quarantine

        Idea:
        network_house (A, B) is many-to-one, i.e., each row contains only one value.
        Using argmax, we obtained the household index of the newly diagnosed agents.
        network_house tranpose (B,A) presents one-to-many, one building corresponds to multiple agents.
        Using tensor.nonzero(as_tuple=True), we obtain a list of agents which share the same households
        with newly diagnosed agents.

        tensor.nonzero(as_tuple=True) returns (torch.Tensor(row indices), torch.Tensor(column indices)).
        In our context, torch.Tensor(column indices) represents agents in the infected household.

        Parameter
        ---------
        day : int
        network_house : torch.Tensor

        Return
        ---------
        res : int
        """
        threshold = 7

        a_new_diag = (self.status == 5) & (self.period["sick"] == threshold)
        a_qua = self.get_quarantined()  # get family family already in quarantine
        idx_h = network_house[a_new_diag].argmax(dim=1)
        h_to_a = network_house.t()

        idx_family = h_to_a[idx_h].nonzero(as_tuple=True)[1]
        a_family = torch.zeros_like(self.status).long()
        a_family[idx_h] = 1
        a_family &= ~a_new_diag
        a_family &= ~a_qua

        a_healthy = a_family & (self.status == 1)
        a_infected = a_family & (self.status == 2)

        self.update_start(a_family, "quarantine", day)
        self.update_status(a_healthy, 3)
        self.update_status(a_infected, 4)

        res = a_family.count_nonzero()

        return res

    def update_recovery(self):
        """
        Recover the sick agents in quarantine or hospital.
        #TODO: A list of threshold to increase variance of latent period and recovery period?

        Return
        -------
        res : int
        """
        threshold = [21, 28]

        a_qua = (self.status == 5) & (self.period["sick"] == threshold[0])
        a_hos = (self.status == 6) & (self.period["sick"] == threshold[1])

        self.update_status(a_qua | a_hos, 7)
        self.reset_period(a_qua, "quarantine")
        self.update_start(a_qua, "quarantine", 0)
        self.reset_period(a_hos, "admission")
        self.update_start(a_hos, "admission", 0)

        res = (a_qua | a_hos).count_nonzero()

        return res
