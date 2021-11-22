"""
This model simulate the dynamics of a City under Covid-19 pandemic.

DySTUrbD-epi: Model
This program is the implementation of DySTUrbD-Epi class.
"""

import torch
from scipy.sparse import csr_matrix
import psutil  # get number of CPU cores
from time import time
import numpy as np

import buildings
import agents
import networks
import dijkstra_mp64  # multiprocess shortest path


class DySTUrbD_Epi(object):
    """Simulate the world."""

    def __init__(self, args, count):
        """
        Initialize the world.

        Parameter
        ---------
        args : arguments
        """
        self.res = {
            "Sim": count,
            "Time": {},
            "Results": {"Stats": {}, "Buildings": {}, "SAs": {}, "IO_mat": {}},
        }
        self.time = time()
        print("Init start!")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Current Device:", self.device)

        self.cpu = len(psutil.Process().cpu_affinity())
        t1 = time()
        print("Number of CPU:", self.cpu)

        self.buildings = buildings.Buildings(args.buildings_dir, self.device)
        t2 = time()
        self._log_time("Buildings", t2 - t1)

        self.agents = agents.Agents(args, self.buildings, self.device)
        t3 = time()
        self._log_time("Agents", t3 - t2)

        self.network = networks.Networks(self.agents, self.buildings, self.device)
        t4 = time()
        self._log_time("Networks", t4 - t3)

        dist = self._get_dist()
        t5 = time()
        self._log_time("ShortestPath", t5 - t4)

        prob_AA = self._prob_AA(dist)
        self.agents.set_interaction(prob_AA)
        t6 = time()
        self._log_time("Interaction", t6 - t5)

        routine = self._get_routine(dist)
        self.agents.set_routine(routine)
        t7 = time()
        self._log_time("Routine", t7 - t6)

        self.gamma = self._get_gamma()
        t8 = time()
        self._log_time("Gamma", t8 - t7)

        print()
        print("Init Complete!")

    def __call__(self):
        """
        Run the model, Gogogo!
        """
        self._simulate()

    def _log_time(self, key, time):
        """
        Save the computational time to model.res
        """
        self.res["Time"][key] = time
        print(f"{key}: {time}")

    def _log_stats(self, day, key, data):
        """
        Save and print the essential data generated from simulation.
        """
        self.res["Results"]["Stats"][day][key] = data
        print(f"{key}: {data}")

    def _get_gamma(self):
        """
        Get gamma distribution with specific parameters.
        """
        alpha = (4.5 / 3.5) ** 2
        beta = 4.5 / (3.5 ** 2)  # 1 / scale
        res = torch.distributions.gamma.Gamma(alpha, beta)
        res.support = torch.distributions.constraints.greater_than_eq(0)

        return res

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

        res = dijkstra_mp64.multiSearch(csr, self.cpu)[0]
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
        num_aa = self.network.AA.shape[0]  # total number of agents
        num_bb = self.network.BB.shape[0]  # total number of buildings
        idx_0 = torch.arange(num_aa)
        dist_ab = 1  # Given distance threshold for AB network
        dist_bb = 6  # Given distance threshold for BB network

        nodes_ab = dist[:num_aa, -num_bb:] < dist_ab  # a_nodes (A,B)
        nodes_bb = dist[-num_bb:, -num_bb:] < dist_bb  # all buildings (A,B,B)
        dummy_bb = torch.zeros((1, num_bb), device=self.device)  # empty row
        nodes_bb = torch.cat((nodes_bb, dummy_row), 0)

        res = self.network.AB["total"]
        start = ["house", "anchor", "trivial"]
        end = ["anchor", "trivial", "house"]

        for s, e in zip(start, end):
            nodes_end = self.network.AB[e]
            dummy_end = ~(nodes_end.sum(1).bool())  # point to empty row
            nodes_end = torch.cat((nodes_end, dummy_end.view(-1, 1)), 1)
            idx_end = nodes_end.long().argmax(
                dim=1
            )  # get the end building index for each agent
            near_end = nodes_bb.index_select(0, idx_end).bool()
            for cnt in range(2):
                template = torch.zeros_like(self.network.AB["total"])

                nodes_start = self.network.AB[s] if cnt < 1 else choice
                dummy_start = ~(nodes_start.sum(1).bool())  # point to empty row
                nodes_start = torch.cat((nodes_start, dummy_start.view(-1, 1)), 1)
                idx_start = nodes_start.long().argmax(
                    dim=1
                )  # get the start building index for each agent
                near_start = nodes_bb.index_select(0, idx_start).bool()

                candidates = near_start & near_end  # overlap between two buildings
                candidates = (candidates | nodes_ab) & ~(self.network.AB["total"])
                idx_1 = candidates.double().multinomial(
                    1
                )  # randomly pick one from each row
                template[idx_0, idx_1] = 1
                choice = template & candidates
                res |= choice  # add new building to routine (some are zeros)

        return res

    def _compute_R(self, mask, today):
        """
        Compute R value using number of new cases and expected infectious risk.

        Parameter
        ---------
        day : int
        mask : torch.Tensor

        Return
        ---------
        res : int
        """
        sum_I = 0
        recover = 21  # hyperparameter
        new_inf = mask & (self.agents.period["sick"] == today)
        new_inf = new_inf.count_nonzero()

        for day in range(1, recover):
            cnt = mask & (self.agents.period["sick"] == day)
            cnt = cnt.count_nonzero()
            contagious_strength = self.gamma.log.prob(day).exp()
            sum_I += cnt * contagious_strength

        res = new_inf / sum_I if sum_I > 0 else 0

        return res

    def _get_SA_R(self, day):
        """
        Compute R value for each statistical region.

        Parameter
        ---------
        day : int

        Return
        --------
        res : dict
        res
        """
        sas = self.buildings.identity["area"].unique()
        res = {}

        for sa in sas:
            sa_a = self.agents.identity["area"] == sa
            res[sa] = self._compute_R(sa_a, day)

        return res

    def _get_building_inf(self):
        """
        Compute the infection ratio in each building

        Idea:
        1. Calculate total number of resident in each building
            using AB network using sum
        2. Calculate mask infected agents and calculate infected agents
            in each building using sum

        Return
        -------
        inf : torch.Tensor
        ratio : torch.Tensor
        """
        ab_house = self.networks.AB["house"]
        a_inf = self.agents.get_infected()
        inf_house = ab_house.mul(a_inf)

        total = ab_house.sum(0)
        inf = inf_house.sum(0)

        return inf, inf.div(total.view(-1, 1))

    def _simulate(self):
        """
        Simulate policy response.

        status              : Contagious status
        1. Susceptible
        2. Infected, Undiagnosed
        3. Quarantined, Uninfected
        4. Quarantined, Infected, Undiagnosed
        5. Quarantined, Infected, Diagnosed
        6. Infected, Hospitalized
        7. Recovered
        8. Dead
        """
        num_inf = self.agents.get_infected().count_nonzero()
        day = 1
        while num_inf > 0:

            """
            Computation to obtain data
            """
            t1 = time()
            routine = self.agents.update_routine(
                self.buildings.status, self.network.AB["house"]
            )  # A copy of updated routine
            t2 = time()
            self._log_time("Update Rountine", t2 - t1)

            self.agents.update_period(day)
            t3 = time()
            self._log_time("Update Period", t3 - t2)

            new_admission = self.agents.update_admission(day)
            t4 = time()
            self._log_time("Update Admission", t4 - t3)

            new_death = self.agents.update_death()
            t5 = time()
            self._log_time("Update Death", t5 - t4)

            new_inf, new_qua = self.agents.update_infection(day, routine, self.gamma)
            t6 = time()
            self._log_time("Update Infection", t6 - t5)

            self.agents.end_quarantine()
            t7 = time()
            self._log_time("End Quarantine", t7 - t6)

            new_qua += self.agents.update_diagnosis(day)
            t8 = time()
            self._log_time("Update Diagnosis", t8 - t7)

            new_qua += self.agents.update_diagnosed_family(day, self.network.AH)
            t9 = time()
            self._log_time("Update Diagnosed Family", t9 - t8)

            new_recovered = self.agents.update_recovery()
            t10 = time()
            self._log_time("Update Recovery", t10 - t9)

            num_inf = self.agents.get_infected.count_nonzero()
            num_qua = self.agents.get_quarantined().count_nonzero()
            num_death = self.agents.get_dead().count_nonzero()
            num_admission = self.agents.get_hospitalized().count_nonzero()
            num_recovered = self.agents.get_recovered().count_nonzero()
            R_total = self._compute_R(torch.ones_like(self.agents.status), day)
            R_sa = self._get_SA_R(day)
            num_closed = self.buildings.get_closed().count_nonzero()
            total_inf = (
                num_inf
                if day == 0
                else self.res["Results"]["Stats"][day - 1]["Total Infections"] + new_inf
            )
            num_susceptible = self.agents.identity["id"].shape[0] - total_inf
            b_inf, b_ratio = self._get_building_inf()

            """
            Logging data to output
            """
            res = {}
            self.res["Results"]["SAs"][day] = R_sa
            self.res["Results"]["Buildings"][day]["inf"] = b_inf.tolist()
            self.res["Results"]["Buildings"][day]["ratio"] = b_ratio.tolist()
            self.res["Results"]["Stats"][day] = {}
            self._log_stats(day, "Total Infections", total_inf)
            self._log_stats(day, "Active Infections", num_inf)
            self._log_stats(day, "Daily Infections", new_inf)
            self._log_stats(day, "Total Recovered", num_recovered)
            self._log_stats(day, "Daily Recovered", new_recovered)
            self._log_stats(day, "Total Quarantined", num_qua)
            self._log_stats(day, "Daily Quarantined", new_qua)
            self._log_stats(day, "Total Hospitalized", num_admission)
            self._log_stats(day, "Daily Hospitalized", new_admission)
            self._log_stats(day, "Total Deaths", num_death)
            self._log_stats(day, "Daily Deaths", new_death)
            self._log_stats(day, "R_total", R_total)
            self._log_stats(day, "Closed Buildings", num_closed)
            self._log_stats(day, "Susceptible Agents:", num_susceptible)

            day += 1

            break
