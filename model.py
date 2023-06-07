"""
This model simulate the dynamics of a City under Covid-19 pandemic.

DySTUrbD-epi: Model
This program is the implementation of DySTUrbD-Epi class.
"""

import torch
from scipy.sparse import csr_matrix
import psutil  # get number of CPU cores
from time import time
import json
import gc
from datetime import datetime
import os

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
        self.theme = args["theme"]
        self.scenario = args["scenario"]
        self.disease = args["disease"]
        self.profile = args["profile"]
        self.debug = args["debug"]
        self.out_dir = args["files"]["out_dir"]
        self.count = count
        self.res = {
            "Results": {
                "Time": {},
                "Stats": {},
                "SAs": {},
            },
            "Buildings": {},
            "IO_mat": {},
            "daily_infection": {},
        }
        self.time = time()
        print("Init start!")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Current Device:", self.device)

        self.cpu = len(psutil.Process().cpu_affinity())
        t1 = time()
        print("Number of CPU:", self.cpu)

        self.buildings = buildings.Buildings(
            args,
            self.device,
        )
        t2 = time()
        self._log_time("Buildings", t2 - t1)

        self.agents = agents.Agents(args, self.buildings, self.device)
        t3 = time()
        self._log_time("Agents", t3 - t2)

        self.network = networks.Networks(args, self.agents, self.buildings, self.device)
        t4 = time()
        self._log_time("Networks", t4 - t3)

        dist = self._get_dist()
        t5 = time()
        self._log_time("ShortestPath", t5 - t4)

        prob_AA = self._prob_AA(dist)
        self.agents.set_interaction(prob_AA)
        t6 = time()
        self._log_time("Interaction", t6 - t5)

        routine = self._get_routine(args, dist)
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
        self.res["Results"]["Time"][key] = time
        if self.profile:
            print(f"{key}: {time}")

    def _log_stats(self, day, key, data):
        """
        Save and print the essential data generated from simulation.
        """
        self.res["Results"]["Stats"][day][key] = data
        print(f"{key:-<20}{data:->20}")

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
        AA = self.network.AA.detach().clone()
        BB = self.network.BB.detach().clone()
        AB = self.network.AB["total"].detach().clone()
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

    def _get_routine(self, args, dist):
        """
        Get a list of buildings visited by each agent.

        Return
        -------
        res : torch.Tensor (A, B)
        """
        num_aa = self.network.AA.shape[0]  # total number of agents
        num_bb = self.network.BB.shape[0]  # total number of buildings
        idx_0 = torch.arange(num_aa)
        dist_ab = args["distance"]["dist_ab"]  # Given distance threshold for AB network
        dist_bb = args["distance"]["dist_bb"]  # Given distance threshold for BB network

        nodes_ab = dist[:num_aa, -num_bb:] < dist_ab  # a_nodes (A,B)
        nodes_bb = dist[-num_bb:, -num_bb:] < dist_bb  # all buildings (A,B,B)
        dummy_bb = torch.zeros((1, num_bb), device=self.device)  # empty row
        nodes_bb = torch.cat((nodes_bb, dummy_bb), 0)

        res = self.network.AB["total"].detach().clone()
        start = ["house", "anchor", "trivial"]
        end = ["anchor", "trivial", "house"]

        for s, e in zip(start, end):
            nodes_end = self.network.AB[e].detach().clone()
            dummy_end = ~(nodes_end.sum(1).bool())  # point to empty row
            nodes_end = torch.cat((nodes_end, dummy_end.view(-1, 1)), 1)
            idx_end = nodes_end.long().argmax(
                dim=1
            )  # get the end building index for each agent
            near_end = nodes_bb.index_select(0, idx_end).bool()
            for cnt in range(2):
                template = torch.zeros_like(self.network.AB["total"])

                nodes_start = self.network.AB[s].detach().clone() if cnt < 1 else choice
                dummy_start = ~(nodes_start.sum(1).bool())  # point to empty row
                nodes_start = torch.cat((nodes_start, dummy_start.view(-1, 1)), 1)
                idx_start = nodes_start.long().argmax(
                    dim=1
                )  # get the start building index for each agent
                near_start = nodes_bb.index_select(0, idx_start).bool()

                candidates = near_start & near_end  # overlap between two buildings
                candidates = (candidates | nodes_ab) & ~(self.network.AB["total"])

                idx_weight = candidates == True
                weight = candidates.detach().clone().double()
                if args["multinomial"]["symmetry"] is True:
                    weight[:] = 1.0
                else:
                    weight[idx_weight] = args["multinomial"]["asymmetry"]["possible"]
                    weight[~idx_weight] = args["multinomial"]["asymmetry"]["impossible"]
                idx_1 = weight.multinomial(1)  # randomly pick one from each row
                template[idx_0, idx_1.view(-1)] = 1
                choice = template & candidates

                res |= choice  # add new building to routine (some are zeros)

        del (idx_0, nodes_ab, nodes_bb, dummy_bb)

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
        recover = self.disease["recover"][0]  # hyperparameter
        new_inf = mask & (self.agents.start["sick"] == today)
        new_inf = new_inf.count_nonzero()

        for day in range(1, recover + 1):
            cnt = mask & (self.agents.period["sick"] == day)
            cnt = cnt.count_nonzero()
            contagious_strength = self.gamma.log_prob(day).to(self.device).exp()
            sum_I += cnt * contagious_strength

        res = (new_inf / sum_I).item() if sum_I.gt(0.0) else 0

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
        sas = self.buildings.identity["area"].unique(sorted=True)
        res = {}

        for sa in sas:
            sa_a = self.agents.identity["area"] == sa
            res[sa.item()] = self._compute_R(sa_a, day)

        del sas
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
        ab_house = self.network.AB["house"].detach().clone()
        a_inf = self.agents.get_infected()
        inf_house = ab_house.mul(a_inf.view(-1, 1))

        total = ab_house.sum(0)
        inf = inf_house.sum(0)
        ratio = torch.zeros_like(total, dtype=torch.float)

        if total.count_nonzero() != 0:
            idx = total.nonzero()
            ratio[idx] = inf[idx].div(total[idx])

        del ab_house, a_inf, inf_house
        return inf, ratio

    def _get_IO_mat(self, inf_mat):
        """
        Obtain in-degree SA infection matrix
        Based on agents.daily_infection

        Idea:
        1. Loop through all SAs.
        2. For each SA, get a mask of corresponding agents.
        3. Mask inf_mat. (A, A)
        4. Sum inf_mat row-wise to obtain infecting agents of this SA. (1, A)
        5. Perform mat-mat multiplication with ASA network (1, A) @ (A, SA) = (1, SA)

        Parameter
        ----------
        inf_mat : torch.Tensor (A, A)

        Return
        -------
        res : torch.Tensor (SA, SA)
        """
        ASA = self.network.ASA.detach().clone()  # (A, SA)
        num_SA = ASA.shape[1]

        res = torch.zeros((num_SA, num_SA))

        for idx in range(num_SA):
            mask = ASA[:, idx]  # step 2
            infecting_a = mask.view(-1, 1) & inf_mat  # step 3
            infecting_a = infecting_a.sum(0)  # step 4 (1, A)
            res[idx] = (infecting_a.float() @ ASA.float()).bool()  # step 5 (1, SA)

        del ASA, num_SA

        return res  # (SA, SA)

    def _get_vis_R(self, day):
        """
        Return vis_R according to lockdown scenario and day

        Parameter
        ---------
        day : int

        Return
        -------
        vis_R: int if not diff else dict
        prev_vis_R: int if not diff else dict
        """
        diagnose = self.disease["diagnose"]
        if not self.scenario["DIFF"]:
            vis_R = self.res["Results"]["Stats"][day - diagnose]["R_total"]
            prev_vis_R = self.res["Results"]["Stats"][day - (diagnose + 1)]["R_total"]
        else:
            vis_R = self.res["Results"]["SAs"][day - diagnose]
            prev_vis_R = self.res["Results"]["SAs"][day - (diagnose + 1)]

        return vis_R, prev_vis_R

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
            print()
            print()
            print("Day:", day)

            """
            Computation to obtain data
            """
            t1 = time()
            routine = self.agents.update_routine(
                self.buildings.status, self.network.AB["house"].detach().clone()
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

            new_inf, inf_mat = self.agents.update_infection(
                day, routine.detach().clone(), self.gamma
            )
            t6 = time()
            self._log_time("Update Infection", t6 - t5)

            self.agents.end_quarantine()
            t7 = time()
            self._log_time("End Quarantine", t7 - t6)

            new_qua = self.agents.update_diagnosis(day)
            t8 = time()
            self._log_time("Update Diagnosis", t8 - t7)

            new_qua += self.agents.update_diagnosed_family(
                day, self.network.AH.detach().clone()
            )
            t9 = time()
            self._log_time("Update Diagnosed Family", t9 - t8)

            new_recovered = self.agents.update_recovery()
            t10 = time()
            self._log_time("Update Recovery", t10 - t9)

            R_total = self._compute_R(torch.ones_like(self.agents.status).bool(), day)
            t11 = time()
            self._log_time("Compute overall R", t11 - t10)

            R_sa = self._get_SA_R(day)
            t12 = time()
            self._log_time("Compute SA R", t12 - t11)

            b_inf, b_ratio = self._get_building_inf()
            t13 = time()
            self._log_time("Compute building infection ratio", t13 - t12)

            io_mat = self._get_IO_mat(inf_mat)
            t14 = time()
            self._log_time("Compute IO matrix", t14 - t13)

            if day > self.disease["diagnose"] + 2:
                vis_R, prev_vis_R = self._get_vis_R(day)
                self.buildings.update_lockdown(vis_R, prev_vis_R)
                t15 = time()
                self._log_time("Update lockdown", t15 - t14)
                del vis_R, prev_vis_R

            num_inf = self.agents.get_infected().count_nonzero()
            num_qua = self.agents.get_quarantined().count_nonzero()
            num_death = self.agents.get_dead().count_nonzero()
            num_admission = self.agents.get_hospitalized().count_nonzero()
            num_recovered = self.agents.get_recovered().count_nonzero()
            total_inf = (
                num_inf
                if day == 1
                else (
                    self.res["Results"]["Stats"][day - 1]["Total Infections"] + new_inf
                )
            )
            num_susceptible = self.agents.identity["id"].shape[0] - total_inf
            num_closed = self.buildings.get_closed().count_nonzero()
            """
            Logging data to output
            """
            self.res["Results"]["SAs"][day] = R_sa
            self.res["Results"]["Stats"][day] = {}
            self._log_stats(day, "Total Infections", total_inf.item())
            self._log_stats(day, "Active Infections", num_inf.item())
            self._log_stats(day, "Daily Infections", new_inf.item())
            self._log_stats(day, "Total Recovered", num_recovered.item())
            self._log_stats(day, "Daily Recovered", new_recovered.item())
            self._log_stats(day, "Total Quarantined", num_qua.item())
            self._log_stats(day, "Daily Quarantined", new_qua.item())
            self._log_stats(day, "Total Hospitalized", num_admission.item())
            self._log_stats(day, "Daily Hospitalized", new_admission.item())
            self._log_stats(day, "Total Deaths", num_death.item())
            self._log_stats(day, "Daily Deaths", new_death.item())
            self._log_stats(day, "R_total", R_total)
            self._log_stats(day, "Closed Buildings", num_closed.item())
            self._log_stats(day, "Susceptible Agents", num_susceptible.item())

            self.res["Buildings"][day] = {}
            self.res["Buildings"][day]["inf"] = b_inf.tolist()
            self.res["Buildings"][day]["ratio"] = b_ratio.tolist()
            self.res["IO_mat"][day] = io_mat.tolist()
            self.res["daily_infection"][day] = inf_mat.nonzero().tolist()
            if self.debug:
                print()
                print("DEBUG")
                print("Susceptible:", (self.agents.status == 1).count_nonzero())
                print("Infected:", (self.agents.status == 2).count_nonzero())
                print(
                    "Qurantine, Susceptible:", (self.agents.status == 3).count_nonzero()
                )
                print(
                    "Qurantine, Infected, Undiagnosed:",
                    (self.agents.status == 4).count_nonzero(),
                )
                print(
                    "Quarantine, Infected, Diagnosed:",
                    (self.agents.status == 5).count_nonzero(),
                )
                print("Hospitalized:", (self.agents.status == 6).count_nonzero())
                print("Recovered:", (self.agents.status == 7).count_nonzero())
                print("Dead", (self.agents.status == 8).count_nonzero())
                print()

            if day == 20:
                exit()
                break
            day += 1
            del (
                num_admission,
                num_closed,
                num_death,
                num_qua,
                num_recovered,
                num_susceptible,
                new_admission,
                new_death,
                new_inf,
                new_qua,
                new_recovered,
                routine,
                R_total,
                R_sa,
                b_inf,
                b_ratio,
                io_mat,
                total_inf,
            )
            gc.collect()
            torch.cuda.empty_cache()

        self.output()

        print()
        print("SIMULATION tick", self.count, "COMPLETE!")
        print("Total time:", time() - self.time)
        print()

    def output(self):
        print()
        print("Writing to files....")
        name = ""
        for key, val in self.theme.items():
            if val is True:
                name += key + "_"
        for key, val in self.scenario.items():
            if val is True:
                name += key + "_"
        path = (
            self.out_dir
            + "sim_"
            + str(self.count)
            + "_"
            + name
            + datetime.now().strftime("%d")
            + datetime.now().strftime("%m")
            + datetime.now().strftime("%Y")
            + "_"
            + datetime.now().strftime("%H")
            + datetime.now().strftime("%M")
        )
        os.mkdir(path)
        with open(
            path + "/" + "results" + ".json",
            "w",
        ) as outfile:
            json.dump(self.res["Results"], outfile)
        print("results.json")

        with open(
            path + "/" + "buildings" + ".json",
            "w",
        ) as outfile:
            json.dump(self.res["Buildings"], outfile)
        print("buildings.json")

        with open(
            path + "/" + "io_mat" + ".json",
            "w",
        ) as outfile:
            json.dump(self.res["IO_mat"], outfile)
        print("io_mat.json")

        with open(
            path + "/" + "daily_infection" + ".json",
            "w",
        ) as outfile:
            json.dump(self.res["daily_infection"], outfile)
        print("daily_infection.json")

        print("Done.")


def myprint(d):
    for k, v in d.items():
        if isinstance(v, dict):
            myprint(v)
        else:
            print("{0}{1}:{2}".format(k, type(k), type(v)))
