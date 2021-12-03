"""
This module initializes buildings.

A subset of the attributes will be loaded directly from csv file.
While the remaining will be randomly initialized.
The attributes with specific values are annotated.
"""

import itertools  # to flatten nested list
import torch

import util


class Buildings(object):
    """
    Specify buildings.

    To improve readability, the torch.Tensors will be wrapped using python
    native dictionary structure.

    Attributes
    ----------
    identity            : identity of buildings.
    usg                 : USG codes.
    land                : landuse.
    floor               : floor properties
    status              : open or close
    """

    def __init__(self, path, device, theme, scenario):
        """
        Initialize buildings.

        school0     : kindergarten
        school1     : elementary
        school2     : high school

        schoolR0    : religious kindergarten
        schoolR1    : religious elementary
        schoolR2    : religious high
        schoolR3    : yeshiva @ASK what about normal university?

        etc         : regular building
        religious   : religious
        etcR        : regular building + religious

        Parameters
        ----------
        path : str
        device : str
        theme : dict
        scenario : dict
        """
        self.device = device
        data = util.load_csv(
            path, self.device, cols=[6, 2, 3, 12, 4, 2, 14, 15, 17, 18], nrows=1200
        )
        self.identity = {
            "idx": torch.arange(data.shape[0], device=self.device),
            "id": data[:, 0],
            "area": self._get_SA_idx(data[:, 3]),
            "X": data[:, 6],
            "Y": data[:, 7],
        }

        self.usg_dict = {
            "school0": [5305],
            "school1": [5310],
            "school2": [5338],
            "schoolR0": [5300],
            "schoolR1": [5312],
            "schoolR2": [5523, 5525],
            "schoolR3": [5340],
            "etc": [6512, 6520, 6530, 6600, 5740, 5760, 5600, 5700, 5202, 5253],
            "religious": [5501, 5521],
        }

        self.usg = {"broad": data[:, 8], "specific": data[:, 9]}
        self.land = {"initial": data[:, 5], "current": data[:, 1]}
        self.floor = {"number": data[:, 2], "volume": data[:, 4]}
        self.status = torch.ones((data.shape[0],), device=self.device, dtype=torch.bool)
        self.activity = self._create_activity()
        self.theme = theme
        self.theme_mask = self.get_thematic_mask()
        self.scenario = scenario

    def _get_SA_idx(self, data):
        """
        Convert SAs, into arange of index

        Parameter
        ---------
        data : torch.Tensor (SA,1)

        Return
        -------
        res : torch.Tensor (SA,1)
        """
        res = torch.zeros_like(data)
        SAs = data.unique(sorted=True)
        num_SAs = SAs.shape[0]
        for idx in range(num_SAs):
            mask = data == SAs[idx]
            res[mask] = idx

        return res

    def _create_activity(self):
        """
        Initialize a collection of buildings corresponding to activities.

        Parameter
        ---------
        b   : dict

        Return
        ---------
        res : dict
        """
        res = dict()
        for key, val in self.usg_dict.items():
            res[key] = torch.Tensor([False]).bool().to(self.device)
            for usg in val:
                res[key] = res[key] | (self.usg["specific"] == usg)
            res[key] = self.identity["idx"][res[key].view(-1)]

        res["etcR"] = torch.cat((res["etc"], res["religious"]))
        return res

    def get_closed(self):
        """
        Return mask of closed buildings.
        """
        return self.status == 0

    def groupby_USG_or_SA(self, key_list, group):
        """
        Return mask of buildings group by given keys.

        Parameter
        ---------
        key_list : list
        group : string (USG or SA)

        Return
        -------
        res : torch.Tensor (B, 1)
        """
        assert (group == "SA") or (group == "USG"), "Invalid group for get_USG_or_SA"

        res = torch.zeros_like(self.status)
        attr = self.identity["area"] if group == "SA" else self.usg["specific"]

        for key in key_list:
            res |= attr == key

        return res

    def get_SA_masks(self):
        """
        Return a list of mask for buildings w.r.t SA

        Return
        -------
        res : list
        """
        res = [
            self.groupby_USG_or_SA(sa, "SA")
            for sa in self.identity["area"].unique(sorted=True)
        ]

        return res

    def get_thematic_mask(self, gradual=False):
        """
        Return a building mask according pre-configured lockdown setting

        Parameter
        ---------
        theme : dict of boolean

        ALL
        EDU
        REL

        gradual : boolean

        Return
        ------
        res : torch.Tensor
        """
        # All cannot be on together either EDU or REL
        assert ~(
            self.theme["ALL"] and (self.theme["EDU"] or self.theme["REL"])
        ), "Theme conflict"

        res = (
            torch.zeros_like(self.status)
            if gradual
            else torch.randint_like(self.status, 0, 2)
        )
        # ALL

        if self.theme["EDU"]:
            edu_usg = [
                itertools.chain.from_iterable(
                    [self.usg_dict[x] for x in self.usg_dict.keys() if "school" in x]
                )
            ]
            edu_mask = self.groupby_USG_or_SA(edu_usg, "USG")
            res &= edu_mask
            if gradual:
                idx = edu_mask.nonzero()
                res[idx] = torch.randint(0, 2, (idx.shape[0], 1))

        if self.theme["REL"]:
            rel_usg = self.usg_dict["religious"]
            rel_mask = self.groupby_USG_or_SA(rel_usg, "USG")
            res &= rel_mask
            if gradual:
                idx = rel_mask.nonzero()
                res[idx] = torch.randint(0, 2, (idx.shape[0], 1))

        return res

    def update_lockdown(self, vis_R, prev_vis_R):
        """
        Using visible R values (diagnosed R) to determine building status

        Parameter
        ---------
        vis_R: list
        prev_vis_R: list
        """
        # Get thematic mask

        res = torch.ones_like(self.status)
        b_mask = self.get_SA_masks() if self.scenario["DIFF"] else [res]
        vis_R = list(vis_R.values()) if self.scenario["DIFF"] else [vis_R]
        prev_vis_R = (
            list(prev_vis_R.values()) if self.scenario["DIFF"] else [prev_vis_R]
        )

        assert len(b_mask) == len(
            vis_R
        ), "Number of Building masks is not equal to number of vis_R"
        for idx in range(len(b_mask)):
            if self.scenario["GRADUAL"]:
                if 1 < vis_R[idx] < 2:
                    if ~(1 < prev_vis_R[idx] < 2):
                        theme_mask = self.get_thematic_mask(gradual=True)
                        res[b_mask[idx] & theme_mask] = 0
                    else:
                        res = self.status
                elif vis_R[idx] >= 2:
                    res[b_mask[idx] & self.theme_mask] = 0
            else:
                if vis_R[idx] >= 1:
                    res &= b_mask[idx] & self.theme_mask

        self.status = res
