"""
This module initializes buildings.

A subset of the attributes will be loaded directly from csv file.
While the remaining will be randomly initialized.
The attributes with specific values are annotated.
"""

import torch

import util


class Buildings(object):
    """
    Specify buildings.

    To improve readability, the torch.Tensors will be wrapped using python
    native dictionary structure.

    Attributes
    ----------
    identity            : Identities of the agents.
    usg                 : USG codes.
    land                : landuse.
    floor               : floor properties
    status              : open or close
    """

    def __init__(self, path):
        """
        Initialize buildings.

        Parameters
        ----------
        args : arguments
        """
        data = util.load_csv(path, cols=[6, 2, 3, 12, 4, 2, 14, 15, 17, 18])
        self.identity = {
            "id": data[:, 0],
            "area": data[:, 3],
            "X": data[:, 6],
            "Y": data[:, 7],
        }
        self.usg = {"broad": data[:, 8], "specific": data[:, 9]}
        self.land = {"initial": data[:, 5], "current": data[:, 1]}
        self.floor = {"number": data[:, 2], "volume": data[:, 4]}
        self.status = torch.ones((data.shape[0],))
        self.activity = self._create_activity()

    def _create_activity(self):
        """
        Initialize a collection of buildings corresponding to activities.

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

        Parameter
        ---------
        b   : dict

        Return
        ---------
        res : dict
        """
        res = dict()
        usg_dict = {
            "school0": [5305],
            "school1": [5310],
            "school2": [5338],
            "schoolR0": [5300],
            "schoolR1": [5312],
            "schoolR2": [5523, 5525],
            "schoolR3": [5340],
            "etc": [6512, 6520, 6530, 6600, 5740,
                    5760, 5600, 5700, 5202, 5253],
            "religious": [5501, 5521]}
        for key, val in usg_dict.items():
            res[key] = torch.Tensor([False]).bool()
            for usg in val:
                res[key] = res[key] | (self.usg["specific"] == usg)
            res[key] = torch.nonzero(res[key]).long().view(-1)
            res[key] = self.identity["id"][res[key]]
        res["etcR"] = torch.cat((res["etc"], res["religious"]))
        return res
