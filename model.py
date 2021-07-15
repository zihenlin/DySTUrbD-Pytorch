"""
This model simulate the dynamics of a City under Covid-19 pandemic.
DySTUrbD-epi: Model

This program is the implementation of DySTUrbD-Epi class.
"""

# import torch

import buildings
import agents


class DySTUrbD_Epi(object):
    def __init__(self, args):
        """
        Initialization of the class.
        """
        self.buildings = buildings.Buildings(args.buildings_dir)
        self.agents = agents.Agents(args, self.buildings)
