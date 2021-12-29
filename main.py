"""
This model simulate the dynamics of a City under Covid-19 pandemic.
DySTUrbD-epi: Main

This program acts as an interface for model fusion,
or the project master file for debugging.
"""

import torch
import yaml

from model import DySTUrbD_Epi


def epi_run(args):
    """
    This function runs epidemiological model.
    """
    ticks = args["ticks"]
    for cnt in range(ticks):
        with torch.no_grad():
            model = DySTUrbD_Epi(args, cnt)
            model()


if __name__ == "__main__":
    """
    Enable this program to run in standalone mode,
    while preserving the flexibility to be fused into other models.
    """
    with open("config.yaml") as f:
        args = yaml.load(f, Loader=yaml.FullLoader)
        print(args)
    epi_run(args)
