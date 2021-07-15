"""
This model simulate the dynamics of a City under Covid-19 pandemic.
DySTUrbD-epi: Utility

This program contains several utility functions to aid data I/O,
and data preprocessing.

TBC stands for To Be Created.
"""


import pandas as pd
import torch


# def load_agent(csvFile):
#     """
#     This function creates agents using the follow data:
#     1. agent id (0)
#     2. household id (11)
#     3. disability (2) - (no: 0, yes: 1)
#     4. employment status (3) - (no: 0, yes: 1)
#     5. age group (6) - (child: 1, adult: 2, elderly: 3)
#     6. local work (12) - (no: 0, yes: 1)
#     7. workforce participation (3) - (no: 0, yes: 1)
#     8. household building id (1)
#     9. religious (TBC) - (no: 0, yes: 1)
#     10. job (TBC)
#     11. expected income (TBC)
#     12. actual income (TBC)
#     13. job building id (TBC)
#     14. contagious status (1,2,3,3.5, 4, 5, 6, 7)
#     15. contagious risk by age
#     16. exposure risk
#     17. contagious start
#     18. sick start
#     19. non-anchor activities
#     20. quarantine start
#     21. qurantine period
#     22. infection chain
#     23. stat area
#     24. admission probability
#     25. admission start
#     26. admission period
#     27. mortality probability
#     """
#     agents = load_csv(csvFile, cols=[0, 11, 2, 3, 6, 12, 3, 1])  # (22493, 8)
#     tbc = torch.zeros((agents.shape[0], 19))  # feat. 9-27
#     agents = torch.cat((agents, tbc), axis=1)  # (22493, 13)
#     return agents


# def load_buildings(csvFile):
#     """
#     This function creates buildings using the follow data:
#     1. building id (6)
#     2. land use (2)
#     3. number of floors (3)
#     4. statistical area (12)
#     5. floorspace volume (4) @ASK
#     6. initial land use (2)
#     7. X coordinate (14)
#     8. Y coordinate (15)
#     9. USG group (17)
#     10. USG group (18)
#     11. Building status (TBC) - (open: 1, close: 0)
#     """
#     buildings = load_csv(
#         csvFile, cols=[6, 2, 3, 12, 4, 2, 14, 15, 17, 18]
#     )  # (1015, 10)
#     tbc = torch.ones((buildings.shape[0], 1))  # feat. 11
#     buildings = torch.cat((buildings, tbc), axis=1)  # (1015, 11)
#     return buildings


# def load_households(csvFile):
#     """
#     This function creates households using the follow data:
#     1. household id (11)
#     2. building id (1)
#     3. income (10)
#     4. car (13) - (no car: 0, owns car: 1 ,??: 2) @ASK
#     5. religious (TBC) - (no: 0, yes: 1)
#     """

#     def groupby_households(df):
#         """
#         A functor to preprocess household data.
#         This allow us to hard code the column index,
#         while keeping the flexibility to switch preprocessing methods.
#         """
#         return df.groupby(11).first().reset_index()  # @ASK

#     houseHolds = load_csv(
#         csvFile, cols=[11, 1, 10, 13], preprocess=groupby_households
#     )  # (8655, 4)
#     tbc = torch.randint(0, 2, (houseHolds.shape[0], 1))  # feat. 5
#     houseHolds = torch.cat((houseHolds, tbc), axis=1)  # (8655, 5)
#     return houseHolds


# def load_prob(infection, admission, mortality):
#     """
#     Loads infection, admission, and mortality probabilty
#     from csv files.
#     """
#     return load_csv(infection), load_csv(admission), load_csv(mortality)


def load_csv(csvFile, cols=None, preprocess=None):
    """
    This function loads data from given csv path.
    Params  : csvFile (str), cols (list)
    Success : Return data as a tensor (<dim1>, ...,<dim4>)
    Fail    : Raise exception and halt
    """
    try:
        csvData = pd.read_csv(csvFile, header=None, usecols=cols)
        if cols is not None:
            csvData = csvData[cols]  # Preserve positional order
        if preprocess is not None:
            csvData = preprocess(csvData)  # Preprocess data
    except Exception as e:
        print("Unable to load data ", csvFile, ":", e)
        raise
    try:
        return torch.Tensor(csvData.to_numpy())
    except Exception as e:
        print("Unable to proceed: ", e)
        raise
