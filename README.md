This is the Pytorch implementation of [DySTUrbB](https://github.com/amirca91/DySTUrbD/tree/COVID_ABM).
The class attributes are listed for debugging convenience.

### Class Agents

    Attributes
    ----------
    identity            : Identities of the agents.
        1. id
        2. house
        3. disable (Binary)
        4. group (Child (1), Adult(2), Elderly(3))
        5. age
        6. religious (Binary)

    building            : Building IDs of the agents.
        1. house
        2. anchor
        3. trivial

    job         : Employment related information.
        1. status (Binary)
        2. local (Binary)
        3. job
        4. income

    risk                : Risks of exposure, contagion, admission, mortality.
        1. exposure
        2. contagion
        3. admission
        4. mortality

    timestamp           : Relative start dates and period
        1. c_start      - contagion start
        2. s_start      - sick start
        3. q_start      - quarantine start
        4. a_start      - admission start
        5. q_period     - quarantine period
        6. a_period     - admission period

    status              : Contagious status
        1. Susceptible
        2. Infected, Undiagnosed
        3. Quarantined, Uninfected
        4. Quarantined, Infected, Undiagnosed
        5. Quarantined, Infected, Diagnosed
        6. Hospitalized
        7. Recovered
        8. Dead

### class Buildings

    Attributes
    ----------
    identity        : identities
        1. id
        2. area
        3. X
        4. Y

    land            : land use
        1. Initial
        2. Current

    floor           : floor properties
        1. number
        2. volume

    usg             : USG codes
        1. Broad
        2. Specific

    status          : open or close
        0. open
        1. close
