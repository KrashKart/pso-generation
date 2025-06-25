import pyswarms.backend as be
from pyswarms.backend.topology.base import Topology
from pyswarms.backend.handlers import VelocityHandler, OptionsHandler, BoundaryHandler
import pyswarms.backend.operators as ops
from pyswarms.utils.reporter import Reporter

import logging
import numpy as np
from collections import deque

class WGS_Star(Topology):
    def __init__(self):
        super(WGS_Star, self).__init__(static=True)

    def compute_gbest(self, swarm):
        """Update the global best using a star topology

        This method takes the current pbest_pos and pbest_cost, then returns
        the minimum cost and position from the matrix.

        Parameters
        ----------
        swarm : pyswarms.backend.swarm.Swarm
            a Swarm instance

        Returns
        -------
        numpy.ndarray
            Best position of shape :code:`(n_dimensions, )`
        float
            Best cost
        """
        try:
            if self.neighbor_idx is None:
                self.neighbor_idx = np.tile(
                    np.arange(swarm.n_particles), (swarm.n_particles, 1)
                )
            if np.min(swarm.pbest_cost) <= swarm.best_cost:
                # Get the particle position with the lowest pbest_cost
                # and assign it to be the best_pos
                best_pos = swarm.pbest_pos[np.argmin(swarm.pbest_cost)]
                best_cost = np.min(swarm.pbest_cost)
            else:
                # Just get the previous best_pos and best_cost
                best_pos, best_cost = swarm.best_pos, swarm.best_cost
        except AttributeError:
            print(
                "Please pass a Swarm class. You passed {}".format(type(swarm)))
            raise
        else:
            return (best_pos, best_cost)

    def compute_velocity(
        self,
        swarm,
        clamp=None,
        vh=VelocityHandler(strategy="unmodified"),
        bounds=None,
    ):
        """Compute the velocity matrix

        This method updates the velocity matrix using the best and current
        positions of the swarm. The velocity matrix is computed using the
        cognitive and social terms of the swarm.

        Parameters
        ----------
        swarm : pyswarms.backend.swarms.Swarm
            a Swarm instance
        clamp : tuple of floats (default is :code:`None`)
            a tuple of size 2 where the first entry is the minimum velocity
            and the second entry is the maximum velocity. It
            sets the limits for velocity clamping.
        vh : pyswarms.backend.handlers.VelocityHandler
            a VelocityHandler instance
        bounds : tuple of :code:`np.ndarray` or list (default is :code:`None`)
            a tuple of size 2 where the first entry is the minimum bound while
            the second entry is the maximum bound. Each array must be of shape
            :code:`(dimensions,)`.

        Returns
        -------
        numpy.ndarray
            Updated velocity matrix
        """
        return ops.compute_velocity(swarm, clamp, vh, bounds=bounds)

    def compute_position(
        self, swarm, bounds=None, bh=BoundaryHandler(strategy="periodic")
    ):
        """Update the position matrix

        This method updates the position matrix given the current position and
        the velocity. If bounded, it waives updating the position.

        Parameters
        ----------
        swarm : pyswarms.backend.swarms.Swarm
            a Swarm instance
        bounds : tuple of :code:`np.ndarray` or list (default is :code:`None`)
            a tuple of size 2 where the first entry is the minimum bound while
            the second entry is the maximum bound. Each array must be of shape
            :code:`(dimensions,)`.
        bh : pyswarms.backend.handlers.BoundaryHandler
            a BoundaryHandler instance

        Returns
        -------
        numpy.ndarray
            New position-matrix
        """
        return ops.compute_position(swarm, bounds, bh)

def wgs_optimise(f, iters, report_iters, scale, 
                 n_particles, dimensions, options, bounds,
                 bh_strategy="nearest", vh_strategy="unmodified", oh_strategy=None,
                 init_pos=None, ftol=None, ftol_iter=1,
                 verbose=True):
    
    """ GlobalBestPSO tailored for the WGS problem

    Args:
        f: objective function
        iters (int): iterations to carry out PSO
        report_iters (int): no. of iters to 
        scale (_type_): _description_
        n_particles (_type_): _description_
        dimensions (_type_): _description_
        options (_type_): _description_
        bounds (_type_): _description_
        bh_strategy (str, optional): _description_. Defaults to "nearest".
        vh_strategy (str, optional): _description_. Defaults to "unmodified".
        oh_strategy (_type_, optional): _description_. Defaults to None.
        init_pos (_type_, optional): _description_. Defaults to None.
        ftol (_type_, optional): _description_. Defaults to None.
        ftol_iter (int, optional): _description_. Defaults to 1.

    Returns:
        final_best_cost: final best cost found
        final_best_pos: final best position found 
        gbests: list of global bests found over optimisation loop for recording
        cost_his, pos_his: history of best costs and positions of all particles
    """

    swarm = be.create_swarm(n_particles=n_particles, dimensions=dimensions, options=options, bounds=bounds, init_pos=init_pos)
    gbests = [] # for logging
    cost_his = [] # for logging
    pos_his = [] # for logging

    top = WGS_Star() # topology
    bh = BoundaryHandler(strategy=bh_strategy) 
    vh = VelocityHandler(strategy=vh_strategy)
    oh = OptionsHandler(strategy=oh_strategy)
    rep = Reporter(logger=logging.getLogger(__name__)) # for reporting to cmd line

    # Apply verbosity
    if verbose:
        log_level = logging.INFO
    else:
        log_level = logging.NOTSET

    rep.log("Optimize for {} iters with {}".format(iters, options), lvl=log_level)
    # Populate memory of the handlers
    bh.memory = swarm.position
    vh.memory = swarm.position
    swarm.pbest_cost = np.full(n_particles, np.inf)
    ftol_history = deque(maxlen=ftol_iter)

    for i in rep.pbar(iters, "WGS Optimisation") if verbose else range(iters):
        # Part 1: Update personal best
        swarm.current_cost = be.compute_objective_function(swarm, f) # Compute current cost
        swarm.pbest_pos, swarm.pbest_cost = be.compute_pbest(swarm) # Update and store

        swarm.pbest_pos = check(swarm.pbest_pos, scale)
        best_cost_yet_found = swarm.best_cost

        # Part 2: Update global best
        swarm.best_pos, swarm.best_cost = top.compute_gbest(swarm)
        if verbose:
            rep.hook(best_cost=swarm.best_cost)

        # Part 3: Verify stop criteria based on the relative acceptable cost ftol
        if ftol:
            relative_measure = ftol * (1 + np.abs(best_cost_yet_found))
            delta = (
                np.abs(swarm.best_cost - best_cost_yet_found)
                < relative_measure
            )
            if i < ftol_iter:
                ftol_history.append(delta)
            else:
                ftol_history.append(delta)
                if all(ftol_history):
                    break

        # Part 4: Update position and velocity matrices
        # Note that position and velocity updates are dependent on your topology
        swarm.velocity = top.compute_velocity(swarm, vh=vh, bounds=bounds)
        swarm.position = top.compute_position(swarm, bounds=bounds, bh=bh)
        swarm.options = oh(options, iternow=i, itermax=iters)

        # logging
        if i % report_iters == 0:
            gbests.append(np.copy(swarm.best_pos))
        cost_his.append(swarm.best_cost.copy())
        pos_his.append(swarm.position.copy())

    final_best_cost = swarm.best_cost.copy()
    final_best_pos = swarm.pbest_pos[swarm.pbest_cost.argmin()].copy()

    rep.log("Optimization finished | best cost: {}".format(
                final_best_cost, final_best_pos
            ),
            lvl=log_level,
        )
    
    return final_best_cost, final_best_pos, gbests, cost_his, pos_his
    

def constraint(data, colmaxes):
    # ensure values are confined
    unnorm = data * colmaxes
    if len(data.shape) == 1:
        tot = np.sum(unnorm)
        if tot > 100:
            unnorm *= (100 / tot)

    else:
        tot = np.sum(unnorm, axis=1)
        violates = np.argwhere(tot > 100)
        for row in violates:
            fault = unnorm[row]
            fault *= (100 / tot[row])
            unnorm[row] = fault
    new = unnorm / colmaxes
    return new
    
def check(positions, scale):
    position = np.copy(positions)
    if len(position.shape) == 1:
        position[-8:-2] = constraint(position[-8:-2], scale.colmax[-8:-2])
        help_me = np.concatenate([position[:7], position[9:-9]])
        help_me_maxes = np.concatenate([scale.colmax[:7], scale.colmax[21:-9]])
        new_help_me = constraint(help_me, help_me_maxes)
        position[:7] = new_help_me[:7]
        position[9:-9] = new_help_me[7:]
    else:
        position[:, -8:-2] = constraint(position[:, -8:-2], scale.colmax[-8:-2])
        help_me = np.concatenate([position[:, :7], position[:, 9:-9]], axis=1)
        help_me_maxes = np.concatenate([scale.colmax[:7], scale.colmax[21:-9]])
        new_help_me = constraint(help_me, help_me_maxes)
        position[:, :7] = new_help_me[:, :7]
        position[:, 9:-9] = new_help_me[:, 7:]
    return position