#!/usr/bin/env python2.6
# vim:et sts=4 sw=4

from __future__ import print_function
from __future__ import division

import collections
import itertools
import functools
import operator
import optparse
import os
import numpy as np
import random
import sys
import textwrap
import time

from math import e, pi, cos, sin, exp, sqrt


class Error(Exception): pass
class UnknownAction(Error): pass
class InvalidStateName(Error): pass
class IncompleteParticleDefinition(Error): pass
class IncompleteDimensionSpec(Error): pass
class IncompatibleDimensionSpec(Error): pass

class Function(object):
    _default_per_dim_constraints = (-100, 100)
    allowed_dimensions = None  # allow all
    full_constraints = None  # use per_dim_constraints
    per_dim_constraints = None

    def __init__(self, dim, offset=None):
        self.dim = dim
        if offset is None:
            offset = np.zeros(self.dim)
        else:
            offset = np.array(offset)
            assert offset.size == self.dim
        self.offset = offset
        self.constraints = self.dim_bounds(self.dim)
        self.diagonal = sqrt(sum((a-b)**2 for a, b in self.constraints))
        name = self.__class__.__name__
        if name.startswith('Function'):
            name = name[len('Function'):]
        self.name = name

    @classmethod
    def dim_bounds(cls, dim):
        if cls.full_constraints is not None:
            assert dim is None or dim == len(cls.full_constraints)
            return tuple(cls.full_constraints)
        elif cls.per_dim_constraints is not None:
            return (cls.per_dim_constraints,) * dim
        else:
            return (cls._default_per_dim_constraints,) * dim

    def __call__(self, position):
        raise NotImplementedError()


class Rastrigin(Function):
    per_dim_constraints = (-5.12, 5.12)

    def __call__(self, position):
        func_pos = position - self.offset
        return np.sum(func_pos**2 - 10*np.cos(2*pi*func_pos) + 10)

class Ackley(Function):
    per_dim_constraints = (-32.768, 32.768)

    def __call__(self, position):
        func_pos = position - self.offset
        s1 = np.sum(func_pos**2)
        s2 = np.sum(np.cos(2*pi*func_pos))
        return 20 + e + -20 * exp(-0.2 * sqrt(s1/self.dim)) - exp(s2/self.dim)

class DeJongF4(Function):
    per_dim_constraints = (-20, 20)

    def __init__(self, *args, **kargs):
        super(DeJongF4, self).__init__(*args, **kargs)
        self.idx_vec = np.array(range(1, self.dim+1))

    def __call__(self, position):
        return np.sum(self.idx_vec * (position - self.offset)**4)

class Easom(Function):
    per_dim_constraints = (-100, 100)

    def __call__(self, position):
        func_pos = position - self.offset
        return 1 + exp(-np.sum(func_pos**2)) * np.prod(np.cos(pi+func_pos))

class Gauss(Function):
    per_dim_constraints = (-2, 2)

    def __call__(self, position):
        func_pos = position - self.offset
        return 1 - np.prod(np.exp(-func_pos**2))

class Griewank(Function):
    per_dim_constraints = (-600, 600)

    def __init__(self, *args, **kargs):
        super(Griewank, self).__init__(*args, **kargs)
        self.sqrt_idx_vec = np.sqrt(np.array(range(1, self.dim+1)))

    def __call__(self, position):
        func_pos = position - self.offset
        s = np.sum(func_pos**2)
        p = np.prod(np.cos(func_pos / self.sqrt_idx_vec))
        return s/4000 - p + 1

class Rosenbrock(Function):
    per_dim_constraints = (-100, 100)

    def __call__(self, position):
        func_pos = position - self.offset
        v = func_pos[:-1]
        v1 = func_pos[1:]
        return np.sum(100 * (v1 - v**2)**2 + (v-1)**2)


class SchafferF6(Function):
    per_dim_constraints = (-100, 100)

    def __call__(self, position):
        func_pos = position - self.offset
        xsq = np.sum(func_pos**2)
        return .5 + (sin(sqrt(xsq))**2 - .5)/((1 + .001*xsq)**2)

class SchafferF7(Function):
    per_dim_constraints = (-100, 100)

    def __call__(self, position):
        func_pos = position - self.offset
        xsq = np.sum(func_pos**2)
        return xsq**.25 * (sin(50*xsq**.1)**2 + 1)

class Schwefel(Function):
    per_dim_constraints = (-500, 500)

    def __call__(self, position):
        func_pos = position - self.offset
        return (418.9829 * self.dim +
                np.sum(func_pos * np.sin(np.sqrt(np.abs(func_pos)))))

class Parabola(Function):
    per_dim_constraints = (-50, 50)

    def __call__(self, position):
        """Multi-dimensional parabola.

        Arguments:
            position -- iterable over vector values

        Returns:
            x**2 + y**2 + ...

        >>> str(Parabola(2)(np.array((2, 2))))
        '8.0'
        >>> str(Parabola(3)(np.array((1, 2, 3))))
        '14.0'
        >>> str(Parabola(3)(np.array((-1, 4, 7.2))))
        '68.84'
        """
        return np.sum((position - self.offset)**2)

Sphere = Parabola


class PVCube(object):
    """Position/Velocity Cube.

    Initialize the position within the cube according to the sampler, and
    initialize velocity according to a random vector that could take a particle
    on a boundary all the way to the other boundary in one step.

    So, for each dimension bounded by (min, max), it produces
    location = random.uniform(min, max)
    velocity = random.uniform((min - max), (max - min))
    """

    def __init__(self, pos_bounds, vel_bounds=None, random=None):
        """Create a sample within the specified cube.

        Arguments:
            dim_bounds -- list of boundary pairs (numeric)

        Keyword Arguments:
            random -- np.random.RandomState object

        >>> PVCube(((1, 3), (4, -1)), lambda a,b:(a+b)/2)()
        (array([ 2. ,  1.5]), array([ 0.,  0.]))
        >>> PVCube(((1, 3), (4, -1)), lambda a,b:a)()
        (array([1, 4]), array([-1. , -2.5]))
        >>> PVCube(((1, 3), (4, -1)), lambda a,b:b)()
        (array([ 3, -1]), array([ 1. ,  2.5]))
        >>> PVCube(((1, 3), (4, -1)), lambda a,b:b, None, lambda a,b:a-b)()
        (array([ 3, -1]), array([-2., -5.]))
        """
        self.pos_bounds = pos_bounds
        if vel_bounds is None:
            vel_bounds = tuple((-abs(a-b)/2, abs(a-b)/2) for a, b in pos_bounds)
        self.vel_bounds = vel_bounds
        if not random:
            random = np.random.RandomState()
        self.random = random

    def new_position(self):
        return np.array([self.random.uniform(a, b) for a, b in self.pos_bounds])

    def new_velocity(self, position):
        return np.array([self.random.uniform(a, b) for a, b in self.vel_bounds])

    def __call__(self):
        p = self.new_position()
        v = self.new_velocity(p)
        return p, v


class Particle(object):
    class Scratch(object): pass
    def __init__(self, id, position, velocity,
                 value=None, best_position=None, best_value=None,
                 collisions=None, age_of_best=None):
        self.id = id
        self.position = np.array(position)
        self.velocity = np.array(velocity)

        self.collisions = (0 if collisions is None else collisions)

        self._value = (None if value is None else value)

        # A place to store random temporary things
        self.reset_scratch()

        self.update_best(age_of_best=age_of_best,
                         best_position=best_position,
                         best_value=best_value)

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, v):
        self._value = v
        self.age_of_best += 1

    def reset_scratch(self):
        self.scratch = self.Scratch()

    def update_best(self,
                    age_of_best=None, best_position=None, best_value=None):
        self.age_of_best = (0 if age_of_best is None else age_of_best)
        self.best_position = (np.array(self.position)
                              if best_position is None else best_position)
        self.best_value = (self.value if best_value is None else best_value)

    def distance_from(self, other):
        return sqrt(np.sum((self.position-other.position)**2))

    def __repr__(self):
        return ("{classname}({id!r}, {position!r}, {velocity!r}, "
                "value={val!r}, best_position={best_pos!r}, "
                "best_value={best_val!r}, collisions={collisions!r}, "
                "age_of_best={best_age!r})").format(
                    classname=self.__class__.__name__,
                    id=self.id,
                    position=self.position,
                    velocity=self.velocity,
                    val=self.value,
                    best_pos=self.best_position,
                    best_val=self.best_value,
                    collisions=self.collisions,
                    best_age=self.age_of_best)

    def __str__(self):
        return self.output(show_vectors=True)

    def output(self, show_vectors=False):
        to_show = ["id",
                   "value",
                   "best_value",
                   "position",
                   "best_position",
                   "velocity",
                   "age_of_best",
                   "collisions",]
        if not show_vectors:
            to_show.remove("position")
            to_show.remove("best_position")
            to_show.remove("velocity")

        attr_strs = ((name, str(getattr(self, name)).replace("\n", " "))
                     for name in to_show)
        return ", ".join("{0}={1}".format(name, val) for name, val in attr_strs)


# Simulation Stages
INITIALIZE = "Initialize"
UPDATE = "Update"
EVALUATE = "Evaluate"


def all_pairs(items):
    seen = []
    for item in items:
        for item2 in seen:
            yield item2, item
        seen.append(item)


class Topology(object):
    def __init__(self, initial_particles):
        self.particles = dict((p.id, p) for p in initial_particles)

    def neighbors(self, particle):
        """Iterator over neighbors for this particle."""
        raise NotImplementedError("subclass this class")


class Star(Topology):
    def neighbors(self, particle):
        return (p for p in self.particles if p.id != particle.id)


class Ring(Topology):
    def __init__(self, *args, **kargs):
        super(Ring, self).__init__(*args, **kargs)
        self.ordered_particles = sorted(self.particles.values(),
                                        key=operator.attrgetter("id"))
        self.id_to_index = dict((p.id, i)
                                for i, p in enumerate(self.ordered_particles))
        self.size = len(self.ordered_particles)

    def neighbors(self, particle):
        pidx = self.id_to_index[particle.id]
        return [self.ordered_particles[(pidx-1) % self.size],
                self.ordered_particles[(pidx+1) % self.size]]


class Simulation(object):
    variables = (
        ('max_evals', 100000,
         "Maximum function evaluations.  0 = don't ever stop."),
        ('momentum', '1.0',
         "Momentum (dynamic format).  Can be a plain number, or one "
         "of the following:\n"
         "* linear:<start momenum>:<end momentum>[:max evals]"
         " - takes even steps for every evaluation\n"
         "* random:<bound1>:<bound2> - uniform random between bounds\n"
         "* gamma:<start>:<floor>:<gamma> - uses gamma**t as a multiplier\n"
         "* updatedecay:<start>:<floor>:<gamma> - decays momentum based on"
         " updates of a normalized product series where"
         " weight_t = 1 + gamma(weight_t-1) if updated.  If not,"
         " 1 is not added.  The weight is then normalized by the"
         " same rule without conditioning on upates (always addding 1)\n"
         "* negexp:<start momentum>:<end momentum>:<pos time exponent>"
         " - uses t**-exp as a multiplier\n"
         "* pull:<ceiling>:<floor>:<weight> - starts in the middle, then"
         " when an update occurs, pulls up on it by (1-weight) * difference.\n"
         "If max_evals is not specified for linear, then it is "
         "assumed to be the same as the global max_evals setting."),
        ('vmax_multiplier', 1.0,
         "VMax multiplier, applied to half the distance between bounds "
         "in each dimension, but only if constrict=False."),
        ('constrict', True,
         "If true, applies Clerc's constriction formula to velocity updates."),
        ('kappa', 1.0,
         "Clerc's kappa constant, usually 1.0 - applies if constrict=True"),
        ('social_constant', '2.05',
         "Constant applied to the neighborhood best vector.  "
         "See 'momentum' for a description of the ways this can be changed."),
        ('cognitive_constant', '',
         "Constant applied to the personal best vector.  "
         "If not set or empty, it obtains its value from 'social_constant'. "
         "See 'momentum' for a description of the ways this can be changed."),
        ('radius_multiplier', 0.0,
         "If non-zero, particles bounce.  "
         "This is multiplied by the constraint diagonal to obtain "
         "the true radius."),
        ('radius_gamma', 1.0,
         "Multiplier to apply to radius with each bounce."),
        ('adapt_bounce_distance', False,
         "If True, adapts bounce distance basd on number of collisions by "
         "a factor of radius_gamma**-collisions."),
        ('decrement_bounce_on_update', 0,
         "If True, decrements the bounce count by this amount every time a "
         "new best is found."),
        ('use_vmax', True,
         "If True, caps particles with vmax.  "
         "Ignored if 'constricted' is set."),
    )

    def __init__(self,
                 num_initial_particles,
                 init_strategy,
                 topo_strategy,
                 opt_function,
                 random=None,
                 **kargs):
        """Construct a Simulation object.

        Arguments:
            num_initial_particles -- size of swarm when starting up
            init_strategy -- a class that accepts function constraints
                and position and velocity samplers to produce particles.
            topo_strategy -- a factory that accepts a list of particles and
                sets up a topology.
            opt_function -- the cost function to minimize

        Keyword Arguments:
            random -- np.random.RandomState object

        Other arguments: see "variables" in class definition, which provide
        suitable defaults.
        """
        # Create appropriate variables from the keyword arguments.
        class Vars: pass
        self.var = Vars()
        for name, default, _ in self.variables:
            setattr(self.var, name, kargs.get(name, default))

        if not self.var.cognitive_constant:
            self.var.cognitive_constant = self.var.social_constant

        self.function = opt_function
        self.vmax = np.array([abs(a-b)/2 * self.var.vmax_multiplier
                              for a, b in self.function.constraints])
        self.particle_id_iter = self.particle_id_generator()
        self.num_initial_particles = num_initial_particles
        if not random:
            random = np.random.RandomState()
        self.random = random
        self.topo_strategy = topo_strategy
        self.initializer = init_strategy(
            pos_bounds=self.function.constraints,
            vel_bounds=tuple((-v, v) for v in self.vmax),
            random=self.random)
        self.m_value_func = self._value_func_from_flags(
            self.var.momentum)
        self.sc_value_func = self._value_func_from_flags(
            self.var.social_constant)
        self.cc_value_func = self._value_func_from_flags(
            self.var.cognitive_constant)
        self.radius = self.function.diagonal * self.var.radius_multiplier

    def __iter__(self):
        max_evals = self.var.max_evals
        self.particles = []
        batch = 0
        for p in self.initialize():
            self.particles.append(p)
            yield batch, INITIALIZE, p
        self.topology = self.topo_strategy(self.particles)
        evals = 0
        while True:
            for p in self.evaluate():
                if max_evals and evals >= max_evals:
                    return
                yield batch, EVALUATE, p
                evals += 1
            for p in self.update(evals):
                yield batch, UPDATE, p
            batch += 1

    def next_particle_id(self):
        return next(self.particle_id_iter)

    def initialize(self):
        # Just create the number of initial particles by default
        for _ in range(self.num_initial_particles):
            p = Particle(next(self.particle_id_iter), *self.initializer())
            yield p

    def evaluate(self):
        for p in self.particles:
            p.value = self.function(p.position)
            if p.best_value is None or p.value < p.best_value:
                p.update_best()
                p.collisions = max(
                        0, p.collisions-self.var.decrement_bounce_on_update)
            yield p

    @staticmethod
    def _dynamic_flag(flag):
        flag = flag.lower()
        v2 = None
        spec = None
        if ":" not in flag:
            name = "value"
            v1 = float(flag)
        else:
            pieces = flag.split(":")
            assert len(pieces) <= 4, "Too many parts in {0}".format(flag)
            name = pieces[0]
            v1 = float(pieces[1])
            if len(pieces) > 2 and pieces[2]:
                v2 = float(pieces[2])
            if len(pieces) > 3 and pieces[3]:
                spec = float(pieces[3])
        return name, v1, v2, spec

    def _make_value_function(self, name, v1, v2, spec, ident=''):
        if name == 'value':
            return lambda *unused_args: v1
        elif name in ('random', 'uniform'):
            return lambda *unused_args: self.random.uniform(
                    low=min(v1,v2), high=max(v1,v2))
        elif name == 'gamma':
            def g(iters, particle=None):
                age = particle.age_of_best if particle else iters
                return v2 + spec ** age * (v1-v2)
            return g
        elif name == 'linear':
            if not spec:
                spec = self.var.max_evals
            assert spec, "Linear only makes sense with a max eval setting"
            spec = int(spec)
            def l(iters, particle=None):
                return v1 + (v2-v1) * (min(iters, spec-1)/(spec-1))
            return l
        elif name == 'negexp':
            def n(iters, particle=None):
                age = particle.age_of_best if particle else iters
                return v2 + ((age+1) ** -spec) * (v1-v2)
            return n
        elif name == "updatedecay":
            def d(iters, particle):
                age = particle.age_of_best
                norm_name = "{0}_norm".format(ident)
                val_name = "{0}_val".format(ident)
                norm = getattr(particle, norm_name, 1.0)
                val = getattr(particle, val_name, 1.0)
                norm = 1 + spec * norm
                val = int(age == 0) + spec*val
                setattr(particle, norm_name, norm)
                setattr(particle, val_name, val)
                weight = val / norm
                # Now we have a value between 0 and 1.  We use this to scale
                # between start and end values.
                return weight * (v1 - v2) + v2
            return d
        elif name == "pull":
            def p(iters, particle):
                age = particle.age_of_best
                weight_name = "{0}_pull_weight".format(ident)
                weight = getattr(particle, weight_name, (v1+v2)/2)
                pull_toward = v1 if age == 0 else v2
                weight += (1-spec) * (pull_toward - weight)
                setattr(particle, weight_name, weight)
                return weight
            return p
        elif name == "pulltop":
            def p(iters, particle):
                age = particle.age_of_best
                weight_name = "{0}_pull_weight".format(ident)
                weight = getattr(particle, weight_name, v1)
                pull_toward = v1 if age == 0 else v2
                weight += (1-spec) * (pull_toward - weight)
                setattr(particle, weight_name, weight)
                return weight
            return p
        elif name == "urandupdate":
            half_width = abs(v1-v2)/2
            def u(iters, particle):
                age = particle.age_of_best
                # The smoothed value, calculated by drawing it toward the last
                # good observed value.
                smooth_name = "{0}_smoothed_val".format(ident)
                # The last observed value
                observed_name = "{0}_observed_val".format(ident)
                # The last observed value that resulted in improvement - we can
                # only set this after the fact.
                good_obs_name = "{0}_good_obs_val".format(ident)
                obs = getattr(particle, observed_name, (v1+v2)/2)
                good_obs = getattr(particle, good_obs_name, (v1+v2)/2)
                smooth = getattr(particle, smooth_name, (v1+v2)/2)
                # Update good observed value if we updated last time
                if age == 0:
                    good_obs = obs
                    setattr(particle, good_obs_name, good_obs)
                # Now draw the smoothed value toward good_obs
                smooth = (1-spec) * (good_obs - smooth)
                setattr(particle, smooth_name, smooth)
                # Finally, draw from a random distribution
                obs = self.random.uniform(low=smooth-half_width,
                                          high=smooth+half_width)
                setattr(particle, observed_name, obs)
                return obs
            return u
        elif name == "gauss":
            # TODO: Also, add the ability to contract the stdev of that
            # gaussian when things are going well, and to do so around the mean
            # of recent values.
            def g(iters, particle=None):
                return self.random.normal(v1, v2)
            return g
        elif name == "stablegaussupdate":
            # v1 = mean_0
            # v2 = stdev_0
            # spec = initial t >= 1, defaults to 1
            def gu(iters, particle):
                mean_name = "{0}_mean".format(ident)
                var_name = "{0}_var".format(ident)
                time_name = "{0}_time".format(ident)
                momentum_name = "{0}_momentum".format(ident)
                good_name = "{0}_good".format(ident)
                mean = getattr(particle, mean_name, v1)
                var = getattr(particle, var_name, v2**2)
                time = getattr(particle, time_name, spec)
                momentum = getattr(particle, momentum_name, None)
                age = particle.age_of_best
                if momentum is not None and age == 0:
                    # Update the mean, time, and variance
                    mean = (time * mean + momentum) / (time + 1)
                    var = time * var / (time + 1) + (momentum - mean)**2 / time
                    time += 1
                    # Return the old momentum
                else:
                    # No update?  Sample.
                    momentum = self.random.normal(mean, sqrt(var))
                setattr(particle, mean_name, mean)
                setattr(particle, var_name, var)
                setattr(particle, time_name, time)
                setattr(particle, momentum_name, momentum)
                return momentum
            return gu
        elif name == "gaussupdate":
            # v1 = mean_0
            # v2 = stdev_0
            # spec = initial t >= 1, defaults to 1
            def gu(iters, particle):
                # The idea here is that we start out with a prior mean and
                # variance, and we sample from that to get an initial momentum.
                # Then, if the momentum produces a good value, we add that as a
                # new data point and adjust our parameters accordingly using
                # standard iterative procedures.  This requires that we know
                # how many "samples" went into producing the initial mean and
                # variance, so we have spec = t_0, indicating how confident we
                # are about our distribution.  The default is 1, as in, we got
                # our values from one data point.
                mean_name = "{0}_mean".format(ident)
                var_name = "{0}_var".format(ident)
                time_name = "{0}_time".format(ident)
                momentum_name = "{0}_momentum".format(ident)
                good_name = "{0}_good".format(ident)
                mean = getattr(particle, mean_name, v1)
                var = getattr(particle, var_name, v2**2)
                time = getattr(particle, time_name, spec)
                momentum = getattr(particle, momentum_name, None)
                age = particle.age_of_best
                if momentum is not None and age == 0:
                    # Update the mean, time, and variance
                    mean = (time * mean + momentum) / (time + 1)
                    var = time * var / (time + 1) + (momentum - mean)**2 / time
                    time += 1
                momentum = self.random.normal(mean, sqrt(var))
                setattr(particle, mean_name, mean)
                setattr(particle, var_name, var)
                setattr(particle, time_name, time)
                setattr(particle, momentum_name, momentum)
                return momentum
            return gu
        elif name == "gmmupdate":
            # "Gamma using Method of Moments (specify mean and variance)
            def gmmu(iters, particle):
                mean_name = "{0}_mean".format(ident)
                var_name = "{0}_var".format(ident)
                time_name = "{0}_time".format(ident)
                momentum_name = "{0}_momentum".format(ident)
                good_name = "{0}_good".format(ident)
                mean = getattr(particle, mean_name, v1)
                var = getattr(particle, var_name, v2**2)
                time = getattr(particle, time_name, spec)
                momentum = getattr(particle, momentum_name, None)
                age = particle.age_of_best
                if momentum is not None and age == 0:
                    # Update the mean, time, and variance
                    mean = (time * mean + momentum) / (time + 1)
                    var = time * var / (time + 1) + (momentum - mean)**2 / time
                    time += 1
                # Gamma functions takes "k, theta" as parameters, in that
                # order.  The method of moments allows us to use mean and
                # variance to provide these, by the following formulas:
                # k = mean^2 / variance
                # theta = variance / mean
                momentum = self.random.gamma(mean**2 / var, var / mean)
                setattr(particle, mean_name, mean)
                setattr(particle, var_name, var)
                setattr(particle, time_name, time)
                setattr(particle, momentum_name, momentum)
                return momentum
            return gmmu

        # TODO: Alternately, if a value works, and is above the center, then
        # draw in the lower bound by a little.  If it's lower, then draw in the
        # top.  The disadvantage to this is that it can't expand...
        else:
            raise ValueError("Unknown flag type: {0}".format(name))

    def _value_func_from_flags(self, flag):
        return self._make_value_function(*self._dynamic_flag(flag), ident=flag)

    def update(self, evals_so_far=0):
        max_evals = self.var.max_evals
        constrict = self.var.constrict
        use_vmax = self.var.use_vmax
        kappa = self.var.kappa
        vmax = self.vmax
        radius = self.radius
        radius_gamma = self.var.radius_gamma
        adapt_bounce = self.var.adapt_bounce_distance

        m_value_func = self.m_value_func
        sc_value_func = self.sc_value_func
        cc_value_func = self.cc_value_func

        # First set up the scratch information.
        for i, p in enumerate(self.particles):
            iteration = evals_so_far + i
            p.reset_scratch()

            best_n = min(self.particles_in_neighborhood(p),
                         key=operator.attrgetter('best_value'))

            ap = cc_value_func(iteration, p)
            an = sc_value_func(iteration, best_n)

            if constrict:
                psi = ap + an
                constriction = (2 * kappa / abs(2 - psi - sqrt(psi*(psi - 4)))
                                if psi > 4
                                else kappa)
            else:
                constriction = 1.0

            p_randvec = self.random.uniform(size=p.position.size)
            n_randvec = self.random.uniform(size=p.position.size)

            personal_vec = p.best_position - p.position
            neighbor_vec = best_n.best_position - p.position

            m = m_value_func(iteration, p)

            p.scratch.velocity = constriction * (m*p.velocity +
                                                 ap*p_randvec*personal_vec +
                                                 an*n_randvec*neighbor_vec)
            if constrict or not use_vmax:
                p.scratch.capped_velocity = p.scratch.velocity
            else:
                mask = np.abs(p.scratch.velocity) > vmax
                p.scratch.capped_velocity = (~mask*p.scratch.velocity +
                                             mask*vmax*np.sign(
                                                     p.scratch.velocity))
            p.scratch.position = p.position + p.scratch.capped_velocity

        if radius:
            # Annotate particles that should be bouncing
            for p1, p2 in all_pairs(self.particles):
                p1_m = radius_gamma**p1.collisions
                p2_m = radius_gamma**p2.collisions
                if p1.distance_from(p2) < radius * (p1_m + p2_m):
                    p1.scratch.do_bounce = True
                    p2.scratch.do_bounce = True

        for p in self.particles:
            if getattr(p.scratch, 'do_bounce', False):
                p.scratch.capped_velocity = -p.scratch.capped_velocity
                # Reflect the position about the previous position
                dist_factor = (1.0 if not adapt_bounce
                               else radius_gamma**-p.collisions)
                p.scratch.position = ((1 + dist_factor) * p.position -
                                      dist_factor * p.scratch.position)
                p.collisions += 1

            p.velocity[:] = p.scratch.capped_velocity
            p.position[:] = p.scratch.position
            yield p

    def particles_in_neighborhood(self, particle, exclude_self=True):
        # Star sociometry is the default (fully-connected graph)
        return (p for p in self.particles
                if not exclude_self or p.id != particle.id)

    def particle_id_generator(self):
        # Simple counter is the default
        return itertools.count()


class Logger(object):
    def __init__(self, filename):
        self._filename = filename
        self._file = (open(filename, "w") if filename else None)

    def __call__(self, *values, **kargs):
        skip_file = kargs.get('skip_file', False)
        to_stderr = kargs.get('to_stderr', True)
        if not skip_file and self._file is not None:
            print(*values, file=self._file)
        if to_stderr:
            print(*values, file=sys.stderr)


FUNC_CLASSES = dict((k, v) for k, v in sorted(globals().items())
                    if isinstance(v, type) and issubclass(v, Function)
                    and k != 'Function')


TOPOLOGY_CLASSES = dict((k, v) for k, v in sorted(globals().items())
                        if isinstance(v, type) and issubclass(v, Topology)
                        and k != 'Topology')


class HelpFormatter(optparse.IndentedHelpFormatter):
    def format_option(self, option):
        # Lifted from the IndentedHelpFormatter with one minor change:
        # textwrap.wrap is called separately per paragraph, as indicated by
        # newlines.
        result = []
        opts = self.option_strings[option]
        opt_width = self.help_position - self.current_indent - 2
        if len(opts) > opt_width:
            opts = "%*s%s\n" % (self.current_indent, "", opts)
            indent_first = self.help_position
        else:                       # start help on same line as opts
            opts = "%*s%-*s  " % (self.current_indent, "", opt_width, opts)
            indent_first = 0
        result.append(opts)
        if option.help:
            help_text = self.expand_default(option)
            help_lines = []
            for par in help_text.split("\n"):
                help_lines.extend(textwrap.wrap(par, self.help_width))
            result.append("%*s%s\n" % (indent_first, "", help_lines[0]))
            result.extend(["%*s%s\n" % (self.help_position, "", line)
                           for line in help_lines[1:]])
        elif opts[-1] != "\n":
            result.append("\n")
        return "".join(result)


def setup():
    """Set up command line args and general startup stuff.

    Returns:
        opts, args
    """
    parser = optparse.OptionParser(formatter=HelpFormatter())
    parser.add_option("", "--test", dest="run_tests",
                      default=False, action="store_true",
                      help="Test this module [%default].")
    parser.add_option("", "--profile", dest="profile",
                      default=False, action="store_true",
                      help="Profile this run [%default].")
    parser.add_option("", "--function", dest="function",
                      default="Parabola",
                      help=("Function to evaluate.  "
                            "Possible values are ({0})").format(
                                ", ".join(FUNC_CLASSES)))
    parser.add_option("", "--dim", dest="dim", default=None, type="int",
                      help=("Number of function dimensions [inferred from "
                            "offset if not specified here]."))
    parser.add_option("", "--per_dim_offset", type="float", default=None,
                      dest="per_dim_offset",
                      help=("Per-dimension offset to apply to function evals. "
                            "Helps to avoid center-seeking behavior.  Only "
                            "used if offset is not specified.  The "
                            "offset_multipliers parameter takes precedence "
                            "if specified."))
    parser.add_option("", "--per_dim_offset_multiplier", type="float",
                      default=None,
                      dest="per_dim_offset_multiplier",
                      help=("Per-dimension offset multiplier to apply to "
                            "function constraint boundaries (e.g., .5 means "
                            "offset half the distance from center of the "
                            "function's constraints to the maximum constraint "
                            "in each dimension, -.5 goes the other direction). "
                            "The per_dim_offset takes precedence over this."))
    parser.add_option("", "--offset", type="str", default=None,
                      dest="offset",
                      help=("Comma-delimited offset tuple for this function. "
                            "Use this for finer control over the offset than "
                            "per_dim_offset allows."))
    parser.add_option("", "--offset_multipliers", type="str", default=None,
                      dest="offset_multipliers",
                      help=("Comma-delimited offset multiplier tuple. "
                            "Advances in the same way as per_dim_offsets, "
                            "but allows each dimension to be specified "
                            "separately.  The offset paramter takes "
                            "precedence if specified."))
    parser.add_option("", "--per_dim_initial_vmax_multiplier", type="float",
                      default="0.5", dest="per_dim_initial_vmax_multiplier",
                      help=("Per-dimension vmax multiplier for particle "
                            "initialization, applied to half the distance "
                            "between corresponding bounds."))
    parser.add_option("", "--particles", type="int", default=20,
                      dest="initial_particles",
                      help=("Number of initial particles in the swarm "
                            "[%default]."))
    parser.add_option("", "--topology", dest="topology",
                      default="Star",
                      help=("Topology of the swarm [%default]. "
                            "Possible values are ({0})").format(
                                ", ".join(TOPOLOGY_CLASSES)))
    parser.add_option("", "--seed", type="str", default=None,
                      dest="seed",
                      help=("Random seed to use: "
                            "None means use time-based seed"))
    parser.add_option("", "--log_to_file", type="str", default=None,
                      dest="log_to_file",
                      help=("If specified, log everything to the given "
                            "filename."))
    parser.add_option("", "--log_vectors", dest="log_vectors", default=False,
                      action="store_true",
                      help="Log pos,vel vectors with other output")
    parser.add_option("", "--log_every", dest="log_every", default=1, type=int,
                      help="Log every n evaluations")


    for varname, default, desc in Simulation.variables:
        long_opt = "--{0}".format(varname)
        special_args = dict()
        if type(default) is bool:
            special_args = dict(type="int")
        else:
            special_args = dict(type=type(default).__name__)
        parser.add_option("", long_opt, default=default,
                          dest=varname,
                          help=("{0}: {1} [%default]".format(varname, desc)),
                          **special_args)

    options, args = parser.parse_args()

    return options, args


def main(options, args):
    log = Logger(options.log_to_file)

    sim_variables = dict((vname, getattr(options, vname))
                         for vname, _, _ in Simulation.variables)

    dim = options.dim
    func_class = FUNC_CLASSES[options.function]
    topo_class = TOPOLOGY_CLASSES[options.topology]
    bad_spec_message = ("You specified both dim ({dim}) and "
                        "offset (length {offset_len}), and "
                        "they are not equal.")
    if options.offset is None:
        if options.offset_multipliers is None:
            if dim is None:
                raise IncompleteDimensionSpec("You must specify either a "
                                              "number of dimensions or one of "
                                              "the offset vectors.")
            if options.per_dim_offset is None:
                if options.per_dim_offset_multiplier is None:
                    # No offset specified - offset is vector of zeros
                    offset = (0.0,) * dim
                else:
                    # offset multiplier specified, set offset vector to the
                    # scaled distance from constraint center to edge.
                    m = options.per_dim_offset_multiplier
                    offset = np.array([m * (r - l) / 2
                                       for l, r in func_class.dim_bounds(dim)])
            else:
                # per_dim_offset is specified absolutely, so use raw value
                offset = (options.per_dim_offset,) * dim
        else:
            # offset multiplier vector specified - calculate the offset in each
            # dimension
            mults = np.array([float(x)
                              for x in options.offset_multipliers.split(',')])
            if dim is None:
                dim = len(mults)
            elif dim != len(mults):
                raise IncompatibleDimensionSpec(
                    bad_spec_message.format(dim=dim, offset_len=len(mults)))

            bounds = func_class.dim_bounds(dim)
            offset = np.array([m * (r - l) / 2
                               for m, (l, r) in zip(mults, bounds)])
    else:
        offset = np.array([float(x) for x in options.offset.split(',')])
        if dim is None:
            dim = offset.size
        elif dim != offset.size:
            raise IncompatibleDimensionSpec(
                bad_spec_message.format(dim=dim, offset_len=offset.size))

    rand = np.random.RandomState(abs(hash(options.seed)))
    func = func_class(dim=dim, offset=offset)
    num_particles = options.initial_particles
    sim = Simulation(num_initial_particles=num_particles,
                     init_strategy=PVCube,
                     topo_strategy=topo_class,
                     opt_function=func,
                     random=rand,
                     **sim_variables)

    log("Mercurial_Revision: {0}".format(os.popen('hg id').read().strip()))
    log("Options:")
    for k, v in options.__dict__.items():
        log("  {k}: {v}".format(k=k, v=v))
    log("Computed:")
    log("  Constraints: {0!r}".format(sim.function.constraints))
    log("  Computed Offset: {0}".format(offset))

    start_time = time.time()
    last_time = start_time
    best_so_far = None
    last_output = -1

    try:
        eval_iter = ((b, a, p) for b, a, p in sim if a == EVALUATE)
        for i, (batch, action, particle) in enumerate(eval_iter):
            if action != EVALUATE:
                continue

            if options.log_to_file and not batch % options.log_every:
                pstr = particle.output(options.log_vectors)
                log("{0}:{1}:{2}".format(batch, action, pstr), to_stderr=False)

            if (best_so_far is None or
                best_so_far.best_value > particle.best_value):
                best_so_far = particle
            if i > last_output + 500:
                t = time.time()
                if t - last_time > 0.25:
                    log(i, best_so_far.best_value, skip_file=True)
                    last_time = t
                    last_output = i
        if i != last_output:
            log(i, best_so_far.best_value, skip_file=True)
    finally:
        log("Seconds elapsed: {0}".format(last_time - start_time))
        log("Best Particle: {0}".format(best_so_far))


if __name__ == '__main__':
    options, args = setup()

    if options.run_tests:
        import doctest
        doctest.testmod()
    elif options.profile:
        import cProfile
        import pstats
        cProfile.run('main(options, args)', 'pso_prof')
        p = pstats.Stats('pso_prof')
        p.print_stats()
    else:
        main(options, args)
