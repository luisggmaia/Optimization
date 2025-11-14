from numpy import clip, random
from math import inf

class Particle:

    def __init__(self, inf_x, sup_x):

        self.x = random.uniform(inf_x, sup_x)
        self.v = random.uniform(-(sup_x - inf_x), (sup_x - inf_x))
        self.best_x = self.x
        self.J_best = inf


class Swarm:

    def __init__(self, hyperparams, inf_x, sup_x):

        self.n_particles = hyperparams.n_particles
        self.particles = [None]*self.n_particles
        
        self.w = hyperparams.w
        self.phi_p = hyperparams.phi_p
        self.phi_g = hyperparams.phi_g

        self.inf_x = inf_x
        self.sup_x = sup_x
        self.inf_v = -(self.sup_x - self.inf_x)
        self.sup_v = (self.sup_x - self.inf_x)
        
        self.best_global = None
        self.J_best = inf

    def initialize_particles(self):

        for i in range(self.n_particles):
            self.particles[i] = Particle(self.inf_x, self.sup_x)

        self.best_global = Particle(self.inf_x, self.sup_x)

    def update_particles(self, J):

        best_iter = Particle(self.inf_x, self.sup_x)
        
        for particle in self.particles:
            r_p = random.uniform()
            r_g = random.uniform()

            v = self.w*particle.v + self.phi_p*r_p*(particle.best_x - particle.x) + self.phi_g*r_g*(self.best_global.x - particle.x)
            x = particle.x + v

            # self.x = max(min(x, self.sup_x), self.inf_x)
            x = clip(x, self.inf_x, self.sup_x)
            # self.v = max(min(v, self.sup_v), self.inf_v)
            v = clip(v, self.inf_v, self.sup_v)

            particle.x = x
            particle.v = v

            J_x = J(particle.x)

            if J_x < particle.J_best:
                particle.best_x = particle.x
                particle.J_best = J_x
            if J_x < best_iter.J_best:
                best_iter = particle

        return best_iter
    
def pso(swarm, J, n_eval):

    i = 1
    while i <= n_eval:

        best_iter = swarm.update_particles(J)

        if best_iter.J_best < swarm.J_best:
            swarm.best_global = best_iter
            swarm.J_best = best_iter.J_best

        print(f"Iteration {i}: Best J = {swarm.J_best}")

        i += 1

    return swarm.best_global.x, swarm.J_best



