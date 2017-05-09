import copy
import csv

from neat.math_util import mean, stdev
from neat.reporting import BaseReporter
from neat.six_util import iteritems


# TODO: Make a version of this reporter that doesn't continually increase memory usage.
# (Maybe periodically write blocks of history to disk, or log stats in a database?)

class StatisticsReporter(BaseReporter):
    def __init__(self):
        BaseReporter.__init__(self)
        self.most_fit_genomes = []
        self.generation_statistics = []
        #self.generation_cross_validation_statistics = []

    def post_evaluate(self, config, population, species, best_genome):
        self.most_fit_genomes.append(best_genome.fitness) #append(copy.deepcopy(best_genome))

        # Store the fitnesses of the members of each currently active species.
        species_stats = {}
        #species_cross_validation_stats = {}
        for sid, s in iteritems(species.species):
            species_stats[sid] = dict((k, v.fitness) for k, v in iteritems(s.members))
            #species_cross_validation_stats[sid] = dict((k, v.cross_fitness) for k, v in iteritems(s.members))
        self.generation_statistics.append(species_stats)
        #self.generation_cross_validation_statistics.append(species_cross_validation_stats)

    def get_fitness_stat(self, f):
        stat = []
        for stats in self.generation_statistics:
            scores = []
            for species_stats in stats.values():
                scores.extend(species_stats.values())
            stat.append(f(scores))

        return stat

    def get_fitness_mean(self):
        """Get the per-generation average fitness."""
        return self.get_fitness_stat(mean)

    def get_fitness_stdev(self):
        """Get the per-generation standard deviation of the fitness."""
        return self.get_fitness_stat(stdev)

    def get_average_cross_validation_fitness(self):
        """Get the per-generation average cross_validation fitness."""
        avg_cross_validation_fitness = []
        for stats in self.generation_cross_validation_statistics:
            scores = []
            for fitness in stats.values():
                scores.extend(fitness)
            avg_cross_validation_fitness.append(mean(scores))

        return avg_cross_validation_fitness

    def get_species_sizes(self):
        all_species = set()
        for gen_data in self.generation_statistics:
            all_species = all_species.union(gen_data.keys())

        max_species = max(all_species)
        species_counts = []
        for gen_data in self.generation_statistics:
            species = [len(gen_data.get(sid, [])) for sid in range(1, max_species + 1)]
            species_counts.append(species)

        return species_counts

    def get_species_fitness(self, null_value=''):
        all_species = set()
        for gen_data in self.generation_statistics:
            all_species = all_species.union(gen_data.keys())

        max_species = max(all_species)
        species_fitness = []
        for gen_data in self.generation_statistics:
            member_fitness = [gen_data.get(sid, []) for sid in range(1, max_species + 1)]
            fitness = []
            for mf in member_fitness:
                if mf:
                    fitness.append(mean(mf))
                else:
                    fitness.append(null_value)
            species_fitness.append(fitness)

        return species_fitness


