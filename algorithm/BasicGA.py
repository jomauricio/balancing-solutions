from deap import tools
from deap import algorithms

class BasicGA():

    def run(self, config, cxpb, mutpb, ngen, stats=None,
                 halloffame=None, verbose=__debug__):

        logbook = tools.Logbook()
        sequencelog = {}
        logbook.header = ['gen', 'nevals', 'pop', 'hof'] + (stats.fields if stats else [])

        # Evaluate the individuals with an invalid fitness
        # invalid_ind = [ind for ind in config.pop if not ind.fitness.valid]
        # fitnesses = config.toolbox.map(config.toolbox.evaluate, invalid_ind)
        # for ind, fit in zip(invalid_ind, fitnesses):
        #     ind.fitness.values = fit


        nevals = 0
        # Evaluate the individual whose sequence is yet to be evaluated
        for ind in config.pop:
            seq = list(filter(lambda x: x != 0, ind))
            if(str(seq) not in sequencelog):
                nevals += 1
                fit = config.toolbox.evaluate(seq)
                sequencelog[str(seq)] = fit
                ind.fitness.values = fit
            else:
                ind.fitness.values = sequencelog[str(seq)]

        if halloffame is not None:
            halloffame.update(config.pop)

        record = stats.compile(config.pop) if stats else {}
        logbook.record(gen=0, nevals=nevals, pop=config.pop, hof=halloffame, **record)
        if verbose:
            print(logbook.stream)

        # Begin the generational process
        for gen in range(1, ngen + 1):
            # Select the next generation individuals
            offspring = config.toolbox.select(config.pop, len(config.pop))

            # Vary the pool of individuals
            offspring = algorithms.varAnd(offspring, config.toolbox, cxpb, mutpb)

            # Evaluate the individuals with an invalid fitness
            # invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            # fitnesses = config.toolbox.map(config.toolbox.evaluate, invalid_ind)
            # for ind, fit in zip(invalid_ind, fitnesses):
            #     ind.fitness.values = fit

            # Evaluate the individual whose sequence is yet to be evaluated
            nevals = 0
            for ind in offspring:
                seq = list(filter(lambda x: x != 0, ind))
                if (str(seq) not in sequencelog):
                    nevals += 1
                    fit = config.toolbox.evaluate(seq)
                    sequencelog[str(seq)] = fit
                    ind.fitness.values = fit
                else:
                    ind.fitness.values = sequencelog[str(seq)]

            # Update the hall of fame with the generated individuals
            if halloffame is not None:
                halloffame.update(offspring)

            # Replace the current population by the offspring
            config.pop[:] = offspring

            # Append the current generation statistics to the logbook
            record = stats.compile(config.pop) if stats else {}
            logbook.record(gen=gen, nevals=nevals, pop=config.pop, hof=halloffame, **record)
            if verbose:
                print(logbook.stream)
        return config.pop, logbook