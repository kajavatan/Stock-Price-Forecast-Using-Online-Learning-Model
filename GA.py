# ----------------------------------------------------------------------
# MIT License
# Copyright (c) 2018 Lee Min Hua. All rights reserved.
# Author: Lee Min Hua
# E-mail: mhlee2907@gmail.com

# ----------------------------------------------------------------------

"""
Implementation of Genetic Algorithm (GA) for hyperparameter optimization
"""

import random
from run import initializedata
from run import calerror

class GA(object):
    def __init__(self, instance):
        self.instance = instance
        self.counter = 0
        self.output = []
        self.sequence = initializedata()

    def run(self):
        population = self.instance.initializePopulation()
        pop_len = len(population)
        fitness_population = [(self.instance.getFitness(individual, self.sequence), individual) for individual in population]


        for individual in fitness_population:
            a, b = individual
            decoded = self.instance.decode(b)

        while 1:

            assert(len(fitness_population) == pop_len)
            self.counter += 1

            if self.instance.showAndCheck(fitness_population, self.counter):
                print 'Best solution found at generation', self.counter
                break;

            fitness_population = self.changePopulation(fitness_population)

    def changePopulation(self, fitness_population):

        parent1, parent2 = self.instance.selectParents(fitness_population)
        parents = (parent1[1],parent2[1])

        allChildren = []

        if random.random() < self.instance.getCrossThreshold():
            children = self.instance.crossover(parents)  # Assigned the crossovered chromosome to children
        else:
            children = parents

        child1, child2 = children

        if random.random() < self.instance.getMutationThreshold():
            child1 = self.instance.mutate(child1)
        else:
            child1 = child1

        if random.random() < self.instance.getMutationThreshold():
            child2 = self.instance.mutate(child2)
        else:
            child2 = child2

        allChildren.append(parent1)
        allChildren.append(parent2)
        child1_fitness = self.instance.getFitness(child1, self.sequence)
        child2_fitness = self.instance.getFitness(child2, self.sequence)
        allChildren.append((child1_fitness, child1))
        allChildren.append((child2_fitness, child2))

        decoded1 = self.instance.decode(child1)
        decoded2 = self.instance.decode(child2)

        allChildren = set(allChildren)
        allChildren = sorted(allChildren)

        allChildren = allChildren[-2:]

        # remove parent 1 and 2 from the population
        fitness_population.remove(parent1)
        fitness_population.remove(parent2)

        fitness_population += allChildren

        return fitness_population


class rules(object):
    def __init__(self, maxLoopsNum, error, populationSize, crossThreshold, mutationThreshold):

        self.maxLoopsNum = maxLoopsNum
        self.error = error
        self.populationSize = populationSize
        self.crossThreshold = crossThreshold
        self.mutationThreshold = mutationThreshold


    def getMutationThreshold(self):
        return self.mutationThreshold

    def getCrossThreshold(self):
        return self.crossThreshold

    def initializePopulation(self):
        population = []
        for i in range(self.populationSize):
            population.append(str(random.randint(1, 200)).zfill(3) + str(random.randint(1, 100)).zfill(3))
        return population

    def decode(self, individual):
        solutions = []
        solutions.append(int(individual[:3]))
        solutions.append(int(individual[-3:]))

        return solutions

    def getFitness(self, individual, sequence):
        # Decode the individual (chromosome) and calculate the fitness\
        fitness = - round((gradeFunction(self.decode(individual),sequence)),4)
        print 'fitness = {}'.format(fitness)
        return fitness

    def showAndCheck(self, fitness_population,counter):

        best = list(sorted(fitness_population))[-1]  # Last Individual
        print "Generation", counter, "Best solution: numlags = ", self.decode(best[1])[0], "neurons =",self.decode(best[1])[1], "with fitness:", best[0]
        return (counter > self.maxLoopsNum) or (best[0] >= self.error)

    def selectParents(self, fitness_population):
        # Construct a iterator here
        # Use Tournament Selection
        parent1 = self.tournament(fitness_population)
        parent2 = parent1
        while parent2 == parent1:
            parent2 = self.tournament(fitness_population)

        return parent1, parent2

    def crossover(self, parents):
        parent1, parent2 = parents

        child1 = parent1[:3] + parent2[-3:]
        child2 = parent2[:3] + parent1[-3:]

        return (child1, child2)

    def mutate(self, child):
        decoded = self.decode(child)

        rd = random.randint(-50,50)
        cond = True

        if random.randint(0,1) == 1:
            while cond:
                temp = decoded[1] + rd
                if 1 <= temp <= 100:
                    cond = False
            decoded[1] = temp
        else:
            while cond:
                temp = decoded[0] + rd
                if 1<= temp <= 200:
                    cond = False
            decoded[0] = temp

        result = str(decoded[0]).zfill(3) + str(decoded[1]).zfill(3)

        return result

    def tournament(self, fitness_population):
        fit1, ch1 = fitness_population[random.randint(0, len(fitness_population) - 1)]
        fit2, ch2 = fitness_population[random.randint(0, len(fitness_population) - 1)]

        return (fit1,ch1) if fit1 > fit2 else (fit2,ch2)


def gradeFunction(decoded_individual, sequence):
    numlags = decoded_individual[0]
    numneurons = decoded_individual[1]
    return calerror(sequence,numlags,numneurons)

GA(rules(10, -0.035, 5, 0.5,0.2)).run()  # maxLoopsNum, error, populationSize, crossThreshold, mutationThreshold
