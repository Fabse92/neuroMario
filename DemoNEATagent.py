# -*- coding: utf-8 -*-
__author__ = "Sergey Karakovskiy, sergey at idsia fullstop ch"
__date__ = "$${date} ${time}$"

import sys
import os
import math

import ctypes
from ctypes import *
from ctypes import POINTER
from ctypes import c_int
from ctypes import py_object

from ctypes.util import find_library
from neatagent import NEATagent
import numpy

from evaluationinfo import EvaluationInfo

import neat
import visualize
import checkpointPlus
import statisticsPlus
import random

class ListPOINTER(object):
    '''Just like a POINTER but accept a list of ctype as an argument'''
    def __init__(self, etype):
        self.etype = etype

    def from_param(self, param):
        if isinstance(param, (list, tuple)):
#            print "Py: IS INSTANCE"
            return (self.etype * len(param))(*param)
        else:
#            print "Py: NOT INSTANCE"
            return param

class ListByRef(object):
    '''An argument that converts a list/tuple of ctype elements into a pointer to an array of pointers to the elements'''
    def __init__(self, etype):
        self.etype = etype
        self.etype_p = POINTER(etype)

    def from_param(self, param):
        if isinstance(param, (list, tuple)):
            val = (self.etype_p * len(param))()
            for i, v in enumerate(param):
                if isinstance(v, self.etype):
                    val[i] = self.etype_p(v)
                else:
                    val[i] = v
            return val
        else:
            return param

def from_param(self, param):
    if isinstance(param, (list, tuple)):
        return (self.etype * len(param))(*param)
    else:
        return param

def cfunc(name, dll, result, * args):
    '''build and apply a ctypes prototype complete with parameter flags'''
    atypes = []
    aflags = []
    for arg in args:
        atypes.append(arg[1])
        aflags.append((arg[2], arg[0]) + arg[3:])
    return CFUNCTYPE(result, * atypes)((name, dll), tuple(aflags))
        
class MarioNet():
  def __init__(self, population = None, feedforward = True):
    
    self.initNeat(population)
    self.initAmico()   
    
    #if reducedGrid:
    #  groups = [[0, 1, 2, 3, 11, 12, 13, 14, 22, 23, 33, 34], [24, 25, 35, 36], [4, 5, 6, 15, 16, 17], [26, 37], [27, 38], [28, 39], [7, 8, 9, 10, 18, 19, 20, 21, 31, 32, 42, 43], [29, 30, 40, 41], [44, 45, 55, 56], [46, 47, 57, 58], [48, 59], [50, 61], [51, 52, 62, 63], [53, 54, 64, 65], [66, 67, 77, 78, 88, 89, 99, 100, 110, 111], [68, 69, 79, 80, 81, 90, 91, 92], [70], [101, 102, 103, 112, 113, 114], [72], [73, 74, 83, 84, 85, 94, 95, 96], [105, 106, 107, 116, 117, 118], [75, 76, 86, 87, 97, 98, 108, 109, 119, 120]]
    
    groups = []
    for i in range(121):
      groups.append([i])    
    
    self.feedforward = feedforward
    self.randomLevels = False
    self.levels = [0]
    self.bestFitness = 0
    self.perecentageOfSuccessfulAgents = 0.0
    self.vis = False    
    self.agent = NEATagent() # agent    
    self.agent.setGroups(groups)
    self.round = 0
    self.iteration = 0
    self.ll = 180
    self.options = "-z on -tl 60 -rfh 11 -rfw 11 -ll " + str(self.ll)
    
  def initNeat(self, population):
    # Create the population, which is the top-level object for a NEAT run.
    if population is None:
      if feedforward:
        configFile = 'config-feedforward'
      else:
        configFile = 'config-recurrent'
      self.config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                       neat.DefaultSpeciesSet, neat.DefaultStagnation,
                       configFile)
      self.pop = neat.Population(self.config)
      self.stats = statisticsPlus.StatisticsReporter()
      self.best_genome = None
    else:      
      self.pop, self.stats = checkpointPlus.CheckpointerPlus.restore_checkpoint(population)
      self.config = self.pop.config
      self.best_genome = self.pop.best_genome

    # Add a stdout reporter to show progress in the terminal.
    self.pop.add_reporter(neat.StdOutReporter(True))    
    self.pop.add_reporter(self.stats)
    #self.checkpointer = neat.Checkpointer(generation_interval = 1, time_interval_seconds = None)
    self.checkpointer = checkpointPlus.CheckpointerPlus()
    self.pop.add_reporter(self.checkpointer)
    
  def initAmico(self):
    """simple AmiCo env interaction"""
    print "Py: AmiCo Simulation Started:"
    print "library found: "
    print "Platform: ", sys.platform
    if (sys.platform == 'linux2'):
      ##########################################
      # find_library on Linux could only be used if your libAmiCoPyJava.so is
      # on system search path or path to the library is added in to LD_LIBRARY_PATH
      #
      # name =  'AmiCoPyJava'
      # loadName = find_library(name)
      ##########################################
      loadName = './libAmiCoPyJava.so'
      libamico = ctypes.CDLL(loadName)
      print libamico
    else: #else if OS is a Mac OS X (libAmiCo.dylib is searched for) or Windows (AmiCo.dll)
      name =  'AmiCoPyJava'
      loadName = find_library(name)
      print loadName
      libamico = ctypes.CDLL(loadName)
      print libamico
    
    javaClass = "ch/idsia/benchmark/mario/environments/MarioEnvironment"
    libamico.amicoInitialize(1, "-Djava.class.path=." + os.pathsep + ":jdom.jar")
    libamico.createMarioEnvironment(javaClass)

    self.libamico = libamico
    self.reset = cfunc('reset', libamico, None, ('list', ListPOINTER(c_int), 1))
    self.getEntireObservation = cfunc('getEntireObservation', libamico, py_object,
                                 ('list', c_int, 1),
                                 ('zEnemies', c_int, 1))
    self.performAction = cfunc('performAction', libamico, None, ('list', ListPOINTER(c_int), 1))
    self.getEvaluationInfo = cfunc('getEvaluationInfo', libamico, py_object)
    self.getObservationDetails = cfunc('getObservationDetails', libamico, py_object)

  def visualizeBestGenome(self):
    if self.feedforward:
      network = neat.nn.FeedForwardNetwork.create(self.best_genome, self.config)
    else:
      network = neat.nn.RecurrentNetwork.create(self.best_genome, self.config)
    for level in self.levels:
      lastObs1 = None
      lastObs2 = None
      stagnancyCount = 0
      options = self.options + " -vis on -gv off -fps 24 -ls " + str(level) 
      print "Running visualition with options: " + options
      self.reset(options)
      obsDetails = self.getObservationDetails()
      while (not self.libamico.isLevelFinished()):
        self.libamico.tick();
        obs = self.getEntireObservation(1, 0)
        self.agent.integrateObservation(obs[0], obs[1], obs[2], obs[3], obs[4], self.getEvaluationInfo());
        action = self.agent.getAction(network, True)
        self.performAction(action);
        if obs[2] == lastObs1 or obs[2] == lastObs2:
          stagnancyCount += 1
          if stagnancyCount >= 200:
            break
        else:
          stagnancyCount = 0
        lastObs2 = lastObs1
        lastObs1 = obs[2]        
      evaluationInfo = self.getEvaluationInfo()
      print "evaluationInfo = \n", EvaluationInfo(evaluationInfo);
      #print self.agent.collectedCoins

  def eval_genome(self, genome, config):
    newBestFitness = False
    if self.feedforward:
      network = neat.nn.FeedForwardNetwork.create(genome, config)
    else:
      network = neat.nn.RecurrentNetwork.create(genome, config)
    fitness = 0.0
    self.completedAllLevels = 0
    for level in self.levels:
      lastObs1 = None
      lastObs2 = None
      stagnancyCount = 0      
      options = self.options + " -vis off -gv off -ls " + str(level)
      #print options
      self.reset(options)
      obsDetails = self.getObservationDetails()
      while (not self.libamico.isLevelFinished()):
        self.libamico.tick();
        obs = self.getEntireObservation(1, 0)
        self.agent.integrateObservation(obs[0], obs[1], obs[2], obs[3], obs[4], self.getEvaluationInfo());
        action = self.agent.getAction(network, False)
        self.performAction(action);
        if obs[2] == lastObs1 or obs[2] == lastObs2:
          stagnancyCount += 1
          if stagnancyCount >= 200:
            break
        else:
          stagnancyCount = 0
        lastObs2 = lastObs1
        lastObs1 = obs[2]        
      self.agent.reset()
      reachedLength = obs[2][0]    
      maxLength = 16.0 * self.ll
      reachedPercentage = reachedLength/maxLength
      completedLevel = 0.0
      if reachedPercentage == 1.0:
        completedLevel = 1.0
      #print self.agent.collectedCoins
      fitness += self.getEvaluationInfo()[10] * 0.02 + reachedPercentage + completedLevel
      self.completedAllLevels += completedLevel
    fitness = fitness / len(self.levels)
    self.completedAllLevels = self.completedAllLevels / len(self.levels)
    #print "Completete All" + str(self.completedAllLevels)
    #print "Current fitness: " + str(fitness)    
    genome.fitness = fitness
    if ((self.bestFitness + 0.05) < fitness and fitness > 0.6):
      self.bestFitness = fitness
      self.best_genome = genome
      newBestFitness = True
      if self.vis:
        self.visualizeBestGenome()
    return newBestFitness
      
  def eval_genomes(self, genomes, config):    
    agentsAbleToReachEnd = 0.0
    agentsNotAbleToReachEnd = 0.0
    newBestFitness = False
    if self.randomLevels:
      self.levels = []
      for i in range(len(levels)):
        self.levels.append(random.randint(0,100000000)) 
    for genome_id, genome in genomes:
      if self.eval_genome(genome, config):
        newBestFitness = True
      if self.completedAllLevels == 1.0:
        agentsAbleToReachEnd += 1
      else:
        agentsNotAbleToReachEnd += 1
    if newBestFitness:
      fileName = '_'+str(self.round)+'_'+str(self.iteration)+'_'+str(self.levels)+'_'+str(self.bestFitness)
      self.checkpointer.save_checkpoint(self.pop, self.best_genome, self.stats, 'neuroMario/checkpoints/checkpoint'+fileName)
      visualize.draw_net(self.config, self.best_genome, view = False, filename='neuroMario/checkpoints/best_net'+fileName)
    self.perecentageOfSuccessfulAgents = agentsAbleToReachEnd / (agentsAbleToReachEnd + agentsNotAbleToReachEnd)
    #print "agentsNotAbleToReachEnd: " + str(agentsNotAbleToReachEnd) 
    #print "agentsAbleToReachEnd: " +str(agentsAbleToReachEnd)
    print "perecentageOfSuccessfulAgents: " + str(self.perecentageOfSuccessfulAgents)
    

if __name__ == "__main__":
  #groups = [[0,1,7,8],[2,3,4,9,10,11],[5,6,12,13],[14,15,21,22],[16,23],[18,25],[19,20,26,27],[28,29,35,36,37,42,43,44],[30],[32],[33,34,39,40,41,46,47,48]]  
  
  feedforward = False
  threshold = 0.1
  startRound = 0
  
  population = None
  #population = 'neuroMario/checkpoints/checkpoint_0_2.0'  
  
  experiment = MarioNet(population, feedforward)  
    
  #experiment.eval_genome(experiment.best_genome, experiment.config)
  #experiment.visualizeBestGenome()

  levels =       [1, 3, 5, 7, 3, 5, 7]
  difficulty =   [0, 1, 1, 1, 1, 1, 1]
  randomLevels = [0, 0, 0, 0, 1, 1, 1]
  vis =          [0, 0, 0, 0, 0, 0, 0] 

  for i in range(len(levels)):
    experiment.iteration = 0
    if startRound <= i:
      while experiment.perecentageOfSuccessfulAgents < threshold:
        experiment.iteration += 1
        experiment.round = i
        experiment.options = experiment.options + " -ld " + str(difficulty[i])
        experiment.levels = []
        for level in range(levels[i]):
          experiment.levels.append(random.randint(0,100000000)) 
        experiment.randomLevels = (randomLevels[i] == 1)    
        experiment.vis = (vis[i] == 1)  
        counter = 0
        while experiment.perecentageOfSuccessfulAgents == 0.0 and counter < 1000:
          winner = experiment.pop.run(experiment.eval_genomes, 1)
          counter += 1
        winner = experiment.pop.run(experiment.eval_genomes, 100)
        experiment.bestFitness = 0
      
        fileName = '_'+str(experiment.round)+'_'+str(experiment.iteration)+'_'+str(experiment.levels)+'_'+str(experiment.bestFitness)
        visualize.plot_stats(experiment.stats, ylog=False, view=False, filename='neuroMario/stats/stats'+fileName+'.png')
        visualize.plot_species(experiment.stats, view=False, filename='neuroMario/stats/species'+fileName+'.png')
  
  print "All done!"
  












