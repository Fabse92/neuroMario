# -*- coding: utf-8 -*-
__author__ = "Sergey Karakovskiy, sergey at idsia fullstop ch"
__date__ = "$August 26, 2010 1:33:34 PM$"

from marioagent import MarioAgent

class NEATagent(MarioAgent):
    """ In fact the Python twin of the
        corresponding Java ForwardJumpingAgent.
    """
    action = [0, 0, 0, 0, 0]
    actionStr = None
    KEY_LEFT = 0
    KEY_RIGHT = 1
    KEY_DUCK = 2
    KEY_JUMP = 3
    KEY_SPEED = 4
    mayMarioJump = None
    isMarioOnGround = None
    marioFloats = None
    enemiesFloats = None
    isEpisodeOver = False
    marioState = None       
    groups = None
    env = []
    collectedCoins = 0
    newCoins = 0
        
    def setGroups(self, groups):
      self.groups = groups
      for i in range(len(self.groups)):
        self.env.append(-1)
      
    def getName(self):
        return self.agentName

    def reset(self):
        self.action = [0, 0, 0, 0, 0]
        self.isEpisodeOver = False
        self.collectedCoins = 0
        
    def __init__(self):
        """Constructor"""
        self.reset()
        self.actionStr = ""
        self.agentName = "NEAT agent"
        
    def getAction(self, network, vis):
      """ Possible analysis of current observation and sending an action back
      """
      if (self.isEpisodeOver):
        return (1, 1, 1, 1, 1)
      NEATinput = [1] + [self.marioState] + self.env
        
      NEATinput = tuple(NEATinput)
      # the given network is activated with the current inputs
      output = network.activate(NEATinput)
      
      if vis:
        print "Input: " + str(NEATinput)        
        print "output: "+ str(output)        
        
      if output[self.KEY_LEFT] > output[self.KEY_RIGHT] + 0.1:
      # go left
        self.action[self.KEY_LEFT] = 1
        self.action[self.KEY_RIGHT] = 0
      elif output[self.KEY_LEFT] + 0.1 < output[self.KEY_RIGHT]:
      # go right
        self.action[self.KEY_LEFT] = 0
        self.action[self.KEY_RIGHT] = 1
      else:
      # neither left nor right
        self.action[self.KEY_LEFT] = 0
        self.action[self.KEY_RIGHT] = 0
        
      if output[self.KEY_JUMP] > output[self.KEY_DUCK] + 0.1:
      # jump
        self.action[self.KEY_JUMP] = 1
        self.action[self.KEY_DUCK] = 0
      elif output[self.KEY_JUMP] + 0.1 < output[self.KEY_DUCK]:
        self.action[self.KEY_JUMP] = 0
        self.action[self.KEY_DUCK] = 1
      else:
      # neither jump nor duck
        self.action[self.KEY_JUMP] = 0
        self.action[self.KEY_DUCK] = 0
        
      if output[self.KEY_SPEED] >= 0.5:
      # speed
        self.action[self.KEY_SPEED] = 1
      else:
      # no speed
        self.action[self.KEY_SPEED] = 0
                
      t = tuple(self.action)
      return t

    def integrateObservation(self, squashedObservation, squashedEnemies, marioPos, enemiesPos, marioState, evaluationInfo = None):
      """This method stores the observation inside the agent"""    
      self.env = []
      for i in range(len(self.groups)):
        self.env.append(0.15) # all cells initialized as containing nothing
      for i in range(len(self.groups)):
        for cell in self.groups[i]:
          if squashedObservation[cell] < 0:
            self.env[i] = -0.15 # obstalce
          elif squashedObservation[cell] == 2:
            self.env[i] = 1 # coin
          
      for i in range(len(self.groups)):
        for cell in self.groups[i]:
          if squashedEnemies[cell] > 5:
            self.env[i] = -1 # enemies

      self.newCoins = evaluationInfo[10] - self.collectedCoins
      self.collectedCoins += self.newCoins
                           
      self.marioFloats = marioPos
      self.enemiesFloats = enemiesPos
      self.mayMarioJump = marioState[3]
      self.isMarioOnGround = marioState[2]
      self.marioState = marioState[1]
      
