# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from game import *
import random, util, math
import sys
from learningAgents import ReinforcementAgent
from featureExtractors import *


class MultimaxAgent(Agent):
    def __init__(self, index=0, depth=5):
        self.depth = depth
        self.index = index
        self.nodeNum = 0

    def getNodeNum(self):
        return self.nodeNum

    def getAction(self, gameState):
        # print self.index
        scores, bestMultiAction = self.multimax(gameState, self.index, self.depth)
        return bestMultiAction

    def multimax(self, currentGameState, agentIndex, dep):
        self.nodeNum += 1
        legal = currentGameState.getLegalActions(agentIndex)
        if Directions.STOP in legal: legal.remove(Directions.STOP)

        if dep == 0 or not legal:
          return (currentGameState.getScores(), None)

        scoreActions = []

        for action in legal:
          childState = currentGameState.generateSuccessor(agentIndex, action)
          childScores, childAction = self.multimax(childState, (agentIndex + 1) % currentGameState.getNumAgents(), dep - 1)
          scoreActions.append((childScores[agentIndex], action))
        
        bestScore = max(scoreActions)[0]
        bestActions = [pair[1] for pair in scoreActions if pair[0] == bestScore]

        return (currentGameState.getScores(), random.choice(bestActions))



class GreedyAgent(Agent):

    def getAction(self, state):
        # Generate candidate actions
        legal = state.getLegalActions(self.index)
        if Directions.STOP in legal: legal.remove(Directions.STOP)

        successors = [(state.generateSuccessor(self.index, action), action) for action in legal]
        scored = [(state.getScores()[self.index], action) for state, action in successors]
        bestScore = max(scored)[0]
        bestActions = [pair[1] for pair in scored if pair[0] == bestScore]
        return random.choice(bestActions)

class RandomAgent(Agent):

    def getAction(self, state):
        dist = self.getDistribution(state)
        return util.chooseFromDistribution( dist )

    def getDistribution( self, state ):
        dist = util.Counter()
        for a in state.getLegalActions( self.index ): dist[a] = 1.0
        dist.normalize()
        return dist



