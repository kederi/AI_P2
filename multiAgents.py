# multiAgents.py
# --------------
# This codebase is adapted from UC Berkeley AI. Please see the following information about the license.
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        food_list = successorGameState.getFood().asList()

        if (successorGameState.isWin()):
            return 100000000
        # To begin write a simple evaluation function in which PacMan just avoids death
        # TODO
        # Get the Manhattan Distance between PacMan and the Ghosts
        for state in newGhostStates:
            man_distGhost = manhattanDistance(newPos, state.getPosition())
            # Check if its the min distance to pac man
            # CHANGE
            if (state.scaredTimer == 0 and man_distGhost < 3):
                return -1000000
        ## Need to get the closest pellet
        list_food = []
        for dot in food_list:
            list_food.append(manhattanDistance(newPos, dot))

        val = min(list_food)
        return successorGameState.getScore() + 1 / val


def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        def max_value(state, depth):
            max_score = float("-inf")
            return_action = Directions.STOP
            for action in state.getLegalActions(self.index):
                score = value(state.generateSuccessor(0, action), depth, 1)
                if score > max_score:
                    max_score = score
                    return_action = action
            # if depth = 0, this is the final value that minimax returns, so we return the best action rather than score
            if depth == 0:
                return return_action
            else:
                return max_score

        def min_value(state, depth, ghost_index):
            score = float("inf")
            for action in state.getLegalActions(ghost_index):
                # if this is the last ghost, the next agent is pacman and we increase depth by 1
                if ghost_index == state.getNumAgents() - 1:
                    score = min(value(state.generateSuccessor(ghost_index, action), depth + 1, 0), score)
                else:
                    score = min(value(state.generateSuccessor(ghost_index, action), depth, ghost_index+1), score)
            return score

        def value(state, depth, agent):
            # call the self evaluation function if the maximum depth is reached, or a win or lose state is reached
            if state.isWin() or state.isLose() or depth == self.depth:
                return self.evaluationFunction(state)

            if agent == 0:  # pacman's turn
                return max_value(state, depth)
            else:   # current agent is a ghost
                return min_value(state, depth, agent)

        return value(gameState, 0, 0)




class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        def max_value(state, a, b, depth):
            max_score = float("-inf")
            return_action = Directions.STOP
            for action in state.getLegalActions(self.index):
                score = value(state.generateSuccessor(0, action), a, b, depth, 1)
                if score > max_score:
                    max_score = score
                    return_action = action
                # alpha-beta pruning
                if max_score > b:
                    return max_score
                a = max(a, max_score)
            # if depth = 0, this is the final value that minimax returns, so we return the best action rather than score
            if depth == 0:
                return return_action
            else:
                return max_score

        def min_value(state, a, b, depth, ghost_index):
            score = float("inf")
            for action in state.getLegalActions(ghost_index):
                # if this is the last ghost, the next agent is pacman and we increase depth by 1
                if ghost_index == state.getNumAgents() - 1:
                    score = min(value(state.generateSuccessor(ghost_index, action), a, b, depth + 1, 0), score)
                else:
                    score = min(value(state.generateSuccessor(ghost_index, action), a, b, depth, ghost_index+1), score)
                # alpha-beta pruning
                if score < a:
                    return score
                b = min(b, score)
            return score

        def value(state, a, b, depth, agent):
            # call the self evaluation function if the maximum depth is reached, or a win or lose state is reached
            if state.isWin() or state.isLose() or depth == self.depth:
                return self.evaluationFunction(state)

            if agent == 0:  # pacman's turn
                return max_value(state, a, b, depth)
            else:   # current agent is a ghost
                return min_value(state, a, b, depth, agent)

        return value(gameState, float('-inf'), float('inf'), 0, 0)


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"

        def max_value(state, depth):
            max_score = float("-inf")
            return_action = Directions.STOP
            for action in state.getLegalActions(self.index):
                score = value(state.generateSuccessor(0, action), depth, 1)
                if score > max_score:
                    max_score = score
                    return_action = action
            # if depth = 0, this is the final value that minimax returns, so we return the best action rather than score
            if depth == 0:
                return return_action
            else:
                return max_score

        def exp_value(state, depth, ghost_index):
            score = 0
            legal_actions = state.getLegalActions(ghost_index)
            p = 1/len(legal_actions)    # each action is given an equal probability of being chosen
            for action in legal_actions:
                # if this is the last ghost, the next agent is pacman and we increase depth by 1
                if ghost_index == state.getNumAgents() - 1:
                    score += p * value(state.generateSuccessor(ghost_index, action), depth + 1, 0)
                else:
                    score += p * value(state.generateSuccessor(ghost_index, action), depth, ghost_index + 1)
            return score

        def value(state, depth, agent):
            # call the self evaluation function if the maximum depth is reached, or a win or lose state is reached
            if state.isWin() or state.isLose() or depth == self.depth:
                return self.evaluationFunction(state)

            if agent == 0:  # pacman's turn
                return max_value(state, depth)
            else:  # current agent is a ghost
                return exp_value(state, depth, agent)

        return value(gameState, 0, 0)

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    pos = currentGameState.getPacmanPosition()
    food_list = currentGameState.getFood().asList()
    ghost_states = currentGameState.getGhostStates()
    score = currentGameState.getScore()

    # find distance to closest ghost
    ghost_dist = []
    for ghost in ghost_states:
        ghost_dist.append(manhattanDistance(pos, ghost.getPosition()))
    ghost_dist.sort()
    closest_ghost = min(ghost_dist)


    # find distance to closest food
    food_dist = []
    avg_food = -1
    closest_food = -1
    if len(food_list) > 0:
        for dot in food_list:
            food_dist.append(manhattanDistance(pos, dot))
        closest_food = min(food_dist)
        if len(food_list) > 1:
             avg_food = sum(food_dist)/len(food_dist)

    if 0 <= closest_food < closest_ghost - 1 or closest_food < 5:
        score *= 2
    if closest_ghost > 5:
        score *= 1.75
    if avg_food > 5:
        score -= avg_food
    score -= closest_food

    return score



# Abbreviation
better = betterEvaluationFunction
