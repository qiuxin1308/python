# multiAgents.py
# --------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

from util import manhattanDistance
from game import Directions
from game import Actions
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
    some Directions.X for some X in the set {North, South, West, East, Stop}
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
    finalScores = successorGameState.getScore()
    foodPacDis = float('inf')

    #when pacman meets ghost, it will lose points
    for ghostStates in newGhostStates:
      ghostPos = ghostStates.getPosition()
      if ghostPos == newPos:
        finalScores = float('-inf')

    remainFood = newFood.asList()
    if len(remainFood) > 0:
      for food in currentGameState.getFood().asList():
        foodPacDis = min(foodPacDis,manhattanDistance(food,newPos))
        finalScores += 10.0/(foodPacDis + 1.0)

    if Directions.STOP in action:
      finalScores = float('-inf')


    return finalScores

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

      Directions.STOP:
        The stop direction, which is always legal

      gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

      gameState.getNumAgents():
        Returns the total number of agents in the game
    """
    "*** YOUR CODE HERE ***"
    return self.minimax_value(gameState,0)

  def minimax_value(self, state, depthValue):
      agentIndex = depthValue % state.getNumAgents()

      if state.isWin() or state.isLose():
        return self.evaluationFunction(state)

      if depthValue == self.depth * state.getNumAgents():
        return self.evaluationFunction(state)

      if agentIndex > 0:
        return self.min_value(state,depthValue)
      else:
        return self.max_value(state,depthValue)

      return 'None'


  def min_value(self, state, depthValue):
      agentIndex = depthValue % state.getNumAgents()

      value = float('inf')
      nextAction = 'None'

      legalActions = state.getLegalActions(agentIndex)
      if legalActions == None:
        return self.evaluationFunction(state)

      for actions in legalActions:
        if actions == Directions.STOP:
          continue
        successor = state.generateSuccessor(agentIndex,actions)
        tempValue = self.minimax_value(successor,depthValue + 1)

        if tempValue < value:
          value = min(tempValue,value)
          nextAction = actions

      return value



  def max_value(self, state, depthValue):
      value = float('-inf')
      nextAction = 'None'

      legalActions = state.getLegalActions(0)
      if legalActions == None:
        return self.evaluationFunction(state)

      for actions in legalActions:
        if actions == Directions.STOP:
          continue
        successor = state.generatePacmanSuccessor(actions)
        tempValue = self.minimax_value(successor,depthValue + 1)

        if tempValue > value:
          value = max(tempValue,value)
          nextAction = actions


      if depthValue == 0:
        return nextAction
      else:
        return value

    #util.raiseNotDefined()

class AlphaBetaAgent(MultiAgentSearchAgent):
  """
    Your minimax agent with alpha-beta pruning (question 3)
  """

  def getAction(self, gameState):
    """
      Returns the minimax action using self.depth and self.evaluationFunction
    """
    "*** YOUR CODE HERE ***"
    return self.alphaBeta_value(gameState,0,float('-inf'),float('inf'))


  def alphaBeta_value(self, state, depthValue,alpha,beta):
      agentIndex = depthValue % state.getNumAgents()

      if state.isWin() or state.isLose():
        return self.evaluationFunction(state)

      if depthValue == self.depth * state.getNumAgents():
        return self.evaluationFunction(state)

      if agentIndex > 0:
        return self.min_value(state,depthValue,alpha,beta)
      else:
        return self.max_value(state,depthValue,alpha,beta)

      return 'None'


  def min_value(self, state, depthValue,alpha,beta):
      agentIndex = depthValue % state.getNumAgents()

      value = float('inf')
      nextAction = 'None'

      legalActions = state.getLegalActions(agentIndex)
      if legalActions == None:
        return self.evaluationFunction(state)

      for actions in legalActions:
        if actions == Directions.STOP:
          continue
        successor = state.generateSuccessor(agentIndex,actions)
        tempValue = self.alphaBeta_value(successor,depthValue + 1,alpha,beta)

        if tempValue < value:
          value = min(tempValue,value)
          nextAction = actions

        if value <= alpha:
          return value

        beta = min(beta,value)

      return value



  def max_value(self, state, depthValue,alpha,beta):
      value = float('-inf')
      nextAction = 'None'

      legalActions = state.getLegalActions(0)
      if legalActions == None:
        return self.evaluationFunction(state)

      for actions in legalActions:
        if actions == Directions.STOP:
          continue;
        successor = state.generatePacmanSuccessor(actions)
        tempValue = self.alphaBeta_value(successor,depthValue + 1,alpha,beta)

        if tempValue > value:
          value = max(tempValue,value)
          nextAction = actions

        if value >= beta:
          return value

        alpha = max(alpha,value)


      if depthValue == 0:
        return nextAction
      else:
        return value

    #util.raiseNotDefined()

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
    return self.expectimax_value(gameState,0)

  def expectimax_value(self, state, depthValue):
      agentIndex = depthValue % state.getNumAgents()

      if state.isWin() or state.isLose():
        return self.evaluationFunction(state)

      if depthValue == self.depth * state.getNumAgents():
        return self.evaluationFunction(state)

      if agentIndex > 0:
        return self.exp_value(state,depthValue)
      else:
        return self.max_value(state,depthValue)

      return 'None'


  def exp_value(self, state, depthValue):
      agentIndex = depthValue % state.getNumAgents()

      value = 0
      nextAction = 'None'

      legalActions = state.getLegalActions(agentIndex)
      if legalActions == None:
        return self.evaluationFunction(state)

      p = 1.0/len(legalActions)

      for actions in legalActions:
        if actions == Directions.STOP:
          continue
        successor = state.generateSuccessor(agentIndex,actions)
        tempValue = self.expectimax_value(successor,depthValue + 1)

        value += p * tempValue
        nextAction = actions

      return value



  def max_value(self, state, depthValue):
      value = float('-inf')
      nextAction = 'None'

      legalActions = state.getLegalActions(0)
      if legalActions == None:
        return self.evaluationFunction(state)

      for actions in legalActions:
        if actions == Directions.STOP:
          continue
        successor = state.generatePacmanSuccessor(actions)
        tempValue = self.expectimax_value(successor,depthValue + 1)

        if tempValue > value:
          value = max(tempValue,value)
          nextAction = actions


      if depthValue == 0:
        return nextAction
      else:
        return value


    #util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
  """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
  """
  "*** YOUR CODE HERE ***"
  #successorGameState = currentGameState.generatePacmanSuccessor(action)
  currentPacmanPos = currentGameState.getPacmanPosition()
  currentFood = currentGameState.getFood()
  currentGhostStates = currentGameState.getGhostStates()
  currentCapsule = currentGameState.getCapsules()
  currentScaredTimes = [ghostState.scaredTimer for ghostState in currentGhostStates]

  finalScores = currentGameState.getScore()
  foodPacDis = 99999
  ghostPacDis = 99999

  #when pacman meets ghost, it will lose points
  for ghostStates in currentGhostStates:
    ghostPos = ghostStates.getPosition()
    ghostPacDis = min(ghostPacDis,manhattanDistance(ghostPos,currentPacmanPos))

  if ghostPacDis > 0:
    #finalScores -= 10/ghostPacDis
    if ghostState.scaredTimer == 0:
      finalScores -= 10/ghostPacDis
    elif ghostState.scaredTimer > 0:
      finalScores += 100/ghostPacDis


  #when pacman meets food
  remainFood = currentFood.asList()
  if len(remainFood) > 0:
    for foodPos in remainFood + currentCapsule:
      foodPacDis = min(foodPacDis,manhattanDistance(foodPos,currentPacmanPos))
      
    finalScores += 10/foodPacDis

  if currentGameState.isWin():
    finalScores = 99999

  return finalScores



  #util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction

class ContestAgent(MultiAgentSearchAgent):
  """
    Your agent for the mini-contest
  """

  def getAction(self, gameState):
    """
      Returns an action.  You can use any method you want and search to any depth you want.
      Just remember that the mini-contest is timed, so you have to trade off speed and computation.

      Ghosts don't behave randomly anymore, but they aren't perfect either -- they'll usually
      just make a beeline straight towards Pacman (or away from him if they're scared!)
    """
    "*** YOUR CODE HERE ***"
    #return self.alphaBeta_value(gameState,0,float('-inf'),float('inf'))
    
    return self.expectimax_value(gameState,0)

  def expectimax_value(self, state, depthValue):
      agentIndex = depthValue % state.getNumAgents()

      if state.isWin() or state.isLose():
        return self.mini_contest(state)

      if depthValue == self.depth * state.getNumAgents():
        return self.mini_contest(state)

      if agentIndex > 0:
        return self.exp_value(state,depthValue)
      else:
        return self.max_value(state,depthValue)

      return 'None'


  def exp_value(self, state, depthValue):
      agentIndex = depthValue % state.getNumAgents()

      value = 0
      nextAction = 'None'

      legalActions = state.getLegalActions(agentIndex)
      if legalActions == None:
        return self.mini_contest(state)

      p = 1.0/len(legalActions)

      for actions in legalActions:
        if actions == Directions.STOP:
          continue
        successor = state.generateSuccessor(agentIndex,actions)
        tempValue = self.expectimax_value(successor,depthValue + 1)

        value += p * tempValue
        nextAction = actions

      return value



  def max_value(self, state, depthValue):
      value = float('-inf')
      nextAction = 'None'

      legalActions = state.getLegalActions(0)
      if legalActions == None:
        return self.mini_contest(state)

      for actions in legalActions:
        if actions == Directions.STOP:
          continue
        successor = state.generatePacmanSuccessor(actions)
        tempValue = self.expectimax_value(successor,depthValue + 1)

        if tempValue > value:
          value = max(tempValue,value)
          nextAction = actions


      if depthValue == 0:
        return nextAction
      else:
        return value  
  """
  def alphaBeta_value(self, state, depthValue,alpha,beta):
      agentIndex = depthValue % state.getNumAgents()

      if state.isWin() or state.isLose():
        return self.mini_contest(state)

      if depthValue == self.depth * state.getNumAgents():
        return self.mini_contest(state)

      if agentIndex > 0:
        return self.min_value(state,depthValue,alpha,beta)
      else:
        return self.max_value(state,depthValue,alpha,beta)

      return 'None'


  def min_value(self, state, depthValue,alpha,beta):
      agentIndex = depthValue % state.getNumAgents()

      value = float('inf')
      nextAction = 'None'

      legalActions = state.getLegalActions(agentIndex)
      if legalActions == None:
        return self.mini_contest(state)

      for actions in legalActions:
        if actions == Directions.STOP:
          continue
        successor = state.generateSuccessor(agentIndex,actions)
        tempValue = self.alphaBeta_value(successor,depthValue + 1,alpha,beta)

        if tempValue < value:
          value = min(tempValue,value)
          nextAction = actions

        if value <= alpha:
          return value

        beta = min(beta,value)

      return value



  def max_value(self, state, depthValue,alpha,beta):
      value = float('-inf')
      nextAction = 'None'

      legalActions = state.getLegalActions(0)
      if legalActions == None:
        return self.mini_contest(state)

      for actions in legalActions:
        if actions == Directions.STOP:
          continue;
        successor = state.generatePacmanSuccessor(actions)
        tempValue = self.alphaBeta_value(successor,depthValue + 1,alpha,beta)

        if tempValue > value:
          value = max(tempValue,value)
          nextAction = actions

        if value >= beta:
          return value

        alpha = max(alpha,value)


      if depthValue == 0:
        return nextAction
      else:
        return value
  """

  def mini_contest(self, currentGameState):
    #successorGameState = currentGameState.generatePacmanSuccessor(action)
    newPacmanPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    foodCount = currentGameState.getNumFood()
    newGhostStates = currentGameState.getGhostStates()
    newCapsule = currentGameState.getCapsules()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
    ghosts =currentGameState.getGhostPositions()
    finalScores = currentGameState.getScore()
    foodPacDis = 99999
    ghostPacDis = 99999
    closest_scaredGhost = 99999

    #when pacman meets ghost, it will lose points
    for ghostStates in newGhostStates:
      ghostPos = ghostStates.getPosition()
      ghostPacDis = min(ghostPacDis,manhattanDistance(ghostPos,newPacmanPos))

    if ghostPacDis == 0:
      finalScores = -99999

    #print ghostPacDis
    if ghostPacDis > 0:
      if ghostState.scaredTimer == 0:
        finalScores -= 10/ghostPacDis
      if ghostState.scaredTimer > 0:
        finalScores += 100/ghostPacDis

    finalScores += ghostState.scaredTimer*0.6
    #print ghostState.scaredTimer

    if newPacmanPos in newCapsule:
      finalScores += 99999

    #when pacman meets food
    remainFood = newFood.asList()

    if len(remainFood) > 0:
      for foodPos in remainFood + newCapsule:
        foodPacDis = min(foodPacDis,manhattanDistance(foodPos,newPacmanPos))  
      finalScores += 10/foodPacDis - len(newCapsule)
    elif len(remainFood) == 0:
      finalScores = 99999



    if currentGameState.isWin():
      finalScores = 99999

    if currentGameState.isLose():
      finalScores = -99999

    return finalScores


    #util.raiseNotDefined()

