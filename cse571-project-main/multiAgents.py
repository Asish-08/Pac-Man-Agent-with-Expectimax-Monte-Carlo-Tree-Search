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


from util import manhattanDistance
from game import Directions

import random, util
import math
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
 
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        positionScore = successorGameState.getScore()
       
        #ghostDist = []
        ghostAlert = 0
        for ghostPos,scaredTime in zip(newGhostStates, newScaredTimes):
            dist = manhattanDistance(ghostPos.getPosition(), newPos)
            #ghostDist.append(dist)     
            if scaredTime > 0:
                ghostAlert -= 1/(dist+1)
            else:
                if dist < 3:
                    ghostAlert += 1/(dist+1)

        foodList = newFood.asList()
        #foodDist = []
        closestFood = 100000
        for foodPos in foodList:
            dist = manhattanDistance(foodPos, newPos)
            if dist < closestFood:
                closestFood = dist

        foodScale = 10
        ghostScale = -40

        positionScore += foodScale/closestFood + ghostScale*ghostAlert

        return positionScore

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

        def minimaxDecision(state, depth, index):
            return maxValue(state, depth, index)[1]
  
        
        def maxValue(state, depth, index):
            if depth == 0 or state.isWin() or state.isLose():
                return self.evaluationFunction(state), None
            v = float("-inf")
            maxAction = None
            a = state.getLegalActions(index)
            for action in a:
                successor = state.generateSuccessor(index, action)
                actionVal, _ = minValue(successor, depth, 1)
                if actionVal > v:
                    v, maxAction = actionVal, action

            return v, maxAction
        
        def minValue(state, depth, index):
            if depth == 0 or state.isWin() or state.isLose():
                return self.evaluationFunction(state), None
            
            v = float("inf")
            minAction = None
            a = state.getLegalActions(index)
            
            moreGhost = False
            if state.getNumAgents() > index + 1:
                moreGhost = True

            for action in a:
                successor = state.generateSuccessor(index, action)
                actionVal = 0
                if moreGhost:
                    actionVal, _ = minValue(successor, depth, index+1)
                else:
                    actionVal, _ = maxValue(successor, depth-1, 0)
                if actionVal < v:
                    v, minAction = actionVal, action
            
            return v, minAction


        return minimaxDecision(gameState, self.depth, self.index)
        util.raiseNotDefined()

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"

        def alphaBetaSearch(state, depth, index):
            alpha = float("-inf")
            beta = float("inf")
            return maxValue(state, depth, index, alpha, beta)[1]
        
        def maxValue(state, depth, index, alpha, beta):
            if depth == 0 or state.isWin() or state.isLose():
                return self.evaluationFunction(state), 
        
            v = float("-inf")
            a = state.getLegalActions(index)
            for action in a:
                successor = state.generateSuccessor(index, action)
                nextIndex = (index + 1) % state.getNumAgents()
                
                actionVal = 0
                if nextIndex == 0:  
                    actionVal = maxValue(successor, depth-1, nextIndex, alpha, beta)[0]
                else:
                    actionVal = minValue(successor, depth, nextIndex, alpha, beta)[0]
                    
                if actionVal > v:
                    v, maxAction = actionVal, action
                if v > beta:
                    return v, maxAction
                alpha = max(alpha, v)

            return v, maxAction
        
        def minValue(state, depth, index, alpha, beta):
            if depth == 0 or state.isWin() or state.isLose():
                return self.evaluationFunction(state), None
            
            v = float("inf")
            minAction = None
            a = state.getLegalActions(index)
            for action in a:
                successor = state.generateSuccessor(index, action)
                nextIndex = (index + 1) % state.getNumAgents()
                
                actionVal = 0
                if nextIndex == 0: 
                    actionVal = maxValue(successor, depth-1, nextIndex, alpha, beta)[0]
                else:
                    actionVal = minValue(successor, depth, nextIndex, alpha, beta)[0]
                
                if actionVal < v:
                    v, minAction = actionVal, action
                if v < alpha:
                    return v, minAction
                beta = min(beta, v)
            
            return v, minAction


        return alphaBetaSearch(gameState, self.depth, self.index)
        util.raiseNotDefined()

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

        def expectiMaxSearch(state, depth, index):
            return maxValue(state, depth, index)[1]
        
        def maxValue(state, depth, index):
            if depth == 0 or state.isWin() or state.isLose():
                return self.evaluationFunction(state), None
            v = float("-inf")
            maxAction = None
            a = state.getLegalActions(index)
            for action in a:
                successor = state.generateSuccessor(index, action)
                actionVal, _ = expectiValue(successor, depth, 1)
                if actionVal > v:
                    v, maxAction = actionVal, action

            return v, maxAction
        
        def expectiValue(state, depth, index):
            if depth == 0 or state.isWin() or state.isLose():
                return self.evaluationFunction(state), None
            
            v = 0
            expectiAction = None
            a = state.getLegalActions(index)

            if not a:  # No legal actions, return evaluated score
                return self.evaluationFunction(state), None

            moreGhost = False
            if state.getNumAgents() > index + 1:
                moreGhost = True

            p = 1/len(a)

            for action in a:
                successor = state.generateSuccessor(index, action)
                actionVal = 0
                if moreGhost:
                    actionVal, _ = expectiValue(successor, depth, index+1)
                else:
                    actionVal, _ = maxValue(successor, depth-1, 0)
                
                v += p*actionVal
            
            return v, expectiAction



        return expectiMaxSearch(gameState, self.depth, self.index)
        util.raiseNotDefined()

class MonteCarloTreeImprovedSearchAgent(MultiAgentSearchAgent):
    class MCTSNode:
        def __init__(self, game_state, move=None, parent=None, agentIndex=0):
            self.game_state = game_state
            self.move = move
            self.parent = parent
            self.children = []
            self.wins = 0
            self.visits = 0
            self.untried_actions = game_state.getLegalActions(agentIndex)
            self.is_terminal = game_state.isWin() or game_state.isLose()
            self.agentIndex = agentIndex
            self.pacman_direction = None
            self.past_positions = {} 
            self.action_history = []  
            self.ACTION_HISTORY_SIZE = 3  

        def uct_select_child(self, exploration_strength=math.sqrt(1)):
            if not self.children:
                return None

            log_parent_visits = math.log(self.visits)
            uct_values = [(child, (child.wins / child.visits) + exploration_strength * math.sqrt((2 * log_parent_visits) / child.visits)) for child in self.children]
            
            max_uct_value = max(uct_values, key=lambda x: x[1])[1]

            best_children = [child for child, uct_value in uct_values if uct_value == max_uct_value]
            
            return random.choice(best_children) if best_children else None

    def getAction(self, gameState):
        rootNode = self.MCTSNode(gameState, agentIndex=0)
        iter = 100
        for _ in range(iter):  
            node = self.select_node(rootNode)
            if node is not None:  
                score = self.simulate(node.game_state)
                self.backpropagate(node, score)
        bestAction = max(rootNode.children, key=lambda x: x.visits).move
        return bestAction
    
    def backpropagate(self, node, score):
        while node is not None:  
            node.visits += 1  
            node.wins += score  
            node = node.parent  

    def adjust_depth_and_strategy(self, closestFood, food_in_vicinity, wall_in_vicinity, legal_actions_count):
        max_depth = 6  
        ifwhile_stmt = '----------Default--------'
        
        if closestFood <= 2:
            if wall_in_vicinity != 1:
                max_depth = 2  
            else:
                max_depth = 1  
        elif food_in_vicinity != 0:
            if closestFood > 10:
                max_depth += 4  
            else:
                max_depth += 1  
        elif 5 <= closestFood <= 9:
            max_depth += (11 - closestFood) if closestFood > 7 else 2
            if legal_actions_count <= 3:  
                max_depth += 3
        elif 10 <= closestFood <= 12 and food_in_vicinity == 0:
            max_depth += 7  
        elif closestFood >= 13:
            max_depth = 18  
        
        if (wall_in_vicinity == 2 or 5 < closestFood <= 10) and legal_actions_count > 2:
            max_depth += 3  
        
        elif wall_in_vicinity == 1 and legal_actions_count >= 3 and food_in_vicinity == 0:
            max_depth += 7  
        
        return max_depth, ifwhile_stmt


    def simulate(self, game_state):
        current_state = game_state
        depth = 0   
        newPos = current_state.getPacmanPosition()
        newFood = current_state.getFood()
        foodList = newFood.asList()
        foodDist = [manhattanDistance(foodPos, newPos) for foodPos in foodList]
        closestFood = min(foodDist, default=float('inf'))
        vicinity_radius = 2
        food_in_vicinity = sum(manhattanDistance(newPos, food) <= vicinity_radius for food in foodList)
        wall_in_vicinity = sum(manhattanDistance(newPos, wall) <= (vicinity_radius-1) for wall in current_state.getWalls().asList())

        max_depth = 20
        legal_actions = current_state.getLegalActions(0)
        legal_actions_count = len(legal_actions)

        max_depth, ifwhile_stmt = self.adjust_depth_and_strategy(closestFood, food_in_vicinity, wall_in_vicinity, legal_actions_count)

        epsilon = 0.3  

        while not current_state.isWin() and not current_state.isLose() and depth < max_depth:
            possibleActions = current_state.getLegalActions(0)
            if not possibleActions:
                break

            if wall_in_vicinity > 0:
                if 'Stop' in possibleActions and len(possibleActions) > 1:
                    possibleActions.remove('Stop')

            best_score = -float('inf')
            best_actions = [] 

            newPos = current_state.getPacmanPosition()
            newFood = current_state.getFood()
            foodList = newFood.asList()
            closestFood = min([manhattanDistance(foodPos, newPos) for foodPos in foodList], default=float('inf'))

            if closestFood >= 3 and closestFood <= 10:
                epsilon += 0.01
            elif closestFood > 10 and food_in_vicinity == 0:
                epsilon *= 1.001     

            for action in possibleActions:
                successor_state = current_state.generateSuccessor(0, action)
                score = self.evaluationFunction(successor_state)
                newFood = successor_state.getFood()
                foodList = newFood.asList()
                closestFood = min([manhattanDistance(foodPos, newPos) for foodPos in foodList], default=float('inf'))

                ghost_states = successor_state.getGhostStates()
                ghost_threat = False
                for ghost_state in ghost_states:
                    if manhattanDistance(newPos, ghost_state.getPosition()) <= 2 and ghost_state.scaredTimer == 0:
                        ghost_threat = True
                        continue  

                successor_state = current_state.generateSuccessor(0, action)
                score = self.evaluationFunction(successor_state)

                score -=  10 * closestFood if closestFood < 10 else 15 * closestFood

                if not ghost_threat:
                    if random.random() < epsilon:
                        best_actions.append(action)
                    else:
                        if score > best_score:
                            best_score = score
                            best_actions = [action]
                        elif score == best_score:
                            best_actions.append(action)

            if best_actions:
                action = random.choice(best_actions)
                current_state = current_state.generateSuccessor(0, action)
                depth += 1
            else:
                break

        return self.evaluationFunction(current_state)

    def select_node(self, node):
        while not node.is_terminal:
            if node.untried_actions:
                return self.expand_node(node)
            else:
                node = node.uct_select_child()
        return node

    def expand_node(self, node):
        action = node.untried_actions.pop(random.randint(0, len(node.untried_actions) - 1))
        next_state = node.game_state.generateSuccessor(node.agentIndex, action)
        if next_state is None:
            return None  
        child_node = self.MCTSNode(next_state, action, node, node.agentIndex)
        node.children.append(child_node)
        return child_node

    def calculate_cluster_bonus(self, food_positions):
        cluster_radius = 2  
        max_cluster_bonus = 0

        for food_pos in food_positions:
            cluster_size = 0
            for other_food_pos in food_positions:
                if manhattanDistance(food_pos, other_food_pos) <= cluster_radius:
                    cluster_size += 1

            max_cluster_bonus = max(max_cluster_bonus, cluster_size)

        return max_cluster_bonus * 5  

    def evaluationFunction(self, currentGameState):
        successorGameState = currentGameState

        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        positionScore = successorGameState.getScore()

        min_ghost_distance = float('inf')
        for ghostState, scaredTime in zip(newGhostStates, newScaredTimes):
            dist = manhattanDistance(ghostState.getPosition(), newPos)
            min_ghost_distance = min(min_ghost_distance, dist)

            if scaredTime == 0 and dist <= 1:  
                return -float('inf')  

        ghostAlert = 0
        for ghostState, scaredTime in zip(newGhostStates, newScaredTimes):
            dist = manhattanDistance(ghostState.getPosition(), newPos)
            if scaredTime > 0:
                ghostAlert -= 0.01 / (dist + 1)  
            else:
                if dist < 3:  
                    ghostAlert += 3 / (dist + 1)
                elif dist < 5:
                    ghostAlert += 1 / (dist + 1)

        foodList = newFood.asList()
        closestFood = min([manhattanDistance(foodPos, newPos) for foodPos in foodList], default=float('inf'))

        foodScale = 40
        ghostScale = -40  
        positionScore += foodScale / closestFood + ghostScale * ghostAlert

        food_positions = successorGameState.getFood().asList() 
        closest_food_dist = min(manhattanDistance(newPos, food) for food in food_positions)
        cluster_bonus = self.calculate_cluster_bonus(food_positions)  

        positionScore -= 15 / closest_food_dist  
        positionScore += cluster_bonus

        return positionScore
       
class MonteCarloTreeRandomSearchAgent(MultiAgentSearchAgent):
    class MCTSNode:
        def __init__(self, game_state, move=None, parent=None, agentIndex=0):
            self.game_state = game_state
            self.move = move
            self.parent = parent
            self.children = []
            self.wins = 0
            self.visits = 0
            self.untried_actions = game_state.getLegalActions(agentIndex)  
            self.is_terminal = game_state.isWin() or game_state.isLose()
            self.agentIndex = agentIndex  
    
    def select_node(self, node):
        while not node.is_terminal:
            if node.untried_actions:
                return self.expand_node(node)
            else:
                node = self.best_child(node)
        return node

    def expand_node(self, node):
        action = random.choice(node.untried_actions)
        next_state = node.game_state.generateSuccessor(node.agentIndex, action)
        child_node = self.MCTSNode(next_state, action, parent=node, agentIndex=node.agentIndex)
        node.children.append(child_node)
        node.untried_actions.remove(action)
        return child_node

    def simulate(self, game_state):
        current_state = game_state
        depth = 0
        max_depth = 10  
        
        while not current_state.isWin() and not current_state.isLose() and depth < max_depth:
            possible_moves = current_state.getLegalActions(0)  
            if not possible_moves:
                break
            chosen_move = random.choice(possible_moves)
            current_state = current_state.generateSuccessor(0, chosen_move)
            depth += 1
        
        return current_state

    def backpropagate(self, node, result_game_state):
        while node is not None:
            node.visits += 1
            if result_game_state.isWin():
                node.wins += 1
            elif result_game_state.isLose():
                node.wins -= 1  
            node = node.parent

    def best_child(self, node):
        best_score = -float('inf')
        best_children = []
        for child in node.children:
            win_ratio = child.wins / child.visits
            if win_ratio > best_score:
                best_children = [child]
                best_score = win_ratio
            elif win_ratio == best_score:
                best_children.append(child)
        return random.choice(best_children) if best_children else None

    def getAction(self, gameState):
        iterations = 50  
        root_node = self.MCTSNode(gameState, agentIndex=0)
        for _ in range(iterations):
            selected_node = self.select_node(root_node)
            simulated_state = self.simulate(selected_node.game_state.deepCopy())
            self.backpropagate(selected_node, simulated_state)
        best_move = self.best_child(root_node).move
        return best_move
    
#class MonteCarloTreeSearchAgent(MultiAgentSearchAgent):