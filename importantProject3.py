import numpy as np
import math
from numpy import random
from numpy.linalg import multi_dot
import os
import time
import json
import multiprocessing
from datetime import datetime
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)

class ship:
    def __init__(self, k, alpha, bot_mode):
        # Dimensions of the Maze Matrix
        self.d = 35
        self.k = k
        self.alpha = alpha
        # At the start, the board is full with closed cell: values of 1
        # Note for the code: 1 indicates a wall, and 0 indicates a space that an entity could possibly on
        self.board = np.full((self.d, self.d), 1)
        self.open_cells = []
        self.open_cells_dictionary = {}  # Make a dictionary to be able to access a given index based on the (row, column)
        self.alien_locations = []
        self.crew_locations = []
        self.init_board()  # create a numpy dxd array
        self.bot_mode = bot_mode

        #Sets the number of aliens and crew on the board based off of the bot mode
        k_aliens = 1
        k_crew = 1
        if self.bot_mode <= 1 and self.bot_mode <= 2:
            k_aliens = 1
            k_crew = 1
        elif self.bot_mode >= 3 and self.bot_mode <= 5:
            k_aliens = 1
            k_crew = 2
        elif self.bot_mode >= 6 and self.bot_mode <= 8:
            k_aliens = 2
            k_crew = 2

        # randomly generates the bot start location, then removes it from open cells
        i = random.randint(len(self.open_cells))
        self.bot_start_location = (self.open_cells[i][0], self.open_cells[i][1])
        self.open_cells.pop(i)
        startRow = self.bot_start_location[0]
        startCol = self.bot_start_location[1]

        # Terminates code early if there is no viable alien placement so k chosen for project < 17 based on math
        if self.k >= 34 - startRow and self.k >= startRow - 0:
            if self.k >= 34 - startCol and self.k >= startCol - 0:
                print("Error: Alien Sensor Detection Too Large, nowhere to place aliens")
                print("(1) Please choose a smaller k < 34")
                print("(2) Or rerun until you get a favorable start, less likely the closer k is to 34")
                raise SystemExit('error in code want to exit')

        # randomly generates the crew locations
        for n in range(k_crew):
            i = random.randint(len(self.open_cells))
            goal = (self.open_cells[i][0], self.open_cells[i][1])
            self.crew_locations.append(goal)
            # Makes it so crew members can't spawn on each other
            self.open_cells.pop(i)

        # Adds crew locations and bot spawn back to list of open_cells
        for crew in self.crew_locations:
            self.open_cells.append(crew)
        self.open_cells.append(self.bot_start_location)

        # Loop over open cells to get indexing for a (row, col)
        for index, cell in enumerate(self.open_cells):
            self.open_cells_dictionary[(cell)] = index

        # If there are more aliens then open cells, it will return failure
        if k_aliens >= len(self.open_cells):
            k_aliens = len(self.open_cells)

        # Finds the list of open cells that are outside the 2k+1, 2k + 1 square
        self.validCellsForAliens = self.pruneChoices()

        # Places aliens outside of the 2k + 1, 2k + 1 square
        while k_aliens > 0:
            i = random.randint(len(self.validCellsForAliens))
            self.alien_locations.append(self.validCellsForAliens[i])
            self.validCellsForAliens.pop(i)
            k_aliens -= 1

        #Places the alien cell back into validAlien Cells
        for alien in self.alien_locations:
            self.validCellsForAliens.append(alien)

        # init bot mode
        if bot_mode == 1:
            self.bot = botBase(self.bot_start_location, self.alpha, self.k, 1, 1, self)
        elif bot_mode == 2:
            self.bot = bot2(self.bot_start_location, self.alpha, self.k, 1, 1, self)
        elif bot_mode == 3:
            self.bot = botBase(self.bot_start_location, self.alpha, self.k, 1, 2, self)
        elif bot_mode == 4:
            self.bot = bot4(self.bot_start_location, self.alpha, self.k, 1, 2, self)
        elif bot_mode == 5:
            self.bot = bot5(self.bot_start_location, self.alpha, self.k, 1, 2, self)
        elif bot_mode == 6:
            self.bot = botBase(self.bot_start_location, self.alpha, self.k, 2, 2, self)
        elif bot_mode == 7:
            self.bot = bot7(self.bot_start_location, self.alpha, self.k, 2, 2, self)
        elif bot_mode == 8:
            self.bot = bot8(self.bot_start_location, self.alpha, self.k, 2, 2, self)

    def init_board(self):
        # Opens an initial cell at random
        rowStart = random.randint(self.d)
        rowEnd = random.randint(self.d)
        self.board[rowStart, rowEnd] = 0

        # Iteratively finds blocked cells with one open neighbor and randomly opens one of the blocked cells
        valid_rows, valid_cols = 1, 1
        while valid_rows >= 1:
            valid_cells = [(-1, -1)]
            for row in range(0, self.d):
                for col in range(0, self.d):
                    # If the cell is open, we move on to the next iteration
                    if self.board[row][col] == 0:
                        continue
                    # The cell is now blocked and checks if there is one open neighbor
                    if self.countOpenNeighbors(row, col) == 1:
                        valid_cells.append((row, col))

            # We found no blocked cells with exactly one open neighbor so we exit
            valid_count = len(valid_cells)
            if valid_count == 1:
                break

            # randomizes which blocked cell to open up
            toOpen_row, toOpen_col = valid_cells[random.randint(1, valid_count)]
            self.board[toOpen_row, toOpen_col] = 0

        # finds all dead cells - on open cell with exactly on open neighbor
        valid_cells = [(-1, -1)]
        for row in range(0, self.d):
            for col in range(0, self.d):
                # If the cell is blocked, we move onto the next iteration
                if self.board[row][col] == 1:
                    continue
                # The cell is now open and has exactly one open neighbor
                if self.countOpenNeighbors(row, col) == 1:
                    valid_cells.append((row, col))

        # Randomizes all closed cells to open
        valid_count = len(valid_cells)
        cells_to_open = [(-1, -1)]
        # For an half of the dead cells, choose a random neighbor to open
        for row in range(1, valid_count, 2):
            toOpen_row, toOpen_col = self.chooseClosedNeighbor(valid_cells[row][0], valid_cells[row][1])
            cells_to_open.append((toOpen_row, toOpen_col))

        # If two open cells randomizes the same closed neighbor
        # That is still picking one randomized closed neighbor for each, it just happens to be the same one
        toOpen_count = len(cells_to_open)
        for row in range(1, toOpen_count):
            self.board[cells_to_open[row][0]][cells_to_open[row][1]] = 0

        for x in range(0, self.d):
            for y in range(0, self.d):
                if self.board[x][y] == 0:
                    self.open_cells.append((x, y))

    #Returns the list of open cells outside the 2k+1, 2k+1 square of the bot's initial cell location
    def pruneChoices(self):
        validAlienChoices = []
        for open_cell in self.open_cells:
            if abs(open_cell[0] - self.bot_start_location[0]) > self.k or abs(
                    open_cell[1] - self.bot_start_location[1]) > self.k:
                validAlienChoices.append(open_cell)

        return validAlienChoices

    #Debugging Tool 1
    def print_board(self, bot):
        # Print column headers with appropriate spacing
        print("   ", end=' ')
        for j in range(self.d):
            print(f"{j:<3}", end='')  # Adjust the spacing as needed
        print()
        # Print the ship grid
        for i in range(self.d):
            print(f"{i:<3}", end='')  # Adjust the spacing as needed
            for j in range(self.d):
                if bot.location == (i, j):
                    print(" R ", end='')
                elif (i, j) in self.alien_locations:
                    print(" X ", end='')
                elif (i, j) in self.crew_locations:
                    print(" C ", end='')
                elif self.board[i][j] == 0:
                    print(" _ ", end='')
                elif self.board[i][j] == 1:
                    print(" | ", end='')
            print()
        print("________________________________________________________")

    #Debugging Tool 2
    def print_indicies(self):
        print("   ", end=' ')
        for j in range(self.d):
            print(f"{j:<5}", end='')  # Adjust the spacing as needed
        print()
        # Print the ship grid
        width = 3
        for i in range(self.d):
            print(f"{i:<4}", end='')  # Adjust the spacing as needed
            for j in range(self.d):
                if self.board[i][j] == 0:
                    print("{:<{width}}".format(self.open_cells_dictionary[(i, j)], width=width), end='  ')
                else:
                    print("{:<{width}}".format('|', width=width), end='  ')
            print()
        print("________________________________________________________")

    # Code to choose a random closed neighbor
    def chooseClosedNeighbor(self, row, col):
        closed_neighbors = np.reshape([-1, -1], (1, 2))
        up = row - 1
        down = row + 1
        left = col - 1
        right = col + 1

        # Adds all possible closed neighbors to a list
        if up >= 0 and self.board[up][col] == 1:
            closed_neighbors = np.append(closed_neighbors, [[up, col]], axis=0)
        if down < self.d and self.board[down][col] == 1:
            closed_neighbors = np.append(closed_neighbors, [[down, col]], axis=0)
        if left >= 0 and self.board[row][left] == 1:
            closed_neighbors = np.append(closed_neighbors, [[row, left]], axis=0)
        if right < self.d and self.board[row][right] == 1:
            closed_neighbors = np.append(closed_neighbors, [[row, right]], axis=0)

        # randomzies which closed neighbor to choose and returns it
        closed_rows, closed_cols = closed_neighbors.shape
        toOpen_row, toOpen_col = closed_neighbors[random.randint(1, closed_rows)]

        return toOpen_row, toOpen_col

    # Method to count the number of open neighbors surrounding a cell
    def countOpenNeighbors(self, row, col):
        count = 0
        up = row - 1
        down = row + 1
        left = col - 1
        right = col + 1
        # Checks if the cells are in bounds and if they are open
        if up >= 0 and self.board[up][col] == 0:
            count = count + 1
        if down < self.d and self.board[down][col] == 0:
            count = count + 1
        if left >= 0 and self.board[row][left] == 0:
            count = count + 1
        if right < self.d and self.board[row][right] == 0:
            count = count + 1

        return count

    def getManhattanDistance(self, row1, col1, row2, col2):
        return abs(row1 - row2) + abs(col1 - col2)

    def withinAlienDetectionSquare(self, row1, col1, row2, col2):
        if abs(row1 - row2) <= self.k and abs(col1 - col2) <= self.k:
            return True
        else:
            return False

    # Code to find open neighbors, including or excluding alien spaces based on need
    def getOpenNeighbors(self, pos, ignore_aliens=False):
        neighbors = [(pos[0] - 1, pos[1]), (pos[0] + 1, pos[1]), (pos[0], pos[1] - 1), (pos[0], pos[1] + 1)]
        openNeighbors = []
        # Checks if the cell is in bounds, the cell allows for an entity
        for n in neighbors:
            if self.isInBounds(n) and self.board[n[0]][n[1]] == 0:
                # Checks if we have the ignore_aliens flag active
                # Otherwise, if the cell does not contain an alien, it is an open neighbor and we append it
                if ignore_aliens or (n not in self.alien_locations):
                    openNeighbors.append(n)

        return openNeighbors

    # Method to check if an index is within bounds
    def isInBounds(self, n):
        return 0 <= n[0] < self.d and 0 <= n[1] < self.d

    # Simulate Beep based off of manhattan distance of crew
    def simulateBeep(self):
        u = random.random()
        beepCount = 0
        for crew in self.crew_locations:
            d = self.getManhattanDistance(*crew, *self.bot.location)
            p = 1.0 / math.exp(self.alpha * (d - 1.0))
            if u <= p:
                beepCount = beepCount + 1
            else:
                continue
        if beepCount > 0:
            return True
        return False

    # Detects if there is at least one alien within the detection square
    def alienSensor(self):
        for alien in self.alien_locations:
            if self.withinAlienDetectionSquare(*alien, *self.bot.location):
                return True
        return False

    def update(self):
        #bot updates its position during each update
        self.bot.update(self)

        #The ship checks if it walked in an alien on the final step
        if self.bot.location in self.alien_locations:
            return -1
        # If bot rescues a crew member it teleports it away
        if self.bot.location in self.crew_locations:
            self.crew_locations.remove(self.bot.location)
        # If the bot rescued all crew successfully it returns a success
        if len(self.crew_locations) == 0:
            return 1

        index = 0
        # Note: The TA allowed us to drop the randomized ordering of which aliens move in the project
        # Moves each alien randomly to an open cell
        for alien in self.alien_locations:
            options = self.getOpenNeighbors(alien)
            # If the alien is not blocked, it can move
            if len(options) != 0:
                choice = random.randint(len(options))
                self.alien_locations[index] = options[choice]
            index = index + 1

        # Checks if an alien has moved onto the bot
        if self.bot.location in self.alien_locations:
            return -1

        #Collects observation per timestep
        beep_detected = self.simulateBeep()
        senseAlien = self.alienSensor()


        #Updates belief of both crew and alien belief values to use for next timestep

        self.bot.updateCrewBelief(beep_detected, self)
        self.bot.updateAlienBelief(senseAlien, self)

        return 0
        # return -1 for fail, 1 for goal, 0 for continue


class Bot:
    def __init__(self, loc, alpha, k, n_aliens, n_crew, ship):
        # print("base init")

        self.alpha = alpha
        self.location = loc
        self.k = k
        self.zero = 0.000000000001  # Anything less than this we assume to be a probability of 0

        # init Alien Probability Matrix and Crew Probability Vectors
        self.alienP = self.initAlienProb(n_aliens, ship)
        self.crewP = self.initCrewProb(n_crew, ship)

        #mSize is the size of the alien and crew belief for the cases there is one alien or crew
        self.mSize = len(ship.open_cells)
        self.alienObs = np.identity(self.mSize)
        self.crewObs = np.identity(self.mSize)
        self.trans = np.identity(self.mSize)
        self.initTransitionMatrix(ship)

    # initial distributions shouldn't change between bots
    def initAlienProb(self, n_aliens, ship):
        aProb = []
        # Valid Cells For Aliens is all the possible locations aliens can spawn at
        initProb = 1.0 / len(ship.validCellsForAliens)

        # Otherwise if it is outside that list, the probability is set to 0
        for cell in ship.open_cells:
            if cell in ship.validCellsForAliens:
                aProb.append(initProb)
            else:
                aProb.append(0)

        return np.asarray(aProb)

    def initCrewProb(self, n_crew, ship):
        cProb = []
        #The crew can spawn at every open_cell except the bot's start location
        initProb = 1.0 / (len(ship.open_cells) - 1.0)

        #If it is the bot's start location set only that cell to a P of 0
        for i in range(len(ship.open_cells)):
            if self.isBotLocation(ship.open_cells[i]):
                cProb.append(0)
            else:
                cProb.append(initProb)

        return np.asarray(cProb)

    #Initializes Transition Matrix based on one alien filter on the overleaf document
    def initTransitionMatrix(self, ship):
        for i in range(self.mSize):
            for j in range(self.mSize):
                if i == j:
                    self.trans[i][j] = 0
                elif ship.getManhattanDistance(*ship.open_cells[i], *ship.open_cells[j]) == 1:
                    self.trans[i][j] = 1.0 / ship.countOpenNeighbors(*ship.open_cells[j])
                else:
                    self.trans[i][j] = 0

    #Updates observation matrix based on the overleaf document
    #If a cell is within the alien detection square and there is a beep set it to 1, no beep set it to 0
    #If a cell is outside the alien detection square and there is a beep set it to 0, no beep set it to 1
    def updateAlienObservationMatrix(self, alienDetected, ship):
        for i in range(self.mSize):
            if ship.withinAlienDetectionSquare(*ship.open_cells[i], *self.location):
                if alienDetected:
                    self.alienObs[i][i] = 1.0
                else:
                    self.alienObs[i][i] = 0.0
            else:
                if alienDetected:
                    self.alienObs[i][i] = 0.0
                else:
                    self.alienObs[i][i] = 1.0

        #Makes sure to set the bot's location to zero as an alien cannot be there
        index = ship.open_cells_dictionary[self.location]
        self.alienObs[index][index] = 0

    #Debugging Tool 3

    def print_alien_probabilities(self, ship):
        matrix = np.zeros((ship.d, ship.d))
        for index, prob in enumerate(self.alienP):
            loc = ship.open_cells[index]
            matrix[loc[0]][loc[1]] = prob

        print("    ", end=' ')
        for j in range(ship.d):
            print(f"{j:<9}", end='')  # Adjust the spacing as needed
        print()
        # Print the ship grid
        for i in range(ship.d):
            print(f"{i:<4}", end='')  # Adjust the spacing as needed
            for j in range(ship.d):
                if ship.board[i][j] == 0:
                    print(' %.5f ' % matrix[i][j], end='')
                else:
                    print(' |       ', end='')
            print()
        print("________________________________________________________")


    #Debugging Tool 4

    def print_crew_probabilities(self, ship):
        matrix = np.zeros((ship.d, ship.d))
        for index, prob in enumerate(self.crewP):
            loc = ship.open_cells[index]
            matrix[loc[0]][loc[1]] = prob


        print("    ", end=' ')
        for j in range(ship.d):
            print(f"{j:<9}", end='')  # Adjust the spacing as needed
        print()
        # Print the ship grid
        for i in range(ship.d):
            print(f"{i:<4}", end='')  # Adjust the spacing as needed
            for j in range(ship.d):
                if ship.board[i][j] == 0:
                    print(' %.5f ' % matrix[i][j], end='')
                else:
                    print(' |       ', end='')
            print()
        print("________________________________________________________")


        print("    ", end=' ')
        for j in range(ship.d):
            print(f"{j:<9}", end='')  # Adjust the spacing as needed
        print()
        # Print the ship grid
        for i in range(ship.d):
            print(f"{i:<4}", end='')  # Adjust the spacing as needed
            for j in range(ship.d):
                if ship.board[i][j] == 0:
                    print(' %.5f ' % matrix[i][j], end='')
                else:
                    print(' |       ', end='')
            print()
        print("________________________________________________________")

    #Checks if a cell is the same location as the bot
    def isBotLocation(self, cell):
        return cell[0] == self.location[0] and cell[1] == self.location[1]

    #Calculates the diagonal value in the crew observation matrix based on Obeep on the overlea
    def calcCrewDiagonalValue(self, beep, cell, ship):
        d = ship.getManhattanDistance(*self.location, *cell)
        if beep:
            return 1.0 / math.exp(self.alpha * (d - 1.0))
        else:
            return 1.0 - 1.0 / math.exp(self.alpha * (d - 1.0))

    #Updates the CrewObservation matrix, and makes sure to set the bot's location to 0
    def updateCrewObservationMatrix(self, beep, ship):
        #  print("--------")
        for i in range(self.mSize):
            if self.isBotLocation(ship.open_cells[i]):
                self.crewObs[i][i] = 0.0
            else:
                self.crewObs[i][i] = self.calcCrewDiagonalValue(beep, ship.open_cells[i], ship)


    #Normalizes the probability for the bot

    def normalize(self, probs):
        sum = np.sum(probs)
        probs = probs * 1.0 / sum
        return probs

    # If no_chance_aliens we only allow moves to cells with alien probability 0
    def generate_actions(self, loc, ship, no_chance_aliens=False):
        neighbors = ship.getOpenNeighbors(loc, True)
        if not no_chance_aliens:
            return neighbors
        options = []
        zero_options = []
        for neighbor in neighbors:
            if self.alienP[ship.open_cells_dictionary[neighbor]] == 0:
                zero_options.append(neighbor)
            options.append(neighbor)
        if len(zero_options) > 0:  # If there is a completely safe cell we return those cells to traverse from
            return zero_options
        else:  # If no cells are completely safe let sort them and take the least probable alien cell
            options.sort(key=lambda neighbor: self.sort_key(neighbor, ship))
            return [options[0]]

    def sort_key(self, neighbor, ship):
        return self.alienP[ship.open_cells_dictionary[neighbor]]


    # Gets target in row, col by finding the largest probability option. If multiple of the same put them in a list a randomly select one

    def get_target_location(self, ship):
        max_prob = -1
        options = []
        for index, prob in enumerate(self.crewP):
            if prob > max_prob:
                options = [ship.open_cells[index]]
                max_prob = prob
            elif prob == max_prob:
                options.append(ship.open_cells[index])

        return options[random.randint(len(options))]  # return random best option

    def BFS(self, start, target, ship):
        fringe = [start]
        closed = []
        prev = {start: start}
        safeSteps = 1

        while len(fringe) > 0:
            cur = fringe.pop(0)
            if cur == target:
                return prev

            children = []
            if safeSteps > 0:
                children = self.generate_actions(cur, ship, True)
                safeSteps -= 1
            else:
                children = self.generate_actions(cur, ship)

            for child in children:
                # Added a check in addition to the pseudo-code to avoid adding children to the fringe multiple times for runtime concerns
                if child not in closed and child not in fringe:
                    fringe.append(child)
                    prev[child] = cur
            closed.append(cur)
        return None

    #BFS code based on CS520 online notes with a set start and a target
    def get_ideal_path(self, start, ship):
        path = {}
        target = self.get_target_location(ship)

        bfs = self.BFS(start, target, ship)
        if bfs is not None:
            cur = target
            while cur != start:
                if cur not in bfs:
                    return None
                prev = bfs[cur]
                path[prev] = cur
                cur = prev
            return path
        return None

    # Moves to the next cell in the shortest path it found. If there is no path, the bot stays in place
    def update(self, ship):
        self.path = self.get_ideal_path(self.location, ship)
        if self.path is not None and self.location in self.path:
            self.location = self.path[self.location]


    #Updates crew belief, the normalizes it

    def updateCrewBelief(self, beep, ship):
        self.updateCrewObservationMatrix(beep, ship)
        crewUpdate = np.matmul(self.crewObs, self.crewP)
        self.crewP = self.normalize(crewUpdate)

    #Updates alien belief, the normalizes it

    def updateAlienBelief(self, alienDetected, ship):
        self.updateAlienObservationMatrix(alienDetected, ship)
        alienUpdate = multi_dot([self.alienObs, self.trans, self.alienP])
        self.alienP = self.normalize(alienUpdate)


# Also Used for Bot 3 and Bot 6
class botBase(
    Bot):  # code that makes the numbering consistent, and makes it so it only generates the shortest path once
    def __init__(self, loc, alpha, k, n_aliens, n_crew, ship):
        Bot.__init__(self, loc, alpha, k, n_aliens, n_crew, ship)
    pass

class bot2(Bot):
    # Initializes the location, and starting shorting path
    def __init__(self, loc, alpha, k, n_aliens, n_crew, ship):
        Bot.__init__(self, loc, alpha, k, n_aliens, n_crew, ship)


    # Gets target in row, col by finding the largest Utility option.
    # If multiple of the same put them in a list a randomly select one
    def get_target_location(self, ship):
        risk = np.copy(self.alienP)
        for i, val in enumerate(risk):
            if val > 0:
                risk[i] = math.ceil(100 * val)
            else:
                risk[i] = 1
        max_utility = -1
        options = []
        for index, probCrew in enumerate(self.crewP):
            distance = ship.getManhattanDistance(*self.location, *ship.open_cells[index])
            if distance == 0:
                distance = 1
            utility = probCrew / (risk[index] * distance)
            if utility > max_utility:
                options = [ship.open_cells[index]]
                max_utility = utility
            elif utility == max_utility:
                options.append(ship.open_cells[index])

        return options[random.randint(len(options))]  # return random best option



class bot4(Bot):
    def __init__(self, loc, alpha, k, n_aliens, n_crew, ship):
        #Flag for checking if we rescuedOneCrew
        self.rescuedOneCrew = False
        Bot.__init__(self, loc, alpha, k, n_aliens, n_crew, ship)
        #Initialzies crewPair probabilities
        self.crewPairP = self.initCrewPairProb(n_crew, ship)
        self.crewPairObsDiagonal = np.zeros(self.mSize * self.mSize)
        self.crewPairUpdate = np.zeros(self.mSize * self.mSize)

        self.identityDiagonal = np.ones(self.mSize * self.mSize)

    def initCrewPairProb(self, n_crew, ship):
        cPairProb = []

        #Counts the number of possible pair crew locations given that a crew is not on the same cell as the bot

        initProb = 1.0 / (self.mSize * self.mSize - self.mSize - self.mSize + 1.0)

        #If either value in the pair is the same location as the bot the P is set to 0
        for i in range(self.mSize):
            for j in range(self.mSize):
                if self.isBotLocation(ship.open_cells[i]) or self.isBotLocation(ship.open_cells[j]):
                    cPairProb.append(0)
                else:
                    cPairProb.append(initProb)

        return np.asarray(cPairProb)

    def identityDiagonalExceptBot(self, ship):
        #Zeroes out all indexes where the bot is at cell i or the bot is at cell j
        n = self.mSize
        IDiagonalBot = np.ones(n * n)
        locIndex = ship.open_cells_dictionary[self.location]
        for i in range(n):
            IDiagonalBot[i * n + locIndex] = 0
        for j in range(n):
            IDiagonalBot[locIndex * n + j] = 0

        return IDiagonalBot

    #The formula on overLeaf, requires O NoBeep for each update
    def calcONoBeep(self, ship):
        for i in range(self.mSize):
            if self.isBotLocation(ship.open_cells[i]):
                self.crewObs[i][i] = 0
            else:
                self.crewObs[i][i] = self.calcCrewDiagonalValue(False, ship.open_cells[i], ship)

    #Updates the CrewPair ObservationMatrix based on formula on overleaf
    #Since every matrix is a diagonal one for two crew, we represent the diagonals as vectors and operate on those
    def updateCrewPairObservationMatrix(self, beep, ship):
        IDiagonalBot = self.identityDiagonalExceptBot(ship)
        self.calcONoBeep(ship)
        crewObsDiagonal = np.diagonal(self.crewObs)

        kronProduct = np.kron(crewObsDiagonal, crewObsDiagonal)
        if beep:
            self.crewPairObsDiagonal = np.multiply(IDiagonalBot, np.subtract(self.identityDiagonal, kronProduct))
        else:
            self.crewPairObsDiagonal = np.multiply(IDiagonalBot, kronProduct)

    def reverseMarginalization(self):
        # Note: The Probability that crew 1 is at cell k is the same as the probability that crew 2 is at cell k
        # P(C1@k or C2@k) = P(C1@k) + P(C2@k) - P(C1@k and C2@k), and the last term is clearly zero
        # Therefore, the relationship of P(C1@i) vs P(C1@j) is the same as P(C1@i or C2@i) vs P(C1@j or C2@j)
        n = self.mSize
        for i in range(n):
            self.crewP[i] = np.sum(self.crewPairP[i * n: i * n + n])

    def update(self, ship):
        super().update(ship)


    #Overrides updating crew belief of bot 1
    def updateCrewBelief(self, beep, ship):
        #If we rescuedOneCrew after the iteration it teleports away, use the one crew filter
        if self.rescuedOneCrew:
            super().updateCrewBelief(beep, ship)
        #If we just rescued a crew member, get the probabilities from the ith kron component
        #which represents the universe that the other crew is at every cell given
        #that the rescued crew member was at the location i we the bot just entered
        #then uses the one crew filter and sets the rescuedOneCrew to true
        elif len(ship.crew_locations) == 1:
            kronComponentIndex = ship.open_cells_dictionary[self.location]
            n = self.mSize
            for j in range(self.mSize):
                self.crewP[j] = self.crewPairP[kronComponentIndex * n + j]
            self.crewP = self.normalize(self.crewP)
            self.rescuedOneCrew = True
            super().updateCrewBelief(beep, ship)
        #Uses the 2CrewFilter, and same logic of only storing the diagonals from above
        else:
            self.updateCrewPairObservationMatrix(beep, ship)
            self.crewPairUpdate = np.multiply(self.crewPairObsDiagonal, self.crewPairP)
            self.crewPairP = self.normalize(self.crewPairUpdate)

            # Calculates The Probability Crew 1 or Crew 2 is at each cell
            self.reverseMarginalization()


    #updating the alien belief is the same for bot4
    def updateAlienBelief(self, alienDetected, ship):
        super().updateAlienBelief(alienDetected, ship)

#bot 5 is a subclass of bot 4 and only overrides the get_target_location function to use utility
class bot5(bot4):
    def __init__(self, loc, alpha, k, n_aliens, n_crew, ship):
        bot4.__init__(self, loc, alpha, k, n_aliens, n_crew, ship)

    def get_target_location(self, ship):
        risk = np.copy(self.alienP)
        for i, val in enumerate(risk):
            if val > 0:
                risk[i] = math.ceil(100 * val)
            else:
                risk[i] = 1
        max_utility = -1
        options = []
        for index, probCrew in enumerate(self.crewP):
            distance = ship.getManhattanDistance(*self.location, *ship.open_cells[index])
            if distance == 0:
                distance = 1
            utility = probCrew / (risk[index] * distance)
            if utility > max_utility:
                options = [ship.open_cells[index]]
                max_utility = utility
            elif utility == max_utility:
                options.append(ship.open_cells[index])
        return options[random.randint(len(options))]  # return random best option


class bot7(bot4):
    def __init__(self, loc, alpha, k, n_aliens, n_crew, ship):
        bot4.__init__(self, loc, alpha, k, n_aliens, n_crew, ship)
        #After getting the transition matrix for one alien, clones it on top of itself n times
        self.trans = np.reshape(self.trans, (self.mSize, self.mSize, 1))
        self.trans2Alien = np.tile(self.trans, (1, 1, self.mSize))

        #Sets both alien1P and alien2P to the right intitial probabilities
        self.alien1P = self.initAlienProb(n_aliens, ship)
        self.alien2P = self.initAlienProb(n_aliens, ship)

        #Reshapes so tensorDot products operations function properly
        self.alienP = np.reshape(self.alienP, (self.mSize, 1))
        self.alien1P = np.reshape(self.alien1P, (self.mSize, 1))
        self.alien2P = np.reshape(self.alien2P, (self.mSize, 1))

    def init2AlienTransitionMatrix(self, ship):
        # Slice k of the 3d array is the theoretical situation there is an alien at cell k
        for k in range(self.mSize):
            # Covers the cases where i != k and j != k
            # Gets the open cells around cell k, which need probability updates
            jCandidates = ship.getOpenNeighbors(ship.open_cells[k], True)
            for jLoc in jCandidates:
                jLocIndex = ship.open_cells_dictionary[jLoc]

                # Gathers the open cells around cell j excluding the location of the alien
                iCandidates = ship.getOpenNeighbors(jLoc, True)
                iCandidates.remove(ship.open_cells[k])
                trueOpenNeighbors = len(iCandidates)

                # Case: j has no true open neighbors, it's the case where cell j is blocked entirely
                if trueOpenNeighbors == 0:
                    self.trans2Alien[jLocIndex][jLocIndex][k] = 1.0
                # Case: j has at least one open neighbor, so we modify each transition from j to i around it
                else:
                    for iLoc in iCandidates:
                        iLocIndex = ship.open_cells_dictionary[iLoc]
                        self.trans2Alien[iLocIndex][jLocIndex][k] = 1.0 / trueOpenNeighbors

            # Covers the cases where i == k or j == k
            for j in range(self.mSize):
                self.trans2Alien[k][j][k] = 0

            for i in range(self.mSize):
                self.trans2Alien[i][k][k] = 0

    #Compared to one alien filter, if a beep is detected, its either 2 aliens in, 1 alien in 1 out
    #So in this case we can only set the bot location to 0 rather than outside the square
    def updateAlienObservationMatrix(self, alienDetected, ship):
        for i in range(self.mSize):
            if ship.withinAlienDetectionSquare(*ship.open_cells[i], *self.location):
                if alienDetected:
                    self.alienObs[i][i] = 1.0
                else:
                    self.alienObs[i][i] = 0.0
            else:
                if alienDetected:
                    self.alienObs[i][i] = 1.0
                else:
                    self.alienObs[i][i] = 1.0
        index = ship.open_cells_dictionary[self.location]
        self.alienObs[index][index] = 0

    def updateAlienBelief(self, alienDetected, ship):
        #Find observation matrix based on if alien was detected
        self.updateAlienObservationMatrix(alienDetected, ship)

        #Updates the alienP based on the notes
        alien1Update = np.tensordot(self.trans2Alien, np.matmul(self.alien1P, np.transpose(self.alien2P)))
        alien1Update = np.reshape(alien1Update, (self.mSize, 1))

        alien2Update = np.tensordot(self.trans2Alien, np.matmul(self.alien2P, np.transpose(alien1Update)))
        alien2Update = np.reshape(alien2Update, (self.mSize, 1))

        #Multiplies by observation matrix
        alien1Update = np.matmul(self.alienObs, alien1Update)
        alien2Update = np.matmul(self.alienObs, alien2Update)

        # Normalizes the aliens
        self.alien1P = self.normalize(alien1Update)
        self.alien2P = self.normalize(alien2Update)

        # Adds it together, since P(alien 1 @ k and alien 2 @ k) = 0
        self.alienP = np.add(self.alien1P, self.alien2P)

        # To Save Runtime for data collection [self.alienP = self.alienP * 0.5]
        # The final P should be divided by 2 for math accuracy, but the code functions
        # The same, so we have it commented out to save around 1/30th update time per update

#Bot8 is a subclass of bot 7 but overrides get_target_location with the utility function
class bot8(bot7):
    def __init__(self, loc, alpha, k, n_aliens, n_crew, ship):
        bot7.__init__(self, loc, alpha, n_aliens, n_crew, k, ship)

    def get_target_location(self, ship):
        #Only difference is squeezing the risk array so array lookup functions properly
        risk = np.squeeze(np.copy(self.alienP))
        for i, val in enumerate(risk):
            if val > 0:
                risk[i] = math.ceil(100 * val)
            else:
                risk[i] = 1
        max_utility = -1
        options = []
        for index, probCrew in enumerate(self.crewP):
            distance = ship.getManhattanDistance(*self.location, *ship.open_cells[index])
            if distance == 0:
                distance = 1
            utility = probCrew / (risk[index] * distance)
            if utility > max_utility:
                options = [ship.open_cells[index]]
                max_utility = utility
            elif utility == max_utility:
                options.append(ship.open_cells[index])
        return options[random.randint(len(options))]  # return random best option

"""
def test1():
    updates = 0

    # board = ship(6, 0.9, 1)i pus

    # For the fact bot is setting non-adjacent cells away from it to 0
    # board = ship(32, 0.1, 1), stop at updates == 141 if what I wrote in discord is an issue
    board = ship(32, 0.1, 7)

    print("Bot Start", board.bot_start_location)
    print("Goal", board.crew_locations)
    print("alien locations start", board.alien_locations)

    while True:
        # print("iteration: ",i)
        # print(board.testBoard)
        # print(board.board)
        # print("bot at ", board.bot.location)
        # print("alien at", board.alien_locations)
        # print(board.bot.location)
        # print("Update #: ", updates)
        check = board.update()
        # print("bot location", board.bot.location)
        # print("alien locations", board.alien_locations)
        if check == -1 or updates == 1000:
            print("FAILURE - BOT CAUGHT")
            break
        updates = updates + 1
        if check == 1:
            print("PASS - BOT GOAL")
            break


def test2():
    benchmark = 0
    benchmarkCount = 0
    fail = 0
    success = 0
    surv = 0
    for i in range(20):
        print("Trial i ", i)
        updates = 0

        # k = random.randint(1,35)
        # randomized = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2.0]
        # alphaInd = random.randint(0,len(randomized))
        # alpha = randomized[alphaInd]
        # print(k, alpha)
        # k = 7

        try:
            board = ship(4, 0.5, 8)
        except:
            continue

        while True:
            benchmarkCount += 1
            # print("iteration: ",i)
            # print(board.testBoard)
            # print(board.board)
            # print("bot at ", board.bot.location)
            # print("alien at", board.alien_locations)

            start = time.time()
            check = board.update()
            end = time.time()
            benchmark = benchmark + end - start
            # print("benchmark for time", benchmark/benchmarkCount)

            # print("bot location", board.bot.location)
            # print("alien locations", board.alien_locations)
            if check == -1:
                print("FAILURE - BOT CAUGHT")
                fail += 1
                break
            updates = updates + 1
            if check == 1:
                print("PASS - BOT GOAL")
                success += 1
                break
            if updates == 1000:
                print("RAN OUT OF TIME_____________________________________________")
                surv += 1
                # return
                break
        print(updates)
    print("success:", success)
    print("fail:", fail)
    print("surv:", surv)
test2()
"""



def simulate_trials(params):
    k, alpha, botmode, trials = params
    success, failure, survive, moves_in_rescue, num_crew_saved = 0, 0, 0, 0, 0
    for i in range(trials):
        updates = 0

        try:
            board = ship(k, alpha, botmode)  # (k, alpha, botmode)
        except:
            continue

        while True:
            try:
                check = board.update()
            except Exception as e:
                print("Error occurred while updating board:", e)
                break
            if check == -1:
                failure += 1
                if botmode >= 3: #2 crew members
                    num_crew_saved += 2 - len(board.crew_locations)
                break
            updates += 1
            if check == 1:
                success += 1
                moves_in_rescue += updates
                if botmode >= 3: #2 crew members
                    num_crew_saved += 2 - len(board.crew_locations)
                else:
                    num_crew_saved += 1
                break
            if updates == 1000:
                survive += 1
                if botmode >= 3: #2 crew members
                    num_crew_saved += 2 - len(board.crew_locations)
                break
    return {'bot_number': botmode,
            'k': k,
            'alpha': alpha,
            'success': success,
            'failed': failure,
            'survive': survive,
            'moves_in_rescue': moves_in_rescue,
            'num_crew_saved': num_crew_saved}


if __name__ == "__main__":
    trials = 50
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    print(timestamp)
    op = []
    alphas = np.arange(0.1, 1.1, 0.1)  # Generate values from 0.1 to 2.0 with step 0.1
    alphas = np.round(alphas * 10) / 10  # Round the values to the nearest 0.1
    folder_name = 'partial_data2'
    os.makedirs(folder_name, exist_ok=True)
    k = 7
    params_list = [(k, alpha, bot_number, trials) for alpha in alphas for bot_number in range(7, 9)]  # Create parameter list for multiprocessing
    max_cores = 6
    with multiprocessing.Pool(processes=max_cores) as pool:  # Multiprocessing stuff
        for result in pool.imap_unordered(simulate_trials, params_list):
            op.append(result)

            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            partial_filename = f'{folder_name}/data_partial_{timestamp}_{len(op)}.json'
            with open(partial_filename, 'w', encoding="utf-8") as f:
                json.dump(op, f, indent=4)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    print(timestamp)

    filename = f'data_{timestamp}.json'  # Save json file with unique name
    with open(filename, 'w', encoding="utf-8") as f:  # Save json using our list
        json.dump(op, f, indent=4)

"""
#print("Bot", board.bot.location)
#print("alien locations", board.alien_locations)

#Data Collection Code
#Note: Comment This Whole Block Out For Testing Purposes
botmode = 1
trials = 250
maxAliens = 650

#When there are no aliens, all trials are successful
#And -1 in the survivalArray to indicate it never failed
aliensArray = [0]
successArray = [trials]
survivalArray = [-1]


#Tests N trials for a range of 1 to maxAliens aliens each
for env in range(1,maxAliens + 1):
    print("Aliens on Board is " + str(env))
    #Total Survival is the sum of all survival time among failed runs
    totalSurvival = 0
    #Number of successes for k aliens
    success = 0
    for trial in range(trials):
        if trial % 25 == 0:
            print("Trial is at " + str(trial))

        #reset survival to 0 for a new trial
        survival = 0

        board = ship(env,30,botmode)

        while True:
            check = board.update()

            if survival > 1000 or check == -1:
                #print("Trial:" + str(trial) + ": FAILURE - BOT CAUGHT")
                #print("Survival Rate: " + str(survival))

                #Since we are only concerned with the totalSurvival for failed attempts, we add it here
                totalSurvival = totalSurvival + survival
                break

            #If the bot was not captured, add one to survival for that trial
            survival = survival + 1
            if check == 1:
                #print("Trial: " + str(trial) + ": PASS - BOT GOAL")
                #If bot found the captain, mark it as successful
                success = success + 1
                break
    aliensArray.append(env)
    successArray.append(success)

    #averageSurvival is the totalSurvival/failures
    failures = trials - success
    if failures == 0:
        averageSurvival = -1
    else:
        averageSurvival = totalSurvival / failures
    survivalArray.append(averageSurvival)

#Outputs everything to a json file to make data more easily accessible
file = ""
if botmode == 1:
    file = "data1new.json"
if botmode == 2:
    file = "data2new.json"
if botmode == 3:
    file = "data3new.json"
if botmode == 4:
    file = "data4new.json"

with open(file, "w") as outfile:
    outfile.write("Aliens: " + json.dumps(aliensArray)+"\n")
    outfile.write("Success: " + json.dumps(successArray)+"\n")
    outfile.write("Survival: " + json.dumps(survivalArray))
"""
