import os
import pygame
import random
import sys


class GUI(object):
    _GROUND_RESOURCE = 'resources/ground.jpg'
    _BANDIT_RESOURCE = 'resources/bandit.jpg'
    _HEALTH_RESOURCE = 'resources/health.jpg'
    _PLAYER_RESOURCE = 'resources/player.jpg'

    _LAND_ICON = ' '
    _BANDIT_ICON = 'B'
    _HEALTH_ICON = 'H'
    _PLAYER_ICON = "P"

    _icon = {
        _LAND_ICON: _GROUND_RESOURCE,
        _BANDIT_ICON: _BANDIT_RESOURCE,
        _HEALTH_ICON: _HEALTH_RESOURCE,
        _PLAYER_ICON: _PLAYER_RESOURCE,
    }

    def __init__(self, size, grid, tile_size=50):
        self.tile_size = tile_size
        self.size = size  # Change this value to your needs!(12 max)

        self.icon = {}
        for key, val in self._icon.items():
            img = pygame.image.load(val)
            self.icon[key] = pygame.transform.scale(img, (self.tile_size, self.tile_size))

        self.width, self.height = self.size * self.tile_size, self.size * self.tile_size
        pygame.init()
        self.screen = pygame.display.set_mode((self.width, self.height))

        if sys.platform == "linux" or sys.platform == "linux2":  # Clear cmd/terminal text screen
            self.clear = lambda: os.system('clear')
        elif sys.platform == "darwin":
            self.clear = lambda: os.system('clear')
        elif sys.platform == "win32":
            self.clear = lambda: os.system('cls')

        # Boards
        self.background_layer = []  # Static objects such as buttons, ground and walls.
        # self.memory_board = []  # Counts the steps each block has recieved for limited use items such as health pacs
        self.movable_layer = []  # Front-end. Move your player around this board without affecting static objects.

        self.idx_to_xy = {}
        for row_idx, row in enumerate(grid):
            for col_idx, state in enumerate(row):
                self.idx_to_xy[state] = (col_idx, row_idx)

        # Map array writer
        for click in range(self.size):
            self.background_layer.append([self._LAND_ICON] * self.size)
            self.movable_layer.append([self._LAND_ICON] * self.size)
            # self.memory_board.append([0] * self.size)

        # vars
        error_message = ""
        self.player_xy = [(size - 2), 0, 0]
        self.movable_layer[self.player_xy[0]][self.player_xy[1]] = self._PLAYER_ICON

    def render_action(self, action):
        # User input loop.
        self.print_board()

        old_pos = self.player_xy[:]
        self.player_xy = self.board_transport(action, '', self.player_xy)
        if self.player_xy != old_pos:
            self.movable_layer[self.player_xy[0]][self.player_xy[1]] = self._PLAYER_ICON
            self.movable_layer[old_pos[0]][old_pos[1]] = self.background_layer[old_pos[0]][old_pos[1]]
        else:
            self.movable_layer[self.player_xy[0]][self.player_xy[1]] = self._PLAYER_ICON

    def board_transport(self, move_choice, em, who):
        # Board transport determins if the move it has been ordered to process is legal or not
        # (valid input and on-map) Once the move has been validated, the players x_y is changed to the new location.
        global error_message

        if len(move_choice) == 1 and move_choice[0] in ('w', 'a', 's', 'd'):
            # W.a.s.d movements. Add special keys above this if statement. eg if move_choice = 'x': playerstats()
            if move_choice[0] == "w":
                if who[0] - int(1) >= 0:  # UP
                    if self.movable_layer[(who[0] - int(1))][who[1]] not in ():  # add restricted icons here.
                        who[0] -= int(1)
                    else:
                        em = "You can't move there!"
                else:
                    em = "You can't move there!"

            elif move_choice[0] == "s":
                if (who[0] + int(1)) <= (len(self.background_layer) - 1):  # DOWN
                    if self.movable_layer[(who[0] + int(1))][who[1]] not in ():  # add restricted icons here.
                        who[0] += int(1)
                    else:
                        em = "You can't move there!"
                else:
                    em = "You can't move there!"

            elif move_choice[0] == "d":
                if who[1] + int(1) <= (len(self.background_layer) - 1):  # RIGHT
                    if self.movable_layer[who[0]][(who[1] + int(1))] not in ():  # add restricted icons here.
                        who[1] += int(1)
                    else:
                        em = "You can't move there!"
                else:
                    em = "You can't move there!"

            elif move_choice[0] == "a":
                if who[1] - int(1) >= 0:  # LEFT
                    if self.movable_layer[who[0]][(who[1] - int(1))] not in ():  # add restricted icons here.
                        who[1] -= int(1)
                    else:
                        em = "You can't move there!"
                else:
                    em = "You can't move there!"
            else:
                em = "What?"
        else:
            em = "Controls: w,a,s,d + enter"
        error_message = em
        return who

    def print_board(self, player_state, terminal_states, mode="graphical"):
        # Change the contents of the icon dictionary to print the correct graphics..
        # Print_board will translate the 2d array icon board(playerboard) into a x*y graphical board.
        # Vars
        # global icons, error_message
        if mode == "ascii":
            self.clear()
            # Text board printer
            for row in self.movable_layer[:]:
                    print(" ".join(row))
        else:
            # Graphical board printer(Left to right, row by row)
            for idx, (x, y) in self.idx_to_xy.items():
                if idx == player_state:
                    square = self._PLAYER_ICON
                elif idx in terminal_states:
                    square = self._HEALTH_ICON
                else:
                    square = self._LAND_ICON

                self.screen.blit(self.icon[square], (x * self.tile_size, y * self.tile_size))
            pygame.display.update()


if __name__ == "__main__":
    gui = GUI()
    while True:
        gui.render_action(random.choice(list("wasd")))
