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

    def __init__(self, size, square_size=50):
        self.square_size = square_size
        self.size = size  # Change this value to your needs!(12 max)
        self.width, self.height = self.size * self.square_size, self.size * self.square_size
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

    def print_board(self):
        # Change the contents of the icon dictionary to print the correct graphics..
        # Print_board will translate the 2d array icon board(playerboard) into a x*y graphical board.
        # Vars
        # global icons, error_message
        x = 0
        y = 0
        size = self.square_size
        icon = {
            self._LAND_ICON: self._GROUND_RESOURCE,
            self._BANDIT_ICON: self._BANDIT_RESOURCE,
            self._HEALTH_ICON: self._HEALTH_RESOURCE,
            self._PLAYER_ICON: self._PLAYER_RESOURCE,
        }
        self.clear()
        # Text board printer
        for row in self.movable_layer[:]:
            for j in range(1):
                print(" ".join(row))
        # Graphical board printer(Left to right, row by row)
        for row in self.movable_layer:
            cout = 0
            x = 0
            for square in row:
                cout += 1
                try:
                    img = pygame.image.load(icon[square])
                except Exception as e:
                    img = pygame.image.load(self._GROUND_RESOURCE)
                    print("image failed", square)
                self.screen.blit(img, (x, y))
                if x < self.width:
                    x += size
                else:
                    x = 0
            y += size
        pygame.display.update()


if __name__ == "__main__":
    gui = GUI()
    while True:
        gui.render_action(random.choice(list("wasd")))
