import os
# from collections import frozendict
import pygame
import random
import sys


class GUI(object):
    _GROUND_RESOURCE = 'envs/resources/ground.jpg'
    _BANDIT_RESOURCE = 'envs/resources/bandit.jpg'
    _HEALTH_RESOURCE = 'envs/resources/health.jpg'
    _PLAYER_RESOURCE = 'envs/resources/player.jpg'
    _UPARRORW_RESOURCE = 'envs/resources/arrow_up.jpg'
    _DOWNARRORW_RESOURCE = 'envs/resources/arrow_down.jpg'
    _LEFTARRORW_RESOURCE = 'envs/resources/arrow_left.jpg'
    _RIGHTARRORW_RESOURCE = 'envs/resources/arrow_right.jpg'

    _LAND_ICON = ' '
    _BANDIT_ICON = 'B'
    _HEALTH_ICON = 'H'
    _PLAYER_ICON = "P"

    _UP = 0
    _RIGHT = 1
    _DOWN = 2
    _LEFT = 3

    _icon = {
        _LAND_ICON: _GROUND_RESOURCE,
        _BANDIT_ICON: _BANDIT_RESOURCE,
        _HEALTH_ICON: _HEALTH_RESOURCE,
        _PLAYER_ICON: _PLAYER_RESOURCE,
        _UP: _UPARRORW_RESOURCE,
        _DOWN: _DOWNARRORW_RESOURCE,
        _LEFT: _LEFTARRORW_RESOURCE,
        _RIGHT: _RIGHTARRORW_RESOURCE,
    }

    def __init__(self, width, tile_size=50):
        self.tile_size = tile_size
        self.width = width
        self.height = width

        self.icon = {}
        for key, val in self._icon.items():
            img = pygame.image.load(val)
            self.icon[key] = pygame.transform.scale(img, (int(self.tile_size * 0.5), int(self.tile_size * 0.5)))

        self.display_width, self.display_height = self.width * self.tile_size, self.width * self.tile_size
        pygame.init()
        self.screen = pygame.display.set_mode((self.display_width, self.display_height))

        # vars
        self.player_xy = [(self.width - 2), 0, 0]

    def print_board(self, player_position=None, terminal_states=(), some_matrix=None, policy=None, goals=None,
                    walls={}, boxes=(), player_state=None):

        font = pygame.font.SysFont("monospace", 15)
        policy_font = pygame.font.SysFont("monospace", 50)
        self.screen.fill((255, 255, 255))
        num_of_tiles = self.width * self.height
        if some_matrix is not None:
            some_matrix_max = 1e-9 + some_matrix[num_of_tiles: num_of_tiles * 2].max()
            some_matrix_min = 1e-9 + some_matrix[num_of_tiles: num_of_tiles * 2].min()
        for tile_idx in range(num_of_tiles):
            # plot the values for hungry ~thirsty
            state_hash = tile_idx + num_of_tiles
            x = tile_idx % self.width
            y = tile_idx // self.width
            if tile_idx == player_position:
                square_icon = self.icon[self._PLAYER_ICON]
            elif policy is not None:
                policy_act = policy[state_hash]
                try:
                    square_icon = self.icon[policy_act]
                except KeyError:
                    self.icon[policy_act] = policy_font.render(str(policy_act), 1, (255, 255, 0))
                    square_icon = self.icon[policy_act]
            # elif tile_idx == player_position:
            #     square_icon = self._PLAYER_ICON
            elif tile_idx in terminal_states:
                square_icon = self._BANDIT_ICON
            elif tile_idx in boxes:
                square_icon = self._HEALTH_ICON
            else:
                square_icon = self._LAND_ICON
            if goals is not None and tile_idx in goals:
                square_icon = self._HEALTH_ICON

            icon_x = x * self.tile_size
            icon_y = y * self.tile_size

            self.screen.blit(square_icon, (icon_x, icon_y))

            blue = (0, 0, 255)
            if some_matrix is not None:
                value = (some_matrix[state_hash] - some_matrix_min) / (some_matrix_max - some_matrix_min)
                value *= 255.0

                # color = (value, 255, 0)
                value = str(int(some_matrix[state_hash] * 100) / 100)
                text = font.render(value, 2, blue)
                # rect_coords = (icon_x + self.tile_size / 2, icon_y + self.tile_size / 2, 10, 10)
                # pygame.draw.rect(self.screen, color, rect_coords)
            else:
                text = font.render(str(tile_idx), 1, blue)

            # self.screen.blit(font_repr, (icon_x + self.tile_size / 2, icon_y + self.tile_size / 2))
            self.screen.blit(text, (icon_x, icon_y + self.tile_size / 2))

            try:
                tile_walls = walls[tile_idx]
            except KeyError:
                pass
            else:
                if tile_idx + 1 in tile_walls:
                    pygame.draw.rect(self.screen, (255, 255, 0),
                                     [icon_x + self.tile_size - 10, icon_y, 10, self.tile_size])
                if tile_idx - 1 in tile_walls:
                    pygame.draw.rect(self.screen, (255, 0, 0), [icon_x, icon_y, 10, self.tile_size])
                if tile_idx + 6 in tile_walls:
                    pygame.draw.rect(self.screen, (0, 255, 0),
                                     [icon_x, icon_y + self.tile_size - 10, self.tile_size, 10])
                if tile_idx - 6 in tile_walls:
                    pygame.draw.rect(self.screen, (0, 0, 255), [icon_x, icon_y, self.tile_size, 10])
        pygame.display.update()

    def act_to_icon(self, policy_act):
        pass
