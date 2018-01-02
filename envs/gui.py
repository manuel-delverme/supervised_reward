import os
# from collections import frozendict
import pygame
import random
import sys
import math


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

    def __init__(self, width, tile_size=80):
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
                    walls={}, boxes=(), hungry=False, thirsty=False, highlight_square=None):

        font = pygame.font.SysFont("monospace", 15)
        policy_font = pygame.font.SysFont("monospace", 50)
        self.screen.fill((255, 255, 255))
        num_of_tiles = self.width * self.height

        # if some_matrix is not None:
        #     some_matrix_max = 1e-9 + some_matrix[slice_start: slice_end].max()
        #     some_matrix_min = 1e-9 + some_matrix[slice_start: slice_end].min()

        # plot policy for hungry?
        slice_start = int(hungry) * num_of_tiles
        # and thirsty?
        slice_start += int(thirsty) * num_of_tiles * 2
        # slice_end = slice_start + num_of_tiles

        for tile_idx in range(num_of_tiles):
            state_hash = tile_idx + slice_start
            x = tile_idx % self.width
            y = tile_idx // self.width
            icon_x = x * self.tile_size + 10
            icon_y = y * self.tile_size

            if policy is not None:
                policy_act = policy[state_hash]
                try:
                    square_icon = self.icon[policy_act]
                except KeyError:
                    self.icon[policy_act] = policy_font.render(str(policy_act), 1, (255, 255, 0))
                    square_icon = self.icon[policy_act]
                self.screen.blit(square_icon, (icon_x, icon_y + self.tile_size/2))

            if tile_idx == player_position:
                square_icon = self.icon[self._PLAYER_ICON]
            # elif tile_idx == player_position:
            #     square_icon = self._PLAYER_ICON
            elif tile_idx in terminal_states:
                square_icon = self.icon[self._BANDIT_ICON]
            elif tile_idx in boxes:
                square_icon = self.icon[self._HEALTH_ICON]
            else:
                square_icon = self.icon[self._LAND_ICON]
            if goals is not None and tile_idx in goals:
                square_icon = self.icon[self._HEALTH_ICON]

            if highlight_square is not None and highlight_square % num_of_tiles == tile_idx:
                pygame.draw.rect(self.screen, (255, 255, 0), [icon_x, icon_y, self.tile_size, self.tile_size])
            if square_icon:
                self.screen.blit(square_icon, (icon_x, icon_y))

            blue = (0, 0, 255)
            red = (255, 0, 0)

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
            if some_matrix is not None:
                # value = (some_matrix[state_hash] + some_matrix_min) / (1e-33 + some_matrix_max - some_matrix_min)
                # value *= 255.0

                # color = (value, 255, 0)
                value = 0
                for offset in range(0, some_matrix.shape[0] // num_of_tiles):
                    value_idx = tile_idx + offset * num_of_tiles
                    value += some_matrix[value_idx]
                    # value = "{1}[{0}]".format(value_idx, value)

                value = str(round(value / (some_matrix.shape[0] // num_of_tiles), 2))

                if value_idx == state_hash:
                    text = font.render(value, 2, red)
                else:
                    text = font.render(value, 2, blue)
                # rect_coords = (icon_x + self.tile_size / 2, icon_y + self.tile_size / 2, 10, 10)
                # pygame.draw.rect(self.screen, color, rect_coords)
                offset_x = 0
                if offset > 1:
                    offset = offset - 2
                    offset_x = 1
                self.screen.blit(text, (icon_x - 5 + self.tile_size / 2 * offset_x, int(icon_y + self.tile_size * 3 / 4)))

            black = (0, 0, 0)
            for offset in range(0, 2):
                value_idx = str(tile_idx + offset * num_of_tiles)
                if value_idx == state_hash:
                    text = font.render(value_idx, 2, red)
                else:
                    text = font.render(value_idx, 2, black)
                self.screen.blit(text, (icon_x, icon_y + offset * 10))
            for offset in range(2, 4):
                value_idx = str(tile_idx + offset * num_of_tiles)
                if value_idx == state_hash:
                    text = font.render(value_idx, 2, red)
                else:
                    text = font.render(value_idx, 2, black)
                self.screen.blit(text, (icon_x + self.tile_size / 2, icon_y + (offset - 2) * 10))
        pygame.display.update()
