import pprint

pygame = "HACK"


class GUI(object):
    _GROUND_RESOURCE = 'envs/resources/ground.jpg'
    _BANDIT_RESOURCE = 'envs/resources/bandit.jpg'
    _HEALTH_RESOURCE = 'envs/resources/health.jpg'
    _PLAYER_RESOURCE = 'envs/resources/player.jpg'
    _WATER_RESOURCE = 'envs/resources/water.jpg'
    _FOOD_RESOURCE = 'envs/resources/health.jpg'
    _UPARRORW_RESOURCE = 'envs/resources/arrow_up.jpg'
    _DOWNARRORW_RESOURCE = 'envs/resources/arrow_down.jpg'
    _LEFTARRORW_RESOURCE = 'envs/resources/arrow_left.jpg'
    _RIGHTARRORW_RESOURCE = 'envs/resources/arrow_right.jpg'
    _BOXCLOSED_RESOURCE = 'envs/resources/closed.jpg'
    _BOXHALF_RESOURCE = 'envs/resources/half.jpg'
    _BOXOPEN_RESOURCE = 'envs/resources/open.jpg'

    _LAND_ICON = ' '
    _BANDIT_ICON = 'B'
    _HEALTH_ICON = 'H'
    _PLAYER_ICON = "P"
    _WATER_ICON = "W"
    _FOOD_ICON = "F"

    _UP = 0
    _RIGHT = 1
    _DOWN = 2
    _LEFT = 3

    _BOXCLOSED_ICON = 'boxclosed'
    _BOXHALF_ICON = 'boxhalf'
    _BOXOPEN_ICON = 'boxopen'

    _icon = {
        _LAND_ICON: _GROUND_RESOURCE,
        _BANDIT_ICON: _BANDIT_RESOURCE,
        _HEALTH_ICON: _HEALTH_RESOURCE,
        _PLAYER_ICON: _PLAYER_RESOURCE,
        _FOOD_ICON: _FOOD_RESOURCE,
        _WATER_ICON: _WATER_RESOURCE,
        _UP: _UPARRORW_RESOURCE,
        _DOWN: _DOWNARRORW_RESOURCE,
        _LEFT: _LEFTARRORW_RESOURCE,
        _RIGHT: _RIGHTARRORW_RESOURCE,
        _BOXCLOSED_ICON: _BOXCLOSED_RESOURCE,
        _BOXHALF_ICON: _BOXHALF_RESOURCE,
        _BOXOPEN_ICON: _BOXOPEN_RESOURCE,
    }

    def __init__(self, width, tile_size=80):
        import pygame
        self.tile_size = tile_size
        self.width = width
        self.height = width

        self.icon = {}
        for key, val in self._icon.items():
            img = pygame.image.load(val)
            self.icon[key] = pygame.transform.scale(img, (int(self.tile_size * 0.5), int(self.tile_size * 0.5)))

        self.display_width, self.display_height = self.width * self.tile_size, self.width * self.tile_size
        pygame.init()
        self.screen = pygame.display.set_mode((self.display_width, self.display_height + 100))

        # vars
        self.player_xy = [(self.width - 2), 0, 0]

    def render_board(
            self, player_position=None, terminal_states=(), some_matrix=None, policy=None, goals=None,
            walls={}, boxes={}, hungry=False, thirsty=False, state_offset=None,
            highlight_square=None, info=False, water_position=None, food_position=False,
    ):
        COLOR_BLUE = (0, 0, 255)
        COLOR_GREEN = (0, 255, 0)
        COLOR_RED = (255, 0, 0)
        COLOR_BLACK = (0, 0, 0)
        COLOR_YELLOW = (255, 255, 0)

        font = pygame.font.SysFont("monospace", 15)
        policy_font = pygame.font.SysFont("monospace", 50)
        self.screen.fill((255, 255, 255))
        num_of_tiles = self.width * self.height

        if state_offset is not None:
            slice_start = state_offset
        else:
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

            policy_position = (icon_x, icon_y + self.tile_size / 2)

            if policy is not None:
                policy_act = policy[state_hash]
                try:
                    policy_icon = self.icon[policy_act]
                except KeyError:
                    self.icon[policy_act] = policy_font.render(str(policy_act), 1, COLOR_GREEN)
                    policy_icon = self.icon[policy_act]
                self.screen.blit(policy_icon, policy_position)

            square_icon = None
            if tile_idx == player_position:
                square_icon = self.icon[self._PLAYER_ICON]
            elif tile_idx in terminal_states:
                square_icon = self.icon[self._BANDIT_ICON]
            elif tile_idx == water_position:
                square_icon = self.icon[self._WATER_ICON]
            elif tile_idx == food_position:
                square_icon = self.icon[self._FOOD_ICON]
            elif tile_idx in boxes.keys():
                box_status = boxes[tile_idx].value
                if box_status == 0:
                    square_icon = self.icon[self._BOXOPEN_ICON]
                elif box_status == 1:
                    square_icon = self.icon[self._BOXHALF_ICON]
                elif box_status == 2:
                    square_icon = self.icon[self._BOXCLOSED_ICON]
            else:
                square_icon = self.icon[self._LAND_ICON]

            # if goals is not None and tile_idx in goals:
            #     square_icon = self.icon[self._HEALTH_ICON]

            if highlight_square is not None and highlight_square % num_of_tiles == tile_idx:
                pygame.draw.rect(self.screen, COLOR_YELLOW, [icon_x, icon_y, self.tile_size, self.tile_size])

            if square_icon:
                self.screen.blit(square_icon, (icon_x, icon_y))

            try:
                tile_walls = walls[tile_idx]
            except KeyError:
                pass
            else:
                if tile_idx + 1 in tile_walls:
                    pygame.draw.rect(self.screen, COLOR_YELLOW,
                                     [icon_x + self.tile_size - 10, icon_y, 10, self.tile_size])
                if tile_idx - 1 in tile_walls:
                    pygame.draw.rect(self.screen, COLOR_RED, [icon_x, icon_y, 10, self.tile_size])
                if tile_idx + 6 in tile_walls:
                    pygame.draw.rect(self.screen, COLOR_GREEN,
                                     [icon_x, icon_y + self.tile_size - 10, self.tile_size, 10])
                if tile_idx - 6 in tile_walls:
                    pygame.draw.rect(self.screen, (0, 0, 255), [icon_x, icon_y, self.tile_size, 10])

            if some_matrix is not None:
                value = 0
                for offset in range(0, some_matrix.shape[0] // num_of_tiles):
                    value_idx = tile_idx + offset * num_of_tiles
                    value += some_matrix[value_idx]
                    # value = "{1}[{0}]".format(value_idx, value)

                value = str(round(value / (some_matrix.shape[0] // num_of_tiles), 2))

                if value_idx == state_hash:
                    text = font.render(value, 2, COLOR_RED)
                else:
                    text = font.render(value, 2, COLOR_BLUE)
                # rect_coords = (icon_x + self.tile_size / 2, icon_y + self.tile_size / 2, 10, 10)
                # pygame.draw.rect(self.screen, color, rect_coords)
                offset_x = 0
                if offset > 1:
                    offset = offset - 2
                    offset_x = 1
                self.screen.blit(text,
                                 (icon_x - 5 + self.tile_size / 2 * offset_x, int(icon_y + self.tile_size * 3 / 4)))

            black = (0, 0, 0)
            for offset in range(0, 2):
                value_idx = str(tile_idx + offset * num_of_tiles)
                if value_idx == state_hash:
                    text = font.render(value_idx, 2, COLOR_RED)
                else:
                    text = font.render(value_idx, 2, black)
                self.screen.blit(text, (icon_x, icon_y + offset * 10))
            for offset in range(2, 4):
                value_idx = str(tile_idx + offset * num_of_tiles)
                if value_idx == state_hash:
                    text = font.render(value_idx, 2, COLOR_RED)
                else:
                    text = font.render(value_idx, 2, black)
                self.screen.blit(text, (icon_x + self.tile_size / 2, icon_y + (offset - 2) * 10))

        if info:
            for idx, row in enumerate(pprint.pformat(info, width=60).split("\n")):
                offset = idx * 20
                text = font.render(row, 2, COLOR_BLACK)
                self.screen.blit(text, (0, self.display_height + 10 + offset))
        pygame.display.update()
