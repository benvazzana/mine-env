
import layouts
from mine import MineLayout, MineEnv

class MineEnv20x15(MineEnv):
    def __init__(self, random_target=False):
        layout = MineLayout(layout=layouts.LAYOUT1)
        screen_size = (layout.width * 48, layout.height * 48)
        if random_target:
            super(MineEnv20x15, self).__init__(mine_layout=layout, screen_size=screen_size)
        else:
            super(MineEnv20x15, self).__init__(mine_layout=layout, target_loc=(19, 14), screen_size=screen_size)