
# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0  
"""Unitree G1 robot task module
contains various task implementations for the G1 robot, such as pick and place, motion control, etc.
"""

# use relative import

from . import h12_pick_place_cylinder_27dof_inspire
from . import h12_stack_rgyblock_27dof_inspire
from . import h12_pick_place_redblock_27dof_inspire
from . import h12_move_cylinder_27dof_inspire_wholebody
from . import h12_warehouse_walk_27dof_inspire_wholebody
from . import h12_torso_flat_27dof_inspire_joint
from . import h12_manager_torso_flat_27dof_inspire_joint


# export all modules
__all__ = [
        "h12_pick_place_cylinder_27dof_inspire",
        "h12_stack_rgyblock_27dof_inspire",
        "h12_pick_place_redblock_27dof_inspire",
        "h12_move_cylinder_27dof_inspire_wholebody",
        "h12_warehouse_walk_27dof_inspire_wholebody",
        "h12_torso_flat_27dof_inspire_joint",
        "h12_manager_torso_flat_27dof_inspire_joint",

]
