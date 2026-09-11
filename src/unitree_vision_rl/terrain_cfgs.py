from mjlab.terrains import TerrainGeneratorCfg
from mjlab.terrains.config import ( pyramid_stairs, random_rough, flat,
                                    pyramid_stairs_inv, hf_pyramid_slope,
                                    hf_pyramid_slope_inv, wave_terrain, random_spread_boxes,
                                    tilted_grid) 

# custom configs when adding new envs. can scale to any type and any difficulty.

def medium_terrains_cfg() -> TerrainGeneratorCfg:
    return TerrainGeneratorCfg(
    size=(8.0, 8.0), border_width=20.0,
    num_rows=12,            # difficulty resolution
    num_cols=20,            # ignored when curriculum=True
    curriculum=True,
    difficulty_range=(0.0, 1.0),
    add_lights=True,
    sub_terrains={
        "pyramid_stairs": pyramid_stairs(proportion=0.25,
                                        step_height_range=(0.0, 0.18),
                                        step_width=0.32),
        "pyramid_stairs_inv": pyramid_stairs_inv(proportion=0.25, 
                                                 step_height_range=(0.0, 0.18),
                                                 step_width=0.32),
        "random_spread_boxes": random_spread_boxes(proportion =0.10,
                                                   num_boxes=80,
                                                   box_width_range=(0.1, 0.84),
                                                   box_length_range=(0.05, 1.0),
                                                   box_yaw_range=(0.0, 360.0),
                                                   add_floor = True,
                                                   platform_width = 1,
                                                   border_width = 0.25
                                                   ),
    },
)

def hard_terrains_cfg() -> TerrainGeneratorCfg:
    return TerrainGeneratorCfg(
    size=(8.0, 8.0), border_width=20.0,
    num_rows=15,            # difficulty resolution
    num_cols=20,            # ignored when curriculum=True
    curriculum=True,
    difficulty_range=(0.0, 1.0),
    add_lights=True,
    sub_terrains={
        "pyramid_stairs": pyramid_stairs(proportion=0.20,
                                        step_height_range=(0.0, 0.24),
                                        step_width=0.32),
        "pyramid_stairs_inv": pyramid_stairs_inv(proportion=0.20,
                                        step_height_range=(0.0, 0.24),
                                        step_width=0.32),
        "random_spread_boxes": random_spread_boxes(proportion =0.20,
                                                   num_boxes=80,
                                                   box_width_range=(0.1, 1.0),
                                                   box_length_range=(0.05, 2.0),
                                                   box_yaw_range=(0.0, 360.0),
                                                   add_floor = True,
                                                   platform_width = 1,
                                                   border_width = 0.25
                                                   ),
        "tilted_grid": tilted_grid(proportion=0.15,
                                   grid_width=1,
                                   tilt_range_deg = 20,
                                   height_range=0.3,
                                   floor_depth=2),
    },
)
