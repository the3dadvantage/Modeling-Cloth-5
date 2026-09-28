import numpy as np
import bpy


try:
    U = bpy.data.texts['utils.py'].as_module()
    MB = bpy.data.texts['matrix_centers_bend.py'].as_module()
except:
    from . import utils as U
    from . import matrix_centers_bend as MB


def setup_groups(C):
    v_list = np.arange(C.vc, dtype=np.int32).tolist()

    bend_group = np.ones(C.vc, dtype=np.float32)[:, None]
    stretch_group = np.ones(C.vc, dtype=np.float32)[:, None]
    pin_group = np.zeros(C.vc, dtype=np.float32)[:, None]
    velocity_group = np.ones(C.vc, dtype=np.float32)[:, None]
    friction_group = np.ones(C.vc, dtype=np.float32)[:, None]
    wind_group = np.ones(C.vc, dtype=np.float32)[:, None]
    inflate_group = np.ones(C.vc, dtype=np.float32)[:, None]
    air_drag_group = np.ones(C.vc, dtype=np.float32)[:, None]
    inverted_air_drag_group = np.ones(C.vc, dtype=np.float32)[:, None]

    C.default_wieghts = {"MC_bend": 1.0,
                         "MC_stretch": 1.0,
                         "MC_pin": 0.0,
                         "MC_velocity": 1.0,
                         "MC_friction": 1.0,
                         "MC_wind": 1.0,
                         "MC_inflate": 1.0,
                         "MC_air_drag": 1.0,
                         "MC_inverted_air_drag": 1.0,
                         }

    groups = {"MC_bend": bend_group,
              "MC_stretch": stretch_group,
              "MC_pin": pin_group,
              "MC_velocity": velocity_group,
              "MC_friction": friction_group,
              "MC_wind": wind_group,
              "MC_inflate": inflate_group,
              "MC_air_drag": air_drag_group,
              "MC_inverted_air_drag": inverted_air_drag_group,
              }

    for k, v in C.default_wieghts.items():
        missing = U.create_missing_groups(C.ob, v_list=v_list, name=k, weight=v)
        if not missing:
            groups[k] = U.get_weights_fast_with_group_check(C.ob, k, groups[k])
    
    C.group_data = groups


def group_check(C, active_name=None):
    ob = C.ob

    if active_name is None:
        active_name = bpy.context.object.vertex_groups.active.name
        active_idx = ob.vertex_groups.active_index
    else:
        active_idx = ob.vertex_groups[active_name].index


    if active_name in C.group_data:
        try:
            U.get_vertex_weights_fast_no_check(ob, active_idx, weights=C.group_data[active_name])
                        
            if active_name == "MC_bend":
                MB.get_hinge_weights(C)
            if active_name == "MC_stretch":
                C.left_group_mult = C.group_data['MC_stretch'][C.es_0]
                C.right_group_mult = C.group_data['MC_stretch'][C.es_1]
        except:
            U.assign_default_weights(C.ob, C.group_data[active_name], 0.0, active_name)
