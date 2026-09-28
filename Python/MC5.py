import bpy
import bmesh
import numpy as np
import time
import math
import traceback

try:
    U = bpy.data.texts['utils.py'].as_module()

    MCB = bpy.data.texts['matrix_centers_bend.py'].as_module()
    G = bpy.data.texts['groups.py'].as_module()
    SC = bpy.data.texts['self_collide_2.py'].as_module()
    OC = bpy.data.texts['object_collide.py'].as_module()
    CPB = bpy.data.texts['c_plus_bridge.py'].as_module()
    CACHE = bpy.data.texts['cache.py'].as_module()
    HOOK = bpy.data.texts['hook_tools.py'].as_module()
    WIND = bpy.data.texts['wind.py'].as_module()
    ST = bpy.data.texts['split_tuner.py'].as_module()
    NATIVE = bpy.data.texts['collide_native.py'].as_module()

except:
    from . import utils as U
    from . import matrix_centers_bend as MCB
    from . import groups as G
    from . import self_collide_2 as SC
    # these two were missing, so an installed addon hit a NameError the
    # first time it ran physics
    from . import object_collide as OC
    from . import c_plus_bridge as CPB
    from . import cache as CACHE
    from . import hook_tools as HOOK
    from . import wind as WIND
    from . import split_tuner as ST
    from . import collide_native as NATIVE

# scipy is only needed by magnetic targets.  Imported when that feature is
# first used, not here: as an installed addon a failed pip (no network, a
# locked-down machine) would otherwise stop the addon from enabling at all.
# U.require installs it at most once per session, so a failure cannot retry
# pip on every frame.
def cKDTree(*args, **kwargs):
    global cKDTree
    if U.require("scipy") is None:
        return None
    try:
        from scipy.spatial import cKDTree as _kd
    except Exception as e:          # installed but broken is still no scipy
        print("MC: scipy is there but unusable: %s" % e)
        return None
    cKDTree = _kd
    return _kd(*args, **kwargs)


DTF = np.float32
DTI = np.int32

def dprint(*args):
    """Developer noise: only when Debug Mode is on in the MC5 settings.

    These used to print on every frame -- solver timings, setup traces -- which
    buried anything that actually mattered in the console."""
    try:
        if not bpy.context.scene.MC_props.debug_mode:
            return
    except AttributeError:
        return                      # no scene properties yet (registering)
    print(*args)


DATA = {}
MAGNETIC_DATA = {}
COLLISION_DATA = {}
OC_DATA ={}
OC_DATA['obs'] = []


def apply_as_target(ob):
    if ob.data.is_editmode:
        ob.update_from_editmode()
    
    U.manage_shapes(ob, shapes=['Basis', 'MC_target', 'MC_current'], values=[0.0, 0.0, 1.0])
    co = U.get_shape_co_mode(ob, co=None, key='MC_current')
    
    if ob.data.is_editmode:
        obm = U.get_bmesh(ob)
        obm.verts.ensure_lookup_table()

        # shape key layers on the bmesh (edit-mode equivalent of key_blocks[...].data)
        layers = obm.verts.layers.shape
        basis_layer = layers.get('Basis')
        target_layer = layers.get('MC_target')
        current_layer = layers.get('MC_current')

        for i, v in enumerate(obm.verts):
            v.co = co[i]
            v[basis_layer] = co[i]
            v[target_layer] = co[i]
            v[current_layer] = co[i]

        bmesh.update_edit_mesh(ob.data)
    else:
        ob.data.vertices.foreach_set('co', co.ravel())
        ob.data.shape_keys.key_blocks['Basis'].data.foreach_set('co', co.ravel())
        ob.data.shape_keys.key_blocks['MC_target'].data.foreach_set('co', co.ravel())
        ob.data.shape_keys.key_blocks['MC_current'].data.foreach_set('co', co.ravel())
        ob.data.update()
    
    cloth = get_cloth(ob)
    cloth.velocity[:] = 0.0
    cloth.vel_start = co.copy()
    cloth.start_co = co.copy()
    

def reset_cloth_ob(ob):
    if ob.data.is_editmode:
        ob.update_from_editmode()
    
    U.manage_shapes(ob, shapes=['Basis', 'MC_target', 'MC_current'], values=[0.0, 0.0, 1.0])
    co = U.get_shape_co_mode(ob, co=None, key='MC_target')
    
    sel = np.ones(len(ob.data.vertices), dtype=bool)
    if bpy.context.scene.MC_props.reset_selected:
        ob.data.vertices.foreach_get('select', sel)
        cco = U.get_shape_co_mode(ob, co=None, key='MC_current')
        co[~sel] = cco[~sel]
        
    if ob.data.is_editmode:
        obm = U.get_bmesh(ob)
        obm.verts.ensure_lookup_table()
        for i, j in enumerate(co):
            if sel[i]:    
                obm.verts[i].co = j
    else:
        ob.data.shape_keys.key_blocks['MC_current'].data.foreach_set('co', co.ravel())
        ob.data.update()
    
    cloth = get_cloth(ob)
    cloth.velocity[:] = 0.0
    cloth.vel_start[:] = co
    cloth.start_co[:] = co
    # restart the wind clock with the cloth, so playing an animated shot from
    # the reset frame gives the same gusts every time
    cloth.wind_step = 0


def update_ob_colliders(data, counts=None):

    col_obs = [ob for ob in bpy.data.objects if ob.MC_props.ob_collision]
    data["obs"] = col_obs
    # Element numbering is rebuilt below, so contact pairs cached against the
    # old numbering (recollide) must not be used again -- they check this.
    data["version"] = data.get("version", 0) + 1

    if not col_obs:
        # Turning off (or deleting) the last collider used to reach the
        # np.concatenate calls below with empty lists, which raises
        # "need at least one array to concatenate" out of a property callback.
        # There is nothing to build, so clear the arrays and stop.
        for k in ("tridexes", "tridex_edges", "tridex_eidx", "eidxer",
                  "tri_edge_idxer", "edge_to_normal_idxer",
                  "edge_to_normal_counts", "edge_to_normal_add_idxer"):
            data[k] = np.zeros(0, dtype=np.int32)
        data["counts"] = []
        data["oc_co"] = np.empty((0, 3), dtype=np.float32)
        data["vert_normals"] = np.empty((0, 3), dtype=np.float32)
        data["local_edge_normals"] = np.zeros((0, 3), dtype=np.float32)
        data["edge_normals"] = np.zeros((0, 3), dtype=np.float32)
        data["eidx_booler"] = np.zeros((0, 2), dtype=bool)
        data["tridex_edge_booler"] = np.zeros(0, dtype=bool)
        return

    tridexes = []
    tridex_edges = []
    tridex_eidx = []
    tri_edge_idxer = []
    edge_to_normal_idxer = [] # e.link_faces in triobm
    edge_to_normal_counts = []
    edge_to_normal_add_idxer = []
    ec = 0
    
    ob_eidxer = []
    ob_eidx = []

    offset = 0
    edge_offset = 0
    eidx_offset = 0
    triangle_offset = 0
    for ob in col_obs:
        
        use_prox = False
        
        if ob.data.is_editmode:
            ob.update_from_editmode()        
        
        if (ob.data.shape_keys is not None) | len(ob.modifiers) > 0:
            ob = U.prox_object(ob)
            use_prox = True

        tridex, tri_eidx, tri_eidxer, tri_edge_idx = U.get_tri_edges(ob, use_prox)    
        tridexes += [tridex + offset]
        
        ec += tri_eidx.shape[0]
        edge_to_normal_idxer_, edge_to_normal_counts_, edge_to_normal_add_idxer_ = U.get_edge_normal_data(ob, use_prox)
        edge_to_normal_idxer += [edge_to_normal_idxer_ + triangle_offset]
        edge_to_normal_counts += [edge_to_normal_counts_]
        edge_to_normal_add_idxer += [edge_to_normal_add_idxer_ + edge_offset]
                
        tridex_edges += [tri_eidxer + edge_offset]
        tridex_eidx += [tri_eidx + offset]
        tri_edge_idxer += [tri_edge_idx + edge_offset]
        
        ec = len(ob.data.edges)
        eidx = np.empty((ec, 2), dtype=np.int32)
        ob.data.edges.foreach_get("vertices", eidx.ravel())
        
        ob_eidxer += [np.arange(ec) + eidx_offset]
        ob_eidx += [eidx + offset]    
        eidx_offset += ec
    
        offset += len(ob.data.vertices)
        edge_offset += tri_eidxer.shape[0]
        triangle_offset += tridex.shape[0]
    
    data["eidx_booler"] = np.zeros((eidx_offset, 2), dtype=bool)
    data["edge_normals"] = np.zeros((ec, 3), dtype=np.float32)
    
    if counts is None:    
        data["counts"] = [(len(ob.data.vertices), len(ob.data.edges), len(ob.data.polygons)) for ob in data["obs"]]
    data["tridexes"] = np.concatenate(tridexes)
    data["tridex_edges"] = np.concatenate(tridex_edges)
    data["tridex_eidx"] = np.concatenate(tridex_eidx)
    
    data["eidxer"] = data["tridex_edges"]
    
    # edge_to_normal:
    data["edge_to_normal_idxer"] = np.concatenate(edge_to_normal_idxer)
    data["edge_to_normal_counts"] = np.concatenate(edge_to_normal_counts)
    data["edge_to_normal_add_idxer"] = np.concatenate(edge_to_normal_add_idxer)
    edge_to_normal_idxer = [] # e.link_faces in triobm
    edge_to_normal_counts = []
    edge_to_normal_add_idxer = []
    
    data["local_edge_normals"] = np.zeros((data["edge_to_normal_counts"].shape[0], 3), dtype=np.float32)
    
    data["tri_edge_idxer"] = np.concatenate(tri_edge_idxer)
    data["tridex_edge_booler"] = np.zeros(data["tridex_eidx"].shape[0], dtype=bool)
    
    data["oc_co"] = np.empty((offset, 3), dtype=np.float32)
    data["vert_normals"] = np.empty((offset, 3), dtype=np.float32)
    data["vert_booler"] = np.zeros(offset, dtype=bool)
    data["joined_co"] = np.empty((offset, 2, 3), dtype=np.float32)
    data["idxer"] = np.arange(offset)

    tc = data["tridexes"].shape[0]
    data["tri_booler"] = np.zeros((tc, 3), dtype=bool)
    data["joined_trico"] = np.empty((tc, 6, 3), dtype=DTF)
    data["joined_normals"] = np.empty((tc, 3), dtype=DTF)
    data["tidx"] = np.arange(tc)

    read_collider_friction(data)


def read_collider_friction(data):
    """(Re)build the joined friction arrays from every collider's
    ob_collider_friction property times its 'MC_ob_friction' vertex group
    (default weight 1.0, auto-created).  Rebuilt on collider setup, on a
    friction-property change, and each frame while a collider is in edit /
    weight-paint mode.  object_collide.py reads:
        data['friction_vert']  (per collider vertex, aligns with joined_co)
        data['friction_tri']   (mean of the 3 triangle verts)
        data['friction_edge']  (mean of the 2 edge verts)
        data['friction_max']   (skip-the-whole-pass guard)
    """
    obs = data.get("obs", [])
    idxer = data.get("idxer")
    if not obs or idxer is None:
        for k in ("friction_vert", "friction_tri", "friction_edge"):
            data[k] = np.zeros(0, dtype=np.float32)
        data["friction_max"] = 0.0
        return

    parts = []
    for ob in obs:
        scalar = float(getattr(ob.MC_props, "ob_collider_friction", 1.0))
        try:
            U.create_missing_groups(ob, list(range(len(ob.data.vertices))), "MC_ob_friction", 1.0)
        except Exception:
            pass
        if ob.data.is_editmode:
            ob.update_from_editmode()
        use_prox = (ob.data.shape_keys is not None) or (len(ob.modifiers) > 0)
        src = U.prox_object(ob) if use_prox else ob
        n_v = len(src.data.vertices)
        fw = np.full(n_v, scalar, dtype=np.float32)
        try:
            if "MC_ob_friction" in src.vertex_groups:
                w = U.get_weights_fast_with_group_check(src, "MC_ob_friction")
                if w.shape[0] == n_v:
                    fw = w.astype(np.float32) * scalar
        except Exception:
            pass
        parts.append(fw)

    fv = np.concatenate(parts).astype(np.float32)
    if fv.shape[0] != idxer.shape[0]:
        return  # topology drifted since the last full rebuild; leave arrays as-is
    data["friction_vert"] = fv
    data["friction_tri"] = fv[data["tridexes"]].mean(axis=1).astype(np.float32)
    data["friction_edge"] = fv[data["tridex_eidx"]].mean(axis=1).astype(np.float32)
    data["friction_max"] = float(fv.max())


def _components_numpy(n_verts, sew_pairs):
    """Connected components of an edge list, union-find, no scipy.

    Blender ships numpy and nothing else, so scipy may simply not be there and
    may not be installable on a locked-down machine.  This is the whole of what
    was being asked of scipy here, so sewing keeps working either way.
    """
    parent = np.arange(n_verts, dtype=np.int64)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]          # path halving
            x = parent[x]
        return x

    for a, b in sew_pairs:
        ra, rb = find(int(a)), find(int(b))
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    roots = np.array([find(i) for i in range(n_verts)], dtype=np.int64)
    # renumber to 0..k-1 so the labels look like scipy's
    _, labels = np.unique(roots, return_inverse=True)
    return labels


def build_sew_labels(n_verts, sew_pairs):
    """
    n_verts: total vertex count
    sew_pairs: (M, 2) array of vertex index pairs from sew edges
               (edges with no linked faces)
    """
    if len(sew_pairs) == 0:
        return np.arange(n_verts)

    # scipy is preferred (it is faster and it is what this shipped with), but
    # it is imported here rather than at module level: an unguarded import at
    # the top took the entire addon down on a Blender whose scipy did not match
    # its numpy.
    sparse = U.require("scipy.sparse", install=False, quiet=True)
    csgraph = U.require("scipy.sparse.csgraph", install=False, quiet=True)
    if sparse is None or csgraph is None:
        return _components_numpy(n_verts, np.asarray(sew_pairs))

    row = sew_pairs[:, 0]
    col = sew_pairs[:, 1]
    data = np.ones(len(row), dtype=bool)
    graph = sparse.coo_matrix((data, (row, col)), shape=(n_verts, n_verts))
    _, labels = csgraph.connected_components(csgraph=graph, directed=False)
    return labels


def basic_sew_springs(cloth):
    
    mesh = cloth.ob.data
    cloth.obm.verts.ensure_lookup_table()
    cloth.obm.edges.ensure_lookup_table()
    cloth.obm.faces.ensure_lookup_table()
    vc = len(mesh.vertices)

    sew_edges = np.array([[e.verts[0].index, e.verts[1].index] for e in cloth.obm.edges if len(e.link_faces) == 0], dtype=np.int32)
    cloth.sew_pairs = sew_edges
    
    cloth.sew_labels = build_sew_labels(cloth.vc, sew_edges)
    
    cloth.sew_verts = np.unique(sew_edges.ravel())
    
    regular_edges = np.array([[e.verts[0].index, e.verts[1].index] for e in cloth.obm.edges if len(e.link_faces) != 0], dtype=np.int32)
    
    uni, inv, counts = np.unique(sew_edges.ravel(), return_inverse=True, return_counts=True)
    dup_verts = uni[counts > 1]

    dup_edges = {}
    for v in dup_verts:
        dup_edges[v] = [e.index for e in cloth.obm.verts[v].link_edges if len(e.link_faces) == 0]
    
    edge_groups = []
    dup_groups = []
    for e in sew_edges:
        if not np.any(e[0] == dup_verts):
            if not np.any(e[1] == dup_verts):
                edge_groups += [e.tolist()]
                continue
        dup_groups += [e.tolist()]
    
    dup_bool = np.ones(vc, dtype=bool)

    multi_groups = []
    for e in dup_groups:
        if dup_bool[e[0]]:
            new_group = set((e[0], e[1]))
            dup_bool[e] = False
            for e2 in dup_groups:
                if (dup_bool[e2[0]] | dup_bool[e2[1]]):
                    if (e2[0] in e) | (e2[1] in e):
                        dup_bool[e2] = False
                        new_group.add(e2[0])
                        new_group.add(e2[1])
            multi_groups += [list(new_group)]
            
    merge_groups = edge_groups + multi_groups
    
    sew_key_idxer = []
    sew_key_multiplier = []
    for e, g in enumerate(merge_groups):
        c = len(g)
        sew_key_idxer += [e] * c
        sew_key_multiplier += [1 / c]
    
    cloth.sew_edge_groups = merge_groups
    cloth.sew_key_idxer = np.array(sew_key_idxer, dtype=np.int32)
    cloth.sew_key_multiplier = np.array(sew_key_multiplier, dtype=np.float32)[:, None]
    cloth.basic_sew_verts = np.array([item for sublist in merge_groups for item in sublist], dtype=np.int32)
    cloth.basic_sew_adder = np.zeros((len(merge_groups), 3), dtype=np.float32)
    
    # for the cpp:
    cloth.sew_group_adder = np.zeros(len(merge_groups) * 3, dtype=np.float32)
    #cloth.sv_in_range = np.zeros(cloth.basic_sew_verts.shape[0], dtype=bool)
    
    edge_difs = cloth.co[regular_edges[:, 1]] - cloth.co[regular_edges[:, 0]]
    dists = U.measure_vecs(edge_difs)
    avg = np.mean(dists)
    cloth.basic_merge_limit = avg

    # target sewing
    sps = cloth.sew_pairs.shape[0]
    target_sew_multiplier = np.empty(cloth.sew_pairs.ravel().shape[0], dtype=np.float32)
    target_sew_multiplier[::2] = 1.0
    target_sew_multiplier[1::2] = -1.0
    cloth.target_sew_multiplier = target_sew_multiplier[:, None]
    cloth.target_sew_mover = np.empty((sps * 2, 3), dtype=np.float32)


def build_sew_hinges(cloth):
    """
    Build sew-hinge data using the same matrix formulation as regular bend.
    Supports:
      - MC_props.sew_bend          (bool enable)
      - MC_props.sew_bend_force    (float strength)
      - MC_props.sew_target_angle  (float degrees, 0 = open-flat)
      - MC_props.sew_bend_from_start (bool)
    """
    ob = cloth.ob
    obm = cloth.obm
    props = ob.MC_props

    # Early-out if disabled
    if not props.sew_bend or props.sew_bend_force == 0.0:
        cloth.sew_hinges = None
        return

    obm.verts.ensure_lookup_table()
    obm.edges.ensure_lookup_table()
    obm.faces.ensure_lookup_table()

    # ----- starting-state coordinates (same source as matrix_bend_data) -----
    tco = np.empty((len(ob.data.vertices), 3), dtype=np.float32)
    ob.data.shape_keys.key_blocks["MC_target"].data.foreach_get('co', tco.ravel())
    tco_cont = np.ascontiguousarray(tco.reshape(-1), dtype=np.float32)

    # face centers & normals from starting state
    U.compute_face_centers_cpp(
        cloth.centers_lib, tco_cont, cloth.face_verts, cloth.face_starts, cloth.fc, cloth.out_centers)
    face_normals = U.compute_face_normals_cpp(
        cloth.centers_lib, tco_cont, cloth.face_verts, cloth.face_starts, cloth.fc, cloth.out_normals)

    # ----- discover sew hinge candidates (your original logic) -----
    sew_pairs = cloth.sew_pairs
    regular_edge_set = set()
    for e in obm.edges:
        if len(e.link_faces) != 0:
            v0, v1 = e.verts[0].index, e.verts[1].index
            regular_edge_set.add((min(v0, v1), max(v0, v1)))

    vert_faces = {}
    for f in obm.faces:
        for v in f.verts:
            vert_faces.setdefault(v.index, []).append(f.index)

    sew_vert_to_idx = {v: k for k, v in enumerate(cloth.basic_sew_verts.tolist())}

    hinge_panel1_v0, hinge_panel1_v1 = [], []
    hinge_panel2_v0, hinge_panel2_v1 = [], []
    hinge_face1, hinge_face2 = [], []
    hinge_merged_idx = []

    M = len(sew_pairs)
    for i in range(M):
        A, A_ = int(sew_pairs[i, 0]), int(sew_pairs[i, 1])
        for j in range(i + 1, M):
            B, B_ = int(sew_pairs[j, 0]), int(sew_pairs[j, 1])
            for (p1v0, p1v1, p2v0, p2v1) in [(A, B, A_, B_), (A, B_, A_, B)]:
                e1 = (min(p1v0, p1v1), max(p1v0, p1v1))
                e2 = (min(p2v0, p2v1), max(p2v0, p2v1))
                if e1 not in regular_edge_set or e2 not in regular_edge_set:
                    continue
                f1_cands = set(vert_faces.get(p1v0, [])) & set(vert_faces.get(p1v1, []))
                f2_cands = set(vert_faces.get(p2v0, [])) & set(vert_faces.get(p2v1, []))
                if len(f1_cands) != 1 or len(f2_cands) != 1:
                    continue
                f1 = list(f1_cands)[0]
                f2 = list(f2_cands)[0]
                if (p1v0 not in sew_vert_to_idx or p1v1 not in sew_vert_to_idx or
                    p2v0 not in sew_vert_to_idx or p2v1 not in sew_vert_to_idx):
                    continue

                hinge_panel1_v0.append(p1v0)
                hinge_panel1_v1.append(p1v1)
                hinge_panel2_v0.append(p2v0)
                hinge_panel2_v1.append(p2v1)
                hinge_face1.append(f1)
                hinge_face2.append(f2)
                hinge_merged_idx.append([
                    sew_vert_to_idx[p1v0], sew_vert_to_idx[p1v1],
                    sew_vert_to_idx[p2v0], sew_vert_to_idx[p2v1]
                ])
                break

    if len(hinge_panel1_v0) == 0:
        cloth.sew_hinges = None
        return

    SH = cloth.sew_hinges = type('SewHinges', (), {})()
    SH.count = len(hinge_panel1_v0)

    SH.panel1_v0 = np.array(hinge_panel1_v0, dtype=np.int32)
    SH.panel1_v1 = np.array(hinge_panel1_v1, dtype=np.int32)
    SH.panel2_v0 = np.array(hinge_panel2_v0, dtype=np.int32)
    SH.panel2_v1 = np.array(hinge_panel2_v1, dtype=np.int32)
    SH.face1_idx = np.array(hinge_face1, dtype=np.int32)
    SH.face2_idx = np.array(hinge_face2, dtype=np.int32)
    SH.merged_idx = np.array(hinge_merged_idx, dtype=np.int32)

    # ----- canonical hinge-edge orientation -----
    # panel1_v0 / panel1_v1 come straight from bmesh edge order, which is
    # arbitrary, so u_hinge -- and with it the sign of the sin(theta) fold term
    # -- points either way at random from hinge to hinge, making the target
    # angle fold the wrong way on part of the seam.  Flip the pair so that
    # (face1_normal x edge) always points toward face1's centroid: every hinge
    # then has the same handedness and folds consistently.  Runtime C++ rebuilds
    # h from these same two indices, so build and runtime frames stay in sync.
    fc_centers = cloth.out_centers.reshape(cloth.fc, 3)
    _e0 = tco[SH.panel1_v0]
    _e1 = tco[SH.panel1_v1]
    _mid = 0.5 * (_e0 + _e1)
    _handed = np.einsum('ij,ij->i',
                        np.cross(face_normals[SH.face1_idx], _e1 - _e0),
                        fc_centers[SH.face1_idx] - _mid)
    _flip = _handed < 0.0
    if np.any(_flip):
        for _a, _b in ((SH.panel1_v0, SH.panel1_v1),
                       (SH.panel2_v0, SH.panel2_v1)):
            _t = _a[_flip].copy()
            _a[_flip] = _b[_flip]
            _b[_flip] = _t
        for _i, _j in ((0, 1), (2, 3)):
            _t = SH.merged_idx[_flip, _i].copy()
            SH.merged_idx[_flip, _i] = SH.merged_idx[_flip, _j]
            SH.merged_idx[_flip, _j] = _t

    # ----- build lf/rf vert lists (same pattern as regular) -----
    # Each face lists its OWN verts.  Relabelling face2's seam verts to face1's
    # looks like it welds the edge, but it makes panel1's seam verts absorb both
    # faces' incidences while panel2's absorb none -- so panel1's seam verts get
    # a bigger denominator in force_multiplier and their *regular* bend is
    # diluted relative to panel2's.  That asymmetric stiffness across the seam
    # pulls it toward the stiffer side.  Keep the counts symmetric instead.
    lf_verts_list, lf_tiler_list, lf_matrix_tiler_list = [], [], []
    rf_verts_list, rf_tiler_list, rf_matrix_tiler_list = [], [], []

    for h, f1, f2 in zip(range(SH.count), hinge_face1, hinge_face2):
        for v in obm.faces[f1].verts:
            lf_verts_list.append(v.index)
            lf_tiler_list.append(f1)
            lf_matrix_tiler_list.append(h)
        for v in obm.faces[f2].verts:
            rf_verts_list.append(v.index)
            rf_tiler_list.append(f2)
            rf_matrix_tiler_list.append(h)

    SH.lf_verts = np.array(lf_verts_list, dtype=np.int32)
    SH.lf_tiler = np.array(lf_tiler_list, dtype=np.int32)
    SH.lf_matrix_tiler = np.array(lf_matrix_tiler_list, dtype=np.int32)
    SH.rf_verts = np.array(rf_verts_list, dtype=np.int32)
    SH.rf_tiler = np.array(rf_tiler_list, dtype=np.int32)
    SH.rf_matrix_tiler = np.array(rf_matrix_tiler_list, dtype=np.int32)
    SH.lf_total = len(lf_verts_list)
    SH.rf_total = len(rf_verts_list)

    # Force multiplier is filled in by combine_bend_multipliers() so the sew
    # hinges share ONE incidence count with the regular hinges (see there).
    # Placeholder until that runs, in case anything reads it early.
    SH.force_multiplier = np.zeros(cloth.vc, dtype=np.float32)

    SH.force_accumulator = np.zeros(cloth.vc * 3, dtype=np.float32)

    # ----- MATRIX / SCALERS (core of the new path) -----
    hc = SH.count
    hinge_eidx = np.stack([SH.panel1_v0, SH.panel1_v1], axis=1)  # use panel1 as the hinge axis

    hinge_co = tco[hinge_eidx]                    # starting-state positions
    hinge_origins = hinge_co[:, 0]
    hinge_vecs = hinge_co[:, 1] - hinge_origins
    u_hinge = U.u_vecs_out(hinge_vecs)            # your existing unit-vector helper

    lf_normals = face_normals[SH.face1_idx]
    rf_normals = face_normals[SH.face2_idx]

    lf_matrix = np.empty((hc, 3, 3), dtype=np.float32)
    rf_matrix = np.empty((hc, 3, 3), dtype=np.float32)

    lf_matrix[:, 0] = u_hinge
    rf_matrix[:, 0] = u_hinge
    lf_matrix[:, 1] = lf_normals
    rf_matrix[:, 1] = rf_normals
    lf_matrix[:, 2] = U.fastest_cross_product(u_hinge, lf_normals)
    rf_matrix[:, 2] = U.fastest_cross_product(u_hinge, rf_normals)

    if props.sew_bend_from_start:
        # Exact same measurement as regular bend
        SH.lf_normal_scalers = np.einsum('vij,vj->vi', rf_matrix, lf_normals)
        SH.rf_normal_scalers = np.einsum('vij,vj->vi', lf_matrix, rf_normals)
    else:
        # User angle (degrees) → open-flat at 0°
        theta = np.deg2rad(props.sew_target_angle)

        # Construct desired normals for open-flat reference + rotation
        # For θ = 0 we want normals parallel (continuous surface).
        # We rotate the *right* normal around the hinge relative to the left.
        cos_t = np.atleast_1d(np.cos(theta)).astype(np.float32)
        sin_t = np.atleast_1d(np.sin(theta)).astype(np.float32)

        # Desired right normal expressed in world = cosθ * lf_n + sinθ * (h × lf_n)
        # (sign of sin chosen so positive angle folds in the conventional cloth direction)
        h_cross_ln = U.fastest_cross_product(u_hinge, lf_normals)
        desired_rf_normal = cos_t[:, None] * lf_normals + sin_t[:, None] * h_cross_ln

        # Now express the *original* left normal in the right frame that would
        # be built from the desired right normal, and vice-versa.
        # Simplest stable approach: build a temporary right matrix with the desired normal
        # and project.
        rf_matrix_des = rf_matrix.copy()
        rf_matrix_des[:, 1] = desired_rf_normal
        rf_matrix_des[:, 2] = U.fastest_cross_product(u_hinge, desired_rf_normal)

        SH.lf_normal_scalers = np.einsum('vij,vj->vi', rf_matrix_des, lf_normals)
        SH.rf_normal_scalers = np.einsum('vij,vj->vi', lf_matrix, desired_rf_normal)

    # store strength so C++ can use it
    SH.sew_bend_force = np.float32(props.sew_bend_force)

    # share one incidence count with the regular hinges (no-op if MB isn't built
    # yet -- cloth_setup calls this again right after matrix_bend_data)
    combine_bend_multipliers(cloth)


def combine_bend_multipliers(cloth):
    """One force_multiplier for BOTH bend solvers.

    compute_bend_forces and compute_sew_bend_forces each divide a vertex's
    accumulated force by that vertex's multiplier.  While they were counted
    separately, a seam vertex was normalised by its regular-hinge count in one
    pass and by its sew-hinge count in the other, so it received
        regular_move / regular_count  +  sew_move / sew_count
    while a vertex one ring inside received only the first term.  The seam is
    then corrected harder than the surface around it every iteration, and that
    imbalance is what walks the seam.

    Counting every hinge-face incidence once -- regular and sew together --
    normalises the seam exactly like the rest of the mesh, which is all the
    regular solver ever needed to stay put.
    """
    MB = getattr(cloth, "MB", None)
    if MB is None:
        return
    SH = getattr(cloth, "sew_hinges", None)
    has_sew = SH is not None and getattr(SH, "count", 0) > 0

    parts = [MB.lf_verts, MB.rf_verts]
    if has_sew:
        parts += [SH.lf_verts, SH.rf_verts]

    counts = np.bincount(np.concatenate(parts), minlength=cloth.vc).astype(np.float32)
    mult = np.zeros(cloth.vc, dtype=np.float32)
    nz = counts > 0.0
    mult[nz] = 1.0 / counts[nz]

    MB.force_multiplier = mult[:, None]
    if has_sew:
        SH.force_multiplier = mult.copy()


def bend_count_report(cloth, verbose=True):
    """Sanity check on the bend force_multiplier incidence counts.

    force_multiplier[v] = 1 / (number of hinge-face incidences listing v), i.e.
    'average the corrections that want to move v' rather than sum them.  Every
    vertex that takes part should therefore end up with a TOTAL applied weight
    of exactly 1.0.  Anything else means that vertex is corrected harder (or
    softer) than the surface around it, which shows up as creep.

    Returns a dict; prints a summary when verbose.
    """
    MB = getattr(cloth, "MB", None)
    if MB is None:
        print("bend_count_report: no MB yet")
        return None
    SH = getattr(cloth, "sew_hinges", None)
    has_sew = SH is not None and getattr(SH, "count", 0) > 0
    vc = cloth.vc

    def bc(*arrays):
        if not arrays:
            return np.zeros(vc, dtype=np.int64)
        return np.bincount(np.concatenate(arrays), minlength=vc)

    reg = bc(MB.lf_verts, MB.rf_verts)
    sew = bc(SH.lf_verts, SH.rf_verts) if has_sew else np.zeros(vc, dtype=np.int64)
    total = reg + sew

    mult = np.zeros(vc, dtype=np.float64)
    nz = total > 0
    mult[nz] = 1.0 / total[nz]
    weight = total * mult                      # must be 1.0 wherever nz

    # verts that belong to a face but get no bend at all
    in_face = np.zeros(vc, dtype=bool)
    for f in cloth.obm.faces:
        for v in f.verts:
            in_face[v.index] = True
    orphans = np.nonzero(in_face & ~nz)[0]

    # seam symmetry: each sew pair should have matching total counts
    asym = []
    if has_sew and getattr(cloth, "sew_pairs", None) is not None:
        for a, b in cloth.sew_pairs:
            if total[a] != total[b]:
                asym.append((int(a), int(b), int(total[a]), int(total[b])))

    out = {"regular": reg, "sew": sew, "total": total, "multiplier": mult,
           "weight": weight, "orphans": orphans, "asymmetric_pairs": asym}

    if verbose:
        print("# ===== bend count report =====")
        print("  verts %d   regular hinges %d   sew hinges %d"
              % (vc, MB.hc, SH.count if has_sew else 0))
        print("  incidences  regular %d   sew %d" % (reg.sum(), sew.sum()))
        if nz.any():
            print("  count/vert  min %d  max %d  mean %.2f"
                  % (total[nz].min(), total[nz].max(), total[nz].mean()))
        bad_w = np.nonzero(nz & (np.abs(weight - 1.0) > 1e-9))[0]
        print("  applied weight != 1.0 : %d vert(s)  %s"
              % (bad_w.size, "OK" if bad_w.size == 0 else bad_w[:10]))
        print("  in a face but no bend : %d vert(s)  %s"
              % (orphans.size, "OK" if orphans.size == 0 else orphans[:10]))
        if has_sew:
            sew_only = np.nonzero((sew > 0) & (reg == 0))[0]
            both = np.nonzero((sew > 0) & (reg > 0))[0]
            print("  sew verts: %d also in regular hinges, %d sew-only"
                  % (both.size, sew_only.size))
            print("  asymmetric sew pairs  : %d  %s"
                  % (len(asym), "OK" if not asym else asym[:6]))
            if both.size:
                print("  seam mult mean %.4f vs interior mult mean %.4f"
                      % (mult[both].mean(),
                         mult[nz & (sew == 0)].mean() if (nz & (sew == 0)).any() else float('nan')))
        print("# ============================")
    return out


def build_sew_hinges__(cloth):
    """
    Identify virtual hinge pairs from sew edges.
    For every pair of sew edges [A,A'] and [B,B'] where:
      - a regular edge [A,B] exists on panel 1
      - a regular edge [A',B'] exists on panel 2
    ...store as a sew hinge candidate.
    Active when all four verts are merged (C.merged).
    """
    obm = cloth.obm
    obm.verts.ensure_lookup_table()
    obm.edges.ensure_lookup_table()
    obm.faces.ensure_lookup_table()

    sew_pairs = cloth.sew_pairs  # (M, 2) int32, no-face edges

    # Build a set of regular edges for fast lookup
    regular_edge_set = set()
    for e in obm.edges:
        if len(e.link_faces) != 0:
            v0, v1 = e.verts[0].index, e.verts[1].index
            regular_edge_set.add((min(v0,v1), max(v0,v1)))

    # Build lookup: vert index -> which face(s) it's in
    # (for finding the adjacent face to a boundary edge)
    vert_faces = {}
    for f in obm.faces:
        for v in f.verts:
            if v.index not in vert_faces:
                vert_faces[v.index] = []
            vert_faces[v.index].append(f.index)

    # Build lookup: vert index -> position in basic_sew_verts (for C.merged indexing)
    sew_vert_to_idx = {v: k for k, v in enumerate(cloth.basic_sew_verts.tolist())}

    # For each pair of sew edges, check if their endpoints
    # are connected by regular edges on each panel
    M = len(sew_pairs)

    hinge_panel1_v0 = []  # A
    hinge_panel1_v1 = []  # B
    hinge_panel2_v0 = []  # A'
    hinge_panel2_v1 = []  # B'
    hinge_face1 = []      # face index adjacent to [A,B]
    hinge_face2 = []      # face index adjacent to [A',B']
    hinge_merged_idx = [] # 4 indices into basic_sew_verts: [k_A, k_A', k_B, k_B']

    for i in range(M):
        A,  A_ = int(sew_pairs[i, 0]), int(sew_pairs[i, 1])
        for j in range(i+1, M):
            B,  B_ = int(sew_pairs[j, 0]), int(sew_pairs[j, 1])

            # Check both orientations: (A-B on p1, A'-B' on p2)
            # and crossed (A-B' on p1, A'-B on p2)
            for (p1v0, p1v1, p2v0, p2v1) in [
                (A, B, A_, B_),
                (A, B_, A_, B)
            ]:
                e1 = (min(p1v0, p1v1), max(p1v0, p1v1))
                e2 = (min(p2v0, p2v1), max(p2v0, p2v1))

                if e1 not in regular_edge_set: continue
                if e2 not in regular_edge_set: continue

                # Find the single adjacent face for each boundary edge
                f1_candidates = set(vert_faces.get(p1v0, [])) & set(vert_faces.get(p1v1, []))
                f2_candidates = set(vert_faces.get(p2v0, [])) & set(vert_faces.get(p2v1, []))

                if len(f1_candidates) != 1: continue  # skip bad topology
                if len(f2_candidates) != 1: continue

                f1 = list(f1_candidates)[0]
                f2 = list(f2_candidates)[0]

                # Make sure all four verts are in basic_sew_verts
                # (they should be, but guard against edge cases)
                if p1v0 not in sew_vert_to_idx: continue
                if p1v1 not in sew_vert_to_idx: continue
                if p2v0 not in sew_vert_to_idx: continue
                if p2v1 not in sew_vert_to_idx: continue

                hinge_panel1_v0.append(p1v0)
                hinge_panel1_v1.append(p1v1)
                hinge_panel2_v0.append(p2v0)
                hinge_panel2_v1.append(p2v1)
                hinge_face1.append(f1)
                hinge_face2.append(f2)
                hinge_merged_idx.append([
                    sew_vert_to_idx[p1v0],
                    sew_vert_to_idx[p1v1],
                    sew_vert_to_idx[p2v0],
                    sew_vert_to_idx[p2v1],
                ])
                break  # found valid orientation for this pair, move on

    if len(hinge_panel1_v0) == 0:
        cloth.sew_hinges = None
        return

    SH = cloth.sew_hinges = type('SewHinges', (), {})()
    SH.count = len(hinge_panel1_v0)

    # Panel edge verts -- these define the virtual hinge axis
    # hinge axis = mean of (p1v0+p2v0)/2 -> (p1v1+p2v1)/2
    # but since they're coincident when active, p1v0 == p2v0 positionally
    SH.panel1_v0 = np.array(hinge_panel1_v0, dtype=np.int32)
    SH.panel1_v1 = np.array(hinge_panel1_v1, dtype=np.int32)
    SH.panel2_v0 = np.array(hinge_panel2_v0, dtype=np.int32)
    SH.panel2_v1 = np.array(hinge_panel2_v1, dtype=np.int32)
    SH.face1_idx = np.array(hinge_face1, dtype=np.int32)
    SH.face2_idx = np.array(hinge_face2, dtype=np.int32)

    # 4 indices per hinge into basic_sew_verts, for C.merged lookup
    SH.merged_idx = np.array(hinge_merged_idx, dtype=np.int32)  # (N_hinges, 4)

    # Per-vertex force accumulation -- reused each frame
    SH.force_accumulator = np.zeros(cloth.vc * 3, dtype=np.float32)

    # Precompute which verts of each panel face need forces applied.
    # Same lf_verts/rf_verts pattern as regular bend solver.
    lf_verts_list = []
    lf_tiler_list = []
    lf_matrix_tiler_list = []
    rf_verts_list = []
    rf_tiler_list = []
    rf_matrix_tiler_list = []

    for h, f1, f2 in zip(range(SH.count), hinge_face1, hinge_face2):
        for v in obm.faces[f1].verts:
            lf_verts_list.append(v.index)
            lf_tiler_list.append(f1)
            lf_matrix_tiler_list.append(h)
        for v in obm.faces[f2].verts:
            rf_verts_list.append(v.index)
            rf_tiler_list.append(f2)
            rf_matrix_tiler_list.append(h)

    SH.lf_verts = np.array(lf_verts_list, dtype=np.int32)
    SH.lf_tiler = np.array(lf_tiler_list, dtype=np.int32)
    SH.lf_matrix_tiler = np.array(lf_matrix_tiler_list, dtype=np.int32)
    SH.rf_verts = np.array(rf_verts_list, dtype=np.int32)
    SH.rf_tiler = np.array(rf_tiler_list, dtype=np.int32)
    SH.rf_matrix_tiler = np.array(rf_matrix_tiler_list, dtype=np.int32)
    SH.lf_total = len(lf_verts_list)
    SH.rf_total = len(rf_verts_list)
    
    #all_verts = np.concatenate([SH.lf_verts, SH.rf_verts])
    #v_counts = np.bincount(all_verts, minlength=cloth.vc)# // 4
    #v_counts = np.bincount(SH.lf_verts, minlength=cloth.vc)
    #SH.force_multiplier = np.where(v_counts > 0, 1.0 / np.maximum(v_counts, 1), 0.0).astype(np.float32)[:, None]
    #SH.force_multiplier = v_counts
    #SH.force_multiplier = np.ones((cloth.vc, 1), dtype=np.float32) * 2

    # Why couldn't Grok or Misanthropic figure this out?:
    mult = np.zeros(cloth.vc, dtype=np.float32)
    for e in cloth.obm.edges:
        if len(e.link_faces) == 0:
            v0 = e.verts[0]
            v1 = e.verts[1]
            mult[v0.index] += 0.5
            mult[v1.index] += 0.5
    SH.force_multiplier = mult        
            


def get_max_radius(C):
    """Per-vertex self collision radius: how far a vertex can swell before it
    reaches the edge opposite it in a triangle it belongs to.

    A vertex in no triangle has no opposing edge to measure against -- a loose
    vertex, or every vertex on a mesh with no faces.  Those have to be left out
    of the result rather than measured as zero: sc_radius is the MINIMUM over
    this array, so a single loose vertex would drag it to nothing and quietly
    turn self collision off.  They take the largest real radius instead, which
    keeps them out of the way of the minimum.
    """
    sco = C.source_co
    tridex = C.tridex
    max_radius = np.zeros(sco.shape[0], dtype=np.float32)
    found = np.zeros(sco.shape[0], dtype=bool)

    for v in C.idxer:
        in_tris = tridex == v
        where = np.any(in_tris, axis=1)
        checks = tridex[where]
        in_tris.shape = tridex.shape
        #if v == 36:
        edges = checks[~in_tris[where]]
        eco = sco[edges]
        evecs = eco[1::2] - eco[::2]
        orivecs = sco[v] - eco[::2]
        uevecs = U.u_vecs(evecs)
        dots = U.compare_vecs(orivecs, uevecs)
        cpoe = uevecs * dots[:, None]
        rad_vecs = orivecs - cpoe
        rads = U.measure_vecs(rad_vecs)
        if rads.shape[0] == 0:
            continue                # no opposing edge, so nothing to measure
        max_radius[v] = np.min(rads)
        found[v] = True

    if np.any(found):
        max_radius[~found] = max_radius[found].max()

    C.max_radius = max_radius[:, None] * 0.98
    # whether anything could actually be measured, so the caller knows not to
    # take a minimum over a meaningless array
    C.max_radius_found = int(np.count_nonzero(found))


def get_spring_edges(ob):
    """Gets the minimum number of edges
    to connect every vertex to every
    other vertex in each face."""
    mesh = ob.data
    spring_pairs = set()
    
    # Iterate over all polygons (faces)
    for poly in mesh.polygons:
        verts = poly.vertices
        # Generate all unique pairs within the face
        for i in range(len(verts)):
            for j in range(i + 1, len(verts)):
                v1, v2 = sorted([verts[i], verts[j]])
                spring_pairs.add((v1, v2))
    
    springs = np.array(sorted(list(spring_pairs)), dtype=np.int32)
    
    return springs


def linear_springs_no_push(C, stretch=None, shrink_grow=1.0):

    np.take(C.co, C.es_0, axis=0, out=C.p0)
    np.take(C.co, C.es_1, axis=0, out=C.p1)
    v = C.p0 - C.p1
    d = np.linalg.norm(v, axis=1)
    scale = ((C.target_dists * shrink_grow) / d - 1.0)
    
    long = d <= scale
    move = scale[:, None] * v
    move[long] = 0.0
    
    C.sum_positions[:] = 0
    np.add.at(C.sum_positions, C.es_0, move * C.right_group_mult)
    np.subtract.at(C.sum_positions, C.es_1, move * C.left_group_mult)
    C.co += np.nan_to_num(C.sum_positions * C.es_mult * stretch)


def linear_springs(C, stretch=None, shrink_grow=1.0):
    """`shrink_grow` scales every spring's rest length; the caller reads the
    property, so both solvers are handed the same number."""
    np.take(C.co, C.es_0, axis=0, out=C.p0)
    np.take(C.co, C.es_1, axis=0, out=C.p1)
    v = C.p0 - C.p1
    d = np.linalg.norm(v, axis=1)
    scale = ((C.target_dists * shrink_grow) / d - 1.0)[:, None]
    move = scale * v
    C.sum_positions[:] = 0
    np.add.at(C.sum_positions, C.es_0, move * C.right_group_mult)
    np.subtract.at(C.sum_positions, C.es_1, move * C.left_group_mult)
    C.co += np.nan_to_num(C.sum_positions * C.es_mult * stretch)


def basic_sew_force(cloth):
    """Basic edge sewing that uses edges with no link faces.
    Takes into account intersections between multiple sew
    lines and pulls towards the mean."""

    target = cloth.ob.MC_props.target_sew_length
    if target > 0:
        
        end = cloth.co[cloth.sew_pairs[:, 1]]
        dif = end - cloth.co[cloth.sew_pairs[:, 0]]
        dist = U.measure_vecs(dif)
        u_dif = dif / dist[:, None]
        t_vecs = u_dif * target
        move = dif - t_vecs
        
        np.add.at(cloth.co, cloth.sew_pairs[:, 0], move * 0.5 * cloth.ob.MC_props.sew_force)
        np.subtract.at(cloth.co, cloth.sew_pairs[:, 1], move * 0.5 * cloth.ob.MC_props.sew_force)
        return
            
    cloth.basic_sew_adder[:] = 0.0
    
    sew_verts = cloth.co[cloth.basic_sew_verts]
    np.add.at(cloth.basic_sew_adder, cloth.sew_key_idxer, sew_verts)
    cloth.basic_sew_adder *= cloth.sew_key_multiplier
    dif = cloth.basic_sew_adder[cloth.sew_key_idxer] - sew_verts
    dist = U.measure_vecs(dif)
    in_range = dist < (cloth.basic_merge_limit * cloth.ob.MC_props.butt_sew_force)
        
    cloth.co[cloth.basic_sew_verts[~in_range]] += dif[~in_range] * cloth.ob.MC_props.sew_force
    cloth.co[cloth.basic_sew_verts[in_range]] += dif[in_range]

    return in_range


def magnetic_force_grok(C, mode=None, k=5):
    """
    mode : None | "low" | "high"
        None – classic single nearest neighbor
        "low"  – among the k nearest, pick the one with the smallest
                 projection onto the normal
        "high" – among the k nearest, pick the one with the largest
                 projection onto the normal
    k : int
        Number of nearest neighbors to examine when mode is "low"/"high".
        Ignored when mode == "none".
    """
    ob = C.ob
    co = np.nan_to_num(C.co)
    norms = U.get_vertex_normals_np(ob, co)


    strength = ob.MC_props.magnetic_force
    range_val = ob.MC_props.magnetic_range

    mco = C.mco#[in_box]

    tree = C.tree
    n_points = len(mco)
    
    if mode is None:
        dists, neighs = tree.query(co, k=1, distance_upper_bound=range_val)
        dists = dists[:, None]
        neighs = neighs[:, None]
    else:
        dists, neighs = tree.query(co, k=k, distance_upper_bound=range_val)

    if dists.ndim == 1:
        dists = dists[:, None]
        neighs = neighs[:, None]

    # vertices that have at least one neighbor inside the range
    valid = np.any(dists < range_val, axis=1)
    if not np.any(valid):
        return

    co_v     = co[valid]
    norms_v  = norms[valid]
    dists_v  = dists[valid]
    neighs_v = neighs[valid]

    # mask of in-range neighbors
    in_range = dists_v < range_val

    # replace out-of-range indices (value == n_points) with 0 so indexing is safe
    safe_neighs = np.where(neighs_v < n_points, neighs_v, 0)

    candidates = mco[safe_neighs]                 # (nv, k, 3)
    difs = candidates - co_v[:, None, :]          # (nv, k, 3)
    dots = np.einsum("...j,...j->...", difs, norms_v[:, None, :])  # (nv, k)

    # discard out-of-range projections
    dots = np.where(in_range, dots, np.inf if mode == "low" else -np.inf)

    if mode is None:
        chosen = dots[:, 0]
    elif mode == "low":
        chosen = dots.min(axis=1)
    else:  # "high"
        chosen = dots.max(axis=1)

    # apply force
    idx = np.nonzero(valid)[0]
    C.co[idx] += norms_v * chosen[:, None] * strength

    # axis locks
    if C.ob.MC_props.lock_axis_x:
        C.co[:, 0] = C.start_co[:, 0]
    if C.ob.MC_props.lock_axis_y:
        C.co[:, 1] = C.start_co[:, 1]
    if C.ob.MC_props.lock_axis_z:
        C.co[:, 2] = C.start_co[:, 2]


def update_pin_select_sew(C, store_merge=False):
    C.merged = basic_sew_force(C)
    apply_pins(C)


def apply_pins(C):
    dif = C.start_co - C.co
    dif *= C.group_data["MC_pin"]
    C.co += dif
    # pin_selected is C.selected minus hooked verts, set once per step in
    # physics(); falls back to C.selected for callers outside a step
    sel = getattr(C, "pin_selected", C.selected)
    C.co[sel] = C.start_co[sel]


# ===== RECOLLIDE ===== #
# Bend and stretch iterate several times a step, each iteration pulling the
# cloth toward its rest shape.  Collision runs once, after all of them.  With a
# stiff cloth on a sharp point the iterations drag the contact verts down past
# the tip, and the single collision pass at the end shoves only those verts
# back up -- a spike -- or, once they are dragged far enough, loses the contact
# and lets the tip through.
#
# Recollide re-resolves the contacts found by the last full collision pass
# between the solver iterations, so the next iteration spreads the correction
# to the neighbours instead.  It is positional only: no friction, no velocity
# feedback -- the full collision pass at the end of the step still does those.
#
# Where the pairs come from, without a broad phase inside the step:
#   * The end-of-step collision pass runs its broad phase with every box
#     stretched along that element's motion into the NEXT step (lookahead),
#     and caches the result.  Pairs the cloth is about to reach are already in
#     it, so the step it first touches a collider is covered.  The pass itself
#     still resolves exactly the unpadded pairs, so its result is unchanged.
#   * At the start of the next step, after velocity has moved the cloth,
#     prepare_recollide trims that cache to the pairs close to touching.  The
#     cache is everything the broad phase could not rule out (~8,800 pairs on
#     the cone test, a handful touching), and recollide pays for each pair on
#     every iteration, so this is where the time goes.
#   * With no usable cache (first step, colliders changed) it runs the broad
#     phase once for this step instead.

def prepare_recollide(C):
    """Called once a step, after velocity is applied and before bend.
    Returns True when the solver loops should stop to recollide."""
    C.recollide_pairs = None
    OC.begin_recollide(C)
    # a new step: the C++ bridge re-packs its arrays (see collide_native._oc_mesh)
    C._native_step = getattr(C, "_native_step", 0) + 1
    if not C.ob.MC_props.ob_recollide or not OC_DATA["obs"]:
        C.oc_lookahead = None
        return False
    if not refresh_colliders(C):
        return False
    recall = getattr(C, "recall_data", None)
    if recall is None or getattr(C, "recall_version", None) != OC_DATA.get("version"):
        recall = OC.find_pairs(OC_DATA, C, C.start_co, C.co)
    pairs = OC.near_pairs(OC_DATA, C, *recall)
    if all(p is None for p in pairs):
        return False
    C.recollide_pairs = pairs
    return True


def solver_recollide(C, iteration):
    """Re-resolve the step's near pairs after solver iteration `iteration`
    (counted from 0 across bend then stretch), on every Nth one."""
    every = max(1, int(getattr(C.ob.MC_props, "ob_recollide_every", 1)))
    if (iteration + 1) % every:
        return
    point_tri, edge_edge, tri_point = C.recollide_pairs
    OC.recollide(OC_DATA, C, point_tri, edge_edge, tri_point)
    apply_pins(C)          # a contact must not drag a pinned vert
# ===== RECOLLIDE ===== #

    
def pin_group(C):
    dif = C.start_co - C.co
    dif *= C.pin_group
    C.co += dif


def cloth_refresh(ob):
    get_cloth(ob, clear=True)
    C = get_cloth(ob, start=True)
    if C is None:
        return
    
#    target = C.ob.MC_props.target_object
#    if target is not None:
#        kitten_mana = np.arange(C.vc)
#        U.set_named_int_attribute(target, kitten_mana, name="Kitten Mana", type="INT", domain="POINT")    
#        target.data.update()
         
    C.obm = U.get_bmesh(ob)
    install_handler()
    return C


# ===== TARGET EDITING ===== #
# The target mesh is what the user edits while the sim keeps running.  When its
# topology changes we have to answer two questions: which verts are the ones we
# were already simulating, and where do the new ones belong in the *simulated*
# cloth (not in the target's rest shape).
#
# Identity: every target vert carries an id in a custom int attribute.  Blender
# carries that attribute through knife, extrude, subdivide, bisect, join and
# edit-mode operators, and a vert created by any of them gets a *copy* of a
# neighbour's id (or 0 for geometry joined in from a mesh that never had the
# attribute).  So after an edit:
#   * an id seen once, that we issued before  -> the vert we already had
#   * an id seen more than once              -> one original, the rest are new
#   * an id we never issued                  -> new
# Ids are re-issued after every change so they stay unique.  This used to be
# read off the selection instead, which only held when the user's selection was
# exactly the new geometry and nothing else.
MANA = "Kitten Mana"


def stamp_mana(ob, count=None):
    """Give every vert of `ob` a fresh unique id."""
    if count is None:
        count = len(ob.data.vertices)
    U.set_named_int_attribute(ob, np.arange(count, dtype=np.int32),
                              name=MANA, type="INT", domain="POINT")


def read_mana(mesh):
    """The ids of a mesh, or None if it was never stamped."""
    if MANA not in mesh.attributes:
        return None
    try:
        return U.get_named_int_attribute_mesh(mesh, name=MANA)
    except Exception:
        return None


def map_target_verts(ids, vc, new_co=None, old_co=None):
    """Follow verts through an edit.

    `ids` are the new mesh's ids, `vc` how many verts we were simulating (ids
    0..vc-1 were the ones we issued).  Returns an int array, one entry per new
    vert, holding the index it had in the sim, or -1 when it is new.

    When an id is duplicated, the copy nearest to where that vert used to be is
    taken as the original -- the others were created from it.  Without
    positions to compare, the first one wins, which is what Blender's own
    ordering gives us anyway.
    """
    new_to_old = np.full(ids.shape[0], -1, dtype=np.int32)
    known = (ids >= 0) & (ids < vc)
    idx = np.arange(ids.shape[0])

    order = np.argsort(ids[known], kind="stable")
    kidx = idx[known][order]
    kids = ids[known][order]
    starts = np.searchsorted(kids, kids, side="left")
    ends = np.searchsorted(kids, kids, side="right")

    seen = set()
    for s, e in zip(starts, ends):
        vid = int(kids[s])
        if vid in seen:
            continue
        seen.add(vid)
        group = kidx[s:e]
        if group.shape[0] == 1:
            new_to_old[group[0]] = vid
            continue
        pick = group[0]
        if new_co is not None and old_co is not None and vid < old_co.shape[0]:
            d = np.einsum('ij,ij->i', new_co[group] - old_co[vid],
                          new_co[group] - old_co[vid])
            pick = group[int(np.argmin(d))]
        new_to_old[pick] = vid
    return new_to_old


def _newell_normals(tar_obm, fidx, co):
    """Face normals of `fidx` from the coordinates `co`, which are laid out on
    the new topology.  Taken from the bmesh rather than a mesh datablock: the
    cloth-space normals used to be read from a mesh built on the *old*
    topology with new face indices, which silently picked the wrong faces as
    soon as any face index shifted."""
    out = np.zeros((len(fidx), 3), dtype=np.float32)
    for e, fi in enumerate(fidx):
        vids = [v.index for v in tar_obm.faces[fi].verts]
        pts = co[vids]
        rolled = np.roll(pts, -1, axis=0)
        n = np.cross(pts, rolled).sum(axis=0)
        ln = np.linalg.norm(n)
        out[e] = n / ln if ln > 0.0 else (0.0, 0.0, 1.0)
    return out


def find_anchors(tar_obm, new_verts, old_mask):
    """For each new vert, the old faces to plot it from.

    Walks outward from the vert until it reaches faces made only of verts we
    were already simulating, and picks the edge of each such face that borders
    the new geometry -- the edge nearest the new vert.  Verts on an island with
    no old geometry at all get nothing, and are left where the target puts
    them.
    """
    tar_obm.verts.ensure_lookup_table()
    tar_obm.faces.ensure_lookup_table()

    old_faces = np.array([bool(old_mask[[v.index for v in f.verts]].all())
                          for f in tar_obm.faces], dtype=bool)

    anchors = {}
    v_mask = np.zeros(len(tar_obm.verts), dtype=bool)
    vidx = np.arange(v_mask.shape[0])

    for v in new_verts:
        v_mask[:] = False
        v_mask[v] = True
        reached = 1
        while True:
            v_mask = U.grow(tar_obm, sel=v_mask, iters=1)
            grown = np.count_nonzero(v_mask)
            if grown <= reached:
                break                      # an island: nothing old to hang it on
            reached = grown

            linked_faces = []
            for nv in vidx[v_mask]:
                linked_faces += [lf.index for lf in tar_obm.verts[nv].link_faces]
            if not linked_faces:
                break
            linked_old = old_faces[linked_faces]
            if not linked_old.any():
                continue

            lfidx = np.array(linked_faces)[linked_old]
            edges = []
            for lfi in lfidx:
                # the edge where an old face meets a new one
                common = tar_obm.faces[lfi].edges[0]
                for edge in tar_obm.faces[lfi].edges:
                    lf = [elf.index for elf in edge.link_faces]
                    if len(lf) == 2 and np.count_nonzero(old_faces[lf]) == 1:
                        common = edge
                edges.append([common.verts[0].index, common.verts[1].index])
            anchors[int(v)] = (lfidx, np.array(edges, dtype=np.int64))
            break
    return anchors


def _face_frame(tar_obm, fi, co):
    """A face's corner and the 3x3 it spans: two edges of it plus its normal.

    Its columns are what a point near the face is measured against, so the
    matrix that carries one frame onto another carries the point with it."""
    verts = [v.index for v in tar_obm.faces[fi].verts]
    a, b, c = co[verts[0]], co[verts[1]], co[verts[2]]
    n = np.cross(b - a, c - a)
    ln = np.linalg.norm(n)
    if ln < 1e-12:
        return a, None
    m = np.empty((3, 3), dtype=np.float64)
    m[:, 0] = b - a
    m[:, 1] = c - a
    m[:, 2] = n / ln                    # unit, so thickness is kept as-is
    return a, m


def plot_new_verts(anchors, tco, cco, tar_obm):
    """Where the new verts go in the running sim.

    For each old face near the new vert, the map that takes that face in the
    *target* onto the same face in the simulated cloth is the one that takes
    the two together with their normal -- the face's deformation.  The vert is
    carried by that map, so a vert cut into the middle of an edge stays in the
    middle of it however the cloth has been stretched, sheared or turned since.
    A frame built on a single edge would only follow stretch along that edge:
    a vert cut across the pull would end up in the wrong place.

    `tco` is the target's coordinates, `cco` the sim's, both laid out on the
    new topology with the new verts still at their target positions.  Returns
    {vert index: position}; verts with no anchor faces are left out.
    """
    out = {}
    for v, (fidx, _eidx) in anchors.items():
        plots = []
        for fi in fidx:
            t_origin, t_m = _face_frame(tar_obm, fi, tco)
            c_origin, c_m = _face_frame(tar_obm, fi, cco)
            if t_m is None or c_m is None:
                continue
            try:
                local = np.linalg.solve(t_m, tco[v] - t_origin)
            except np.linalg.LinAlgError:
                continue                # a degenerate face teaches us nothing
            plots.append(c_origin + c_m @ local)
        if plots:
            out[v] = np.mean(plots, axis=0).astype(np.float32)
    return out


# What a rebuild has to carry over.  cloth_setup builds a cloth from the mesh
# and starts it at rest, so the velocity of the verts we were already
# simulating would be lost on every edit -- the sim visibly stopped and
# restarted.  Stashed here by name, applied once the new cloth exists.
_CARRY = {}


def carry_over(ob, velocity):
    """`velocity` is already laid out on the new topology."""
    _CARRY[ob.name] = velocity


def apply_carry(C):
    """Put the simulation state back after a rebuild."""
    velocity = _CARRY.pop(C.ob.name, None)
    if velocity is None or velocity.shape[0] != C.vc:
        return
    C.velocity[:] = velocity
    C.vel_start[:] = C.co


def retopo_from_target(C, target, tmesh):
    """Rebuild the cloth on the target's new topology, keeping the sim running.

    Verts we were already simulating keep their simulated position and
    velocity.  New ones are plotted into the running cloth from the target
    (see plot_new_verts) and start out moving with the cloth around them, so
    nothing pops.
    """
    tar_obm = U.get_mesh_bmesh(tmesh)
    tco = U.get_mesh_co(tmesh)
    n = tco.shape[0]

    ids = read_mana(tmesh)
    if ids is None:
        # never stamped (an old file, or the target was swapped): nothing to
        # follow, so take the target as it is and start from there
        U.set_bmesh(C.ob, tar_obm)
        C.ob.data.shape_keys.key_blocks["MC_current"].data.foreach_set('co', tco.ravel())
        C.ob.data.update()
        stamp_mana(target, n)
        return True

    new_to_old = map_target_verts(ids, C.vc, tco, C.pco)
    old = new_to_old >= 0
    new_verts = np.arange(n)[~old]

    # positions: old verts keep the sim, new ones start at the target
    co = tco.copy()
    co[old] = C.co[new_to_old[old]]

    anchors = find_anchors(tar_obm, new_verts, old)
    for v, pos in plot_new_verts(anchors, tco, co, tar_obm).items():
        co[v] = pos

    # velocity: new verts move with what they were plotted from
    velocity = np.zeros((n, 3), dtype=DTF)
    velocity[old] = C.velocity[new_to_old[old]]
    for v, (fidx, eidx) in anchors.items():
        anchor_verts = np.unique(eidx)
        anchor_verts = anchor_verts[old[anchor_verts]]
        if anchor_verts.shape[0]:
            velocity[v] = C.velocity[new_to_old[anchor_verts]].mean(axis=0)

    # selection is carried over as it was.  New verts are deliberately left
    # unselected: a selected vert is a grabbed vert and gets pinned (see
    # pin_selection), so marking them would freeze the new geometry in place
    # and drag its neighbours with it.
    selected = np.zeros(n, dtype=bool)
    selected[old] = C.selected[new_to_old[old]]

    U.set_bmesh(C.ob, tar_obm)

    # Vertex groups after the mesh, not before: the new mesh comes from the
    # target, which carries no weights, so anything written first is wiped.
    # The rebuild reads these back off the object.
    for name, weights in C.group_data.items():
        w = np.full(n, C.default_wieghts.get(name, 1.0), dtype=np.float32)
        w[old] = weights.ravel()[new_to_old[old]]
        U.write_vertex_group(C.ob, name, w)

    C.ob.data.shape_keys.key_blocks["MC_current"].data.foreach_set('co', co.ravel())
    C.skip_select_update = True
    C.ob.data.vertices.foreach_set('select', selected)
    C.ob.data.update()
    C.co = co
    C.pco = None                       # the target changed shape; re-read it

    stamp_mana(target, n)
    target.data.update()

    carry_over(C.ob, velocity)
    return True


def carry_cloth_velocity(C, n):
    """Carry the sim across an edit of the cloth's own mesh.

    Blender interpolates the shape keys through an edit, so the positions are
    already right on the new topology -- but velocity lives in a numpy array of
    ours, and the rebuild would start every vert from rest.  The verts are
    followed by the same ids the target uses; a vert the edit created takes the
    velocity of the old verts it ended up joined to, so it arrives moving with
    the cloth instead of hanging still in it.
    """
    ids = read_mana(C.ob.data)
    if ids is None or ids.shape[0] != n:
        return                       # never stamped: nothing to follow
    co = np.empty((n, 3), dtype=np.float32)
    C.ob.data.shape_keys.key_blocks['MC_current'].data.foreach_get('co', co.ravel())

    new_to_old = map_target_verts(ids, C.vc, co, C.co)
    old = new_to_old >= 0
    velocity = np.zeros((n, 3), dtype=DTF)
    velocity[old] = C.velocity[new_to_old[old]]

    fresh = np.arange(n)[~old]
    if fresh.shape[0]:
        obm = U.get_bmesh(C.ob)
        obm.verts.ensure_lookup_table()
        for v in fresh:
            near = [e.other_vert(obm.verts[int(v)]).index
                    for e in obm.verts[int(v)].link_edges]
            near = [i for i in near if old[i]]
            if near:
                velocity[v] = velocity[near].mean(axis=0)
    carry_over(C.ob, velocity)


def push_cloth_to_target(C, target):
    """The cloth itself was edited: bring the target onto its topology.

    The target keeps its own shape.  The cloth's MC_target shape key is a copy
    of the target's positions -- it is re-read from the target every frame --
    and Blender carries shape keys through an edit, so that key already holds
    the target's shape on the new topology, including interpolated positions
    for whatever the edit created.  Writing it back to the target is what keeps
    the pattern intact instead of stamping the simulated shape onto it.

    This used to be skipped unless the cloth was the active object, which left
    the target a topology behind: the next frame then wrote a shape key of the
    wrong length and Blender raised "internal error setting the array".
    """
    ob = C.ob
    n = len(ob.data.vertices)
    tco = np.empty(n * 3, dtype=np.float32)
    ob.data.shape_keys.key_blocks['MC_target'].data.foreach_get('co', tco)

    tmesh = ob.to_mesh(preserve_all_data_layers=True)
    U.set_bmesh(target, U.get_mesh_bmesh(tmesh))
    target.data.vertices.foreach_set('co', tco)
    target.data.update()

    carry_cloth_velocity(C, n)      # reads the old ids, so before re-stamping
    stamp_mana(target, n)
    stamp_mana(ob, n)


def target_updates(target, C, inverted=None):
    """Keep the cloth and its target in step, once per frame.

    Three things can have happened since the last frame:
      * the cloth's own mesh was edited -- the target is made to match it
      * the target's topology changed   -- retopo_from_target rebuilds the
        cloth on it without stopping the sim
      * the target only moved           -- the rest lengths and bend frames
        are re-read from it, which is what "modelling against a target" means
    """
    EDIT = False
    if target.data.is_editmode:
        target.update_from_editmode()
        EDIT = True

    if C.change:        # the cloth's own mesh was edited
        push_cloth_to_target(C, target)
        return

    tmesh = target.to_mesh(preserve_all_data_layers=True)
        
    if EDIT:
        if target.data.shape_keys:
            key = target.active_shape_key
            kco = np.empty((len(key.data) * 3), dtype=np.float32)
            key.data.foreach_get('co', kco)
            tmesh.vertices.foreach_set('co', kco)
    else:    
        if target.data.shape_keys:
            prox = U.prox_object(target)
            prox_co = np.empty((len(prox.data.vertices), 3), dtype=np.float32)
            prox.data.vertices.foreach_get('co', prox_co.ravel())
            tmesh.vertices.foreach_set('co', prox_co.ravel())
            tmesh.update()
        
    change = U.detect_changes_mesh(tmesh, C.vc, C.ec, C.fc)

    if change:
        return retopo_from_target(C, target, tmesh)

    pco = U.get_mesh_co(tmesh)
    if pco.shape[0] != len(C.ob.data.vertices):
        # the two are a topology apart; writing the key would be a hard error.
        # Rebuilding on the next pass sorts it out.
        dprint("MC: target has %d verts, cloth has %d -- waiting for the rebuild"
               % (pco.shape[0], len(C.ob.data.vertices)))
        return True
    C.ob.data.shape_keys.key_blocks["MC_target"].data.foreach_set('co', pco.ravel())

    if C.pco is not None:
        if np.allclose(pco, C.pco, rtol=1e-4, atol=1e-4):
            return
    C.pco = pco
    
    # ===== Update Linear ===== #
    spring_vecs = pco[C.spring_edges[:, 1]] - pco[C.spring_edges[:, 0]]
    C.target_dists = np.linalg.norm(spring_vecs, axis=1)
    # ===== Update Linear ===== #
    
    # ===== Update Bend ===== #
    MB = C.MB
    
    face_normals = U.get_poly_normals_mesh(tmesh)

    hinge_co = pco[MB.hinge_eidx]
    hinge_origins = hinge_co[:, 0]
    hinge_vecs = hinge_co[:, 1] - hinge_origins
    u_hinge_vecs = U.u_vecs_out(hinge_vecs)

    lf_normals = face_normals[MB.lf_idx]
    rf_normals = face_normals[MB.rf_idx]    
    
    lf_matrix = np.empty((MB.hc, 3, 3), dtype=np.float32)
    rf_matrix = np.empty((MB.hc, 3, 3), dtype=np.float32)
    
    # !!! this assumes the edge and the face normal are orthagonal, (it might be a non-planar n-gon)
    lf_matrix[:, 0] = u_hinge_vecs # matrix derived from left face data
    rf_matrix[:, 0] = u_hinge_vecs # matrix derived from right face data
    
    lf_matrix[:, 1] = lf_normals
    rf_matrix[:, 1] = rf_normals
    
    lf_matrix[:, 2] = U.fastest_cross_product(u_hinge_vecs, lf_normals)
    rf_matrix[:, 2] = U.fastest_cross_product(u_hinge_vecs, rf_normals)

    MB.lf_normal_scalers = np.einsum('vij,vj->vi', rf_matrix, lf_normals)
    MB.rf_normal_scalers = np.einsum('vij,vj->vi', lf_matrix, rf_normals)    
    # ===== Update Bend ===== #
    
    dprint("MC: target re-read")


def shrink_grow_values(C, count):
    """Shrink Grow, one value per solver iteration.

    The solver takes a per-iteration rest-length multiplier, which is how the
    C++ side receives Shrink Grow at all -- it has no argument of its own for
    it.  The same number every iteration: it is a rest length, so the cloth
    settles at it rather than being pulled there and let back out.

    (The values used to ramp from the experimental Pre-shrink back to 1.0 over
    the iterations, and Shrink Grow was multiplied in on the python side only.
    Since the C++ solver is the one that runs, Shrink Grow did nothing at all.)
    """
    return np.full(max(count, 0), C.ob.MC_props.shrink_grow, dtype=np.float32)


def stretch_force_np(C, recollide=False, first_iteration=0):

    T = time.time()
    stretch = C.ob.MC_props.stretch
    iters = math.floor(stretch / 1)
    iters -= C.b_iters # running linear in bend force to stabilize bend
    stretch_values = [1.0] * iters
    final = stretch % 1
    
    if final > 0.0:
        stretch_values += [final]
        iters += 1
        stretch_values[-1] = final

    shrink_values = shrink_grow_values(C, len(stretch_values))

    for e, stretch in enumerate(stretch_values):
        linear_springs(C, stretch, shrink_values[e])
        if recollide:
            solver_recollide(C, first_iteration + e)
        update_pin_select_sew(C)
    dprint("%.4f numpy stretch" % (time.time() - T))


def stretch_force_cpp(C, recollide=False, first_iteration=0):

    stretch = C.ob.MC_props.stretch
    iters = math.floor(stretch / 1)
    iters -= C.b_iters
    stretch_values = [1.0] * iters
    final = stretch % 1

    if final > 0.0:
        stretch_values += [final]
        iters += 1
        stretch_values[-1] = final

    shrink_values = shrink_grow_values(C, len(stretch_values))

    if recollide:
        for e, stretch in enumerate(stretch_values):
            C.solver.run_spring_solver(C, [stretch], shrink_values[e:e + 1])
            solver_recollide(C, first_iteration + e)
    else:
        C.solver.run_spring_solver(C, stretch_values, shrink_values)
    #print(time.time() - T, "C++ version")


_SCIPY_WARNED = False


def warn_no_scipy():
    """Say it once, not on every frame of a running sim."""
    global _SCIPY_WARNED
    if _SCIPY_WARNED:
        return
    _SCIPY_WARNED = True
    U.popup_error("Magnetic targets need scipy, which could not be installed. "
                  "MC5 settings > Dependencies > Install scipy.", icon='ERROR')


def magnetic(C):
    ob_mag_force = C.ob.MC_props.magnetic_force
    if ob_mag_force > 0.0:
        targets = [ob for ob in bpy.data.objects if (ob.MC_props.magnetic_target & (not ob == C.ob))]
        if len(targets) > 0:   
            mcos = np.empty((0, 3), dtype=np.float32)    
            for ob in targets:
                vc = len(ob.data.vertices)
                mco = U.co_to_matrix_space(U.get_co(ob), ob, C.ob)
                mcos = np.append(mcos, mco, axis=0)
            C.mco = mcos
                        
            if bpy.context.scene.MC_props.all_magnetic_targets:
                mag_force = ob_mag_force
            else:    
                mag_force = np.empty(0, dtype=np.float32)
                for ob in targets:
                    vc = len(ob.data.vertices)
                    force = ob.MC_props.magnetic_force
                    mag_force = np.append(mag_force, np.full(vc, force, dtype=np.float32))
                mag_force = mag_force[:, None]
            
            # TODO: for a live tree update, diff the co and move this
            # into the physics function
            C.tree = cKDTree(C.mco)
            if C.tree is None:
                warn_no_scipy()        # once per session, then quietly skipped
                return
            C.mag_force = mag_force
            
            T = time.time()            
            mode = None
            if C.ob.MC_props.magnetic_target_high:
                mode = "high"
            if C.ob.MC_props.magnetic_target_low:
                mode = "low"
            k = C.ob.MC_props.magnetic_neightbor_count
            
            magnetic_force_grok(C, mode=mode, k=k)
            update_pin_select_sew(C)

            dprint("%.4f magnetic" % (time.time() - T))


def collision_backend():
    """The C++ collision bridge when the scene asks for it and it loads,
    otherwise None (Python).  Loading is attempted once per session."""
    try:
        if bpy.context.scene.MC_props.collision_backend != 'CPP':
            return None
    except AttributeError:
        return None
    return NATIVE.get()


def object_collision(C):
    if not refresh_colliders(C):
        return

    C.normals = U.get_tri_normals(C.co[C.tridex], normalize=True)
    depth, tuner = split_depth(C, "ob")
    C.oc_split_depth = depth
    T = time.perf_counter()
    # With recollide on, the broad phase also looks ahead into the next step
    # and the result is cached for it -- see prepare_recollide.
    C.recall_data = OC.collision_force(OC_DATA, C, C.ob, OC_DATA["tridexes"], C.tidx,
                                       co_start=C.start_co, co_current=C.co, radius=0.1,
                                       lookahead=getattr(C, "oc_lookahead", None))
    C.recall_version = OC_DATA.get("version")
    if tuner is not None:
        tuner.record(time.perf_counter() - T)
    update_pin_select_sew(C)

    C.start_normals[:] = C.normals
    C.start_local_co[:] = C.local_co

    C.start_local_normals[:] = C.local_normals
    C.start_local_trico[:] = C.local_trico
    C.start_local_vert_norms[:] = C.local_vert_norms
    OC_DATA["start_local_edge_normals"][:] = OC_DATA["local_edge_normals"]


def refresh_colliders(C):
    """Bring every collider's current shape into the cloth's local space.
    Returns False when there is nothing to collide with.  Leaves the
    start_local_* (last step's) geometry alone, so calling it twice in a step
    gives the same arrays both times."""
    colliders = OC_DATA["obs"]

    if not colliders:
        # Nothing to collide with, so the stored contact pairs refer to a
        # collider that is no longer in play.  Left alone, recollide would
        # re-resolve them and hold the cloth against a surface that has been
        # deleted or switched off.
        C.recall_data = None
        C.recollide_pairs = None
        return False

    if colliders:
        dead = False
        for ob in colliders:
            try:
                ob.name
            except:
                dead = True
                break
        if dead:
            update_ob_colliders(OC_DATA)
        else:
            change = False
            for e, ob in enumerate(colliders):
                counts = OC_DATA["counts"][e]
                pm = ob.data
                if ob.data.is_editmode:
                    ob.update_from_editmode()
                
                if (ob.data.shape_keys is not None) | len(ob.modifiers) > 0:
                    ob = U.prox_object(ob)
                    pm = ob.to_mesh()
                    
                if U.detect_changes_mesh(pm, *counts):                        
                    change = True
                    new_counts = (len(pm.vertices), len(pm.edges), len(pm.polygons))
                    OC_DATA["counts"][e] = new_counts
                    
            if change:
                update_ob_colliders(OC_DATA, counts=new_counts)

        # live friction weights while a collider is being edited / weight-painted
        if any(o.data.is_editmode or o.mode == 'WEIGHT_PAINT' for o in OC_DATA["obs"]):
            read_collider_friction(OC_DATA)

        offset = 0
        for ob in OC_DATA["obs"]:
            if ob.data.is_editmode:
                ob.update_from_editmode()

            if (ob.data.shape_keys is not None) | len(ob.modifiers) > 0:
                ob = U.prox_object(ob)

            oc_co = U.apply_transforms(ob, U.get_co(ob))
            co_vc = len(ob.data.vertices)
            oc_vert_norms = U.get_vertex_normals(ob)

            w_vert_norms = U.apply_rotation(ob, oc_vert_norms)
            OC_DATA["vert_normals"][offset:co_vc + offset] = w_vert_norms
            OC_DATA["oc_co"][offset:co_vc + offset] = oc_co
            offset += co_vc
            #edge_offset += ec
        
        C.local_co = U.revert_transforms(C.ob, OC_DATA["oc_co"])
        C.local_vert_norms = U.revert_rotation(C.ob, OC_DATA["vert_normals"])
                    
        C.local_trico = C.local_co[OC_DATA["tridexes"]]
        C.local_normals = U.get_tri_normals(C.local_trico)
        
        # ===== update edge normals in place ===== #
        U.get_edge_normals(C.local_normals, OC_DATA["local_edge_normals"], OC_DATA)
        # ===== update edge normals in place ===== #
        
        if C.start_local_co is None:
            C.start_local_co = C.local_co.copy()
            C.start_local_normals = C.local_normals.copy()
            C.start_local_trico = C.local_trico.copy()
            C.start_local_vert_norms = C.local_vert_norms.copy()
            OC_DATA["start_local_edge_normals"] = OC_DATA["local_edge_normals"].copy()
        
        if C.start_local_co.shape[0] != C.local_co.shape[0]:
            C.start_local_co = C.local_co.copy()
            C.start_local_normals = C.local_normals.copy()
            C.start_local_trico = C.local_trico.copy()
            C.start_local_vert_norms = C.local_vert_norms.copy()
            OC_DATA["start_local_edge_normals"] = OC_DATA["local_edge_normals"].copy()
            
        OC_DATA["joined_co"][:, 1] = C.local_co
        OC_DATA["joined_co"][:, 0] = C.start_local_co
    return True


def gravity_local(C):
    """Gravity as a local-space vector, or None when there is none.

    Gravity is a world direction: rotating a cloth object should not swing
    which way its cloth falls.  It used to be added straight onto local +Z,
    so a rotated cloth fell along its own axis instead of down.  The world
    vector is built first and then brought into the cloth's space, exactly as
    the wind is.

    With a gravity object, its +Z axis is the direction gravity pulls, so it
    can be aimed by rotating it in the viewport.  Without one, the direction is
    world +Z, which leaves an unrotated cloth behaving precisely as before.
    """
    p = C.ob.MC_props
    amount = float(p.gravity) * 0.001
    if amount == 0.0:
        return None

    gob = getattr(p, "gravity_object", None)
    if gob is None:
        # no object: the amount is signed along world Z, so the usual -9.8
        # pulls down, exactly as it always has
        world = np.array([0.0, 0.0, amount])
    else:
        try:
            axis = np.array(gob.matrix_world, dtype=np.float64)[:3, 2]
        except ReferenceError:
            return None
        n = float(np.linalg.norm(axis))
        if n <= 1e-12:
            return None
        # With an object the arrow IS the direction things fall, so its +Z is
        # taken as the pull and only the size of the amount matters.  Using the
        # signed value here would mean an arrow pointing down plus the usual
        # negative gravity cancelled out into an upward pull.
        world = axis / n * abs(amount)

    return WIND.to_local(C.ob, world).astype(C.velocity.dtype)


def collision_steps(C):
    """How many slices to cut this step's movement into.

    A collision test asks "did the segment from where this point was to where
    it is now cross anything".  Move further than the margin in one step and a
    point can be outside the margin at both ends while having passed clean
    through the middle, so the contact is never seen.  Keeping each slice's
    movement to a fraction of the margin removes that gap.

    Returns 1 unless the user asked for more, so the default path is untouched.
    """
    p = C.ob.MC_props
    if not p.collision_auto_substeps:
        return max(1, int(p.collision_substeps))

    move = float(np.abs(C.co - C.start_co).max())
    if not np.isfinite(move) or move <= 0.0:
        return 1

    # Only margins actually in play count.  Object collision contributes its
    # radius when there is something to collide with, self collision when it is
    # switched on -- otherwise an unused radius would decide the substep count
    # for a feature nobody is running.
    margins = []
    if OC_DATA["obs"]:
        margins.append(float(p.ob_collision_radius))
    if p.self_collision:
        margins.append(float(p.sc_radius))

    margins = [m for m in margins if m > 0.0]
    if not margins:
        return 1

    # the TIGHTEST margin is the binding constraint: slicing finely enough for
    # a loose one still lets movement sail through a tight one.  This was max(),
    # which meant a large object radius silently cancelled substepping for a
    # small self collision radius, and vice versa.
    margin = min(margins)

    allowed = margin * max(float(p.collision_substep_margin), 0.05)
    n = int(math.ceil(move / allowed))
    return max(1, min(n, int(p.collision_max_substeps)))


def collision_stage(C):
    """Run the collision passes, optionally subdividing the movement.

    At one substep this is exactly what the solver did before: start_co and co
    are already the ends of the step, so the loop runs once over the whole
    movement and nothing is interpolated.

    Above one, the step is walked in equal slices.  Each slice starts from
    wherever collision actually left the cloth rather than from the ideal
    interpolated point, so a contact resolved early is carried forward instead
    of being undone by the next slice.

    Positions are corrected in every slice, but the velocity side of contact
    (damping, and cancelling the push-out so it doesn't become bounce) is
    applied once, after all of them.  It used to run inside each collision
    call, so it compounded with the substep count and the number of substeps
    changed the physics.
    """
    n = collision_steps(C)
    C.collision_substeps_used = n
    # recollide caches the last call's pairs for the next step; a slice is
    # 1/n of the step's motion, so look n slices ahead
    C.oc_lookahead = float(n) if C.ob.MC_props.ob_recollide else None
    OC.begin_velocity(C)
    SC.begin_velocity(C)

    if n <= 1:
        object_collision(C)
        if C.ob.MC_props.self_collision:
            self_collision(C)
    else:
        full_start = C.start_co.copy()
        delta = (C.co - full_start) / n
        pos = full_start.copy()

        for i in range(n):
            C.start_co[:] = pos
            pos = pos + delta
            C.co[:] = pos
            object_collision(C)
            if C.ob.MC_props.self_collision:
                self_collision(C)
            pos = C.co.copy()          # carry the resolved position forward

        C.start_co[:] = full_start      # the step as a whole still began here

    # object before self, the order they always ran in
    OC.velocity_feedback(C)
    SC.velocity_feedback(C)
    return n


def split_depth(C, which):
    """Broad-phase split depth for this call, and the tuner if one is running.

    With auto on, the depth comes from a DepthTuner kept on the cloth; with it
    off, from the property.  The tuner never writes the property -- it used to
    rewrite sc_box_count every frame, which made the UI value jitter and put a
    change on the undo stack each step.
    """
    p = C.ob.MC_props
    auto = bool(getattr(p, "%s_box_depth_auto" % which, False))
    fixed = int(getattr(p, "%s_box_depth" % which, ST.DEFAULT_DEPTH))
    attr = "%s_tuner" % which
    if not auto:
        setattr(C, attr, None)
        return fixed, None
    tuner = getattr(C, attr, None)
    if tuner is None:
        tuner = ST.DepthTuner(depth=fixed)
        setattr(C, attr, tuner)
    return tuner.current(), tuner


def self_collision(C):

    if C.ob.MC_props.self_collision:
        depth, tuner = split_depth(C, "sc")
        C.sc_split_depth = depth

        C.normals = U.get_tri_normals(C.co[C.tridex], normalize=True)
        # only the collision call is timed: the depth changes nothing else, so
        # anything more would just add noise to what the tuner is judging
        T = time.perf_counter()
        SC.collision_force(C, C.ob, C.tridex, C.tidx, co_start=C.start_co, co_current=C.co, radius=0.1)
        if tuner is not None:
            tuner.record(time.perf_counter() - T)
        C.start_normals[:] = C.normals

    
# ===== Physics ===== #
def physics(C):    
    
    if C is None:
        cloth_refresh(bpy.context.object)
        return            
    
    frame = bpy.context.scene.frame_current
    reset_frame = C.ob.MC_props.reset_frame
    if frame == reset_frame:
        reset_cloth_ob(C.ob)

    # ===== CACHE PLAYBACK ===== #
    if C.ob.MC_props.cache_playback:
        try:
            CACHE.playback_step(C)
        except Exception as e:
            print("MC cache playback error:", e)
            C.ob.MC_props["cache_playback"] = False
        return
    # ===== CACHE PLAYBACK ===== #

    EDIT = C.ob.data.is_editmode
    if EDIT:
        C.ob.update_from_editmode()
        G.group_check(C)
    
    if EDIT != C.EDIT:
        C.obm = U.get_bmesh(C.ob)
        C.EDIT = EDIT    
    
    change = U.detect_changes(C.ob, C.vc, C.ec, C.fc)
    C.change = change

    # ===== TARGET ===== #
    target = C.ob.MC_props.target_object
    if target:
        t_up = target_updates(target, C)
        if t_up:
            change = True
    # ===== TARGET ===== #

    if change:
        C = cloth_refresh(C.ob)
        if C is None:
            return
        apply_carry(C)          # a target edit hands the sim state over

    WEIGHT_PAINT = C.ob.mode == 'WEIGHT_PAINT'
        
    if WEIGHT_PAINT:
        G.group_check(C)
    
    # ===== required for grab move in edit mode ===== #
    keys = C.ob.data.shape_keys.key_blocks
    keys['MC_current'].data.foreach_get('co', C.co.ravel())
    # ===== required for grab move in edit mode ===== #

    if C.ob.active_shape_key != keys['MC_current']:
        dprint("MC: active shape key is not MC_current, skipping physics")
        C.key_switch = True
        return
    
    if C.key_switch:
        C.obm = U.get_bmesh(C.ob)
        C.key_switch = False
    
    if C.EDIT:
        C.ob.data.vertices.foreach_get('select', C.selected)

    # ===== Preserve Boundary ===== #
    if C.ob.MC_props.preserve_boundary_edges:
        C.selected[C.boundary_mask] = True
    # ===== Preserve Boundary ===== #
    
    C.start_co[:] = C.co

    # C++ or Python collision, per the scene setting -- see collide_native
    C.collide_native = collision_backend()

    # Selected verts are grabbed and get pinned, but a hooked vert must follow
    # its hook even though making the hook left it selected -- see
    # HOOK.pin_selection.  Identical to C.selected when there are no hooks.
    C.pin_selected = HOOK.pin_selection(C)
    C.velocity[C.pin_selected] = 0.0
    C.velocity *= C.ob.MC_props.velocity
    C.velocity *= C.velocity_group
    C.co += C.velocity
    C.vel_start[:] = C.co
    
    # ===== Forces ===== #
    CPP = True
    #CPP = False    

    # ===== BEND ===== #    
    T = time.time()
    bend = C.ob.MC_props.bend_force
    b_iters = math.floor(bend / 1)
    C.b_iters = b_iters
    bend_values = [1.0] * b_iters
    b_final = bend % 1
    if b_final > 0.0:
        bend_values += [b_final]

    # With recollide on, the DLL is handed one iteration per call so the
    # contacts can be re-resolved in between.  One value per call runs the
    # same arithmetic as the whole list in one call.
    recollide = prepare_recollide(C)
    if CPP:
        if recollide:
            for i, bv in enumerate(bend_values):
                C.solver.run_bend_solver(C, [bv])
                solver_recollide(C, i)
        else:
            C.solver.run_bend_solver(C, bend_values)
    else:
        for i, bv in enumerate(bend_values):
            MCB.mb_bend(C, bv)
            linear_springs(C, 1.0)
            if recollide:
                solver_recollide(C, i)
            update_pin_select_sew(C)
        dprint("%.4f bend" % (time.time() - T))
    # ===== BEND ===== #


    # ===== STRETCH ===== #
    # iteration numbering carries on from bend, so `every` counts both
    if CPP:
        stretch_force_cpp(C, recollide, len(bend_values))
    else:
        stretch_force_np(C, recollide, len(bend_values))
    # ===== STRETCH ===== #

    # ===== MAGNETIC ===== #
    magnetic(C)
    # ===== MAGNETIC ===== #

    # ===== HOOKS ===== #
    # After the shaping forces so a hook at full strength wins over them, but
    # before collision so a hook cannot shove verts through a collider.
    if HOOK.hook_force(C):
        update_pin_select_sew(C)
    # ===== HOOKS ===== #

    # ===== COLLISION ===== #
    # Both passes together, so the movement can be subdivided across them --
    # see collision_stage().  At the default of one substep this behaves
    # exactly as the two calls it replaced.
    collision_stage(C)
    # ===== COLLISION ===== #
            
    
    vel_move = C.co - C.vel_start
    C.velocity += vel_move

    g = gravity_local(C)
    if g is not None:
        C.velocity += g

    # ===== Optional Normal Effects ===== #
    get_normals = False

    inflate = C.ob.MC_props.inflate
    do_inflate = inflate != 0.0

    air_drag = C.ob.MC_props.air_drag
    do_air_drag = air_drag != 0.0
    
    inverted_air_drag = C.ob.MC_props.inverted_air_drag
    do_inverted_air_drag = inverted_air_drag != 0.0
    
    do_wind = WIND.is_active(C.ob)

    if do_inflate:
        get_normals = True
    if do_air_drag:
        get_normals = True
    if do_inverted_air_drag:
        get_normals = True
    if do_wind:
        get_normals = True      # wind is scaled by how squarely each vert faces it

    if get_normals:        
        cmesh = C.ob.to_mesh(preserve_all_data_layers=True)
        cmesh.vertices.foreach_set('co', C.co.ravel())
        cmesh.update()
        vert_normals = U.get_vertex_normals(cmesh, mesh=True)
    
    # ===== WIND ===== #
    # Before the drag terms, so drag damps the gust the same way it damps any
    # other motion.  apply() advances this cloth's wind clock whether or not
    # there is anything to apply, so gusts keep their timing.
    if do_wind:
        WIND.apply(C, vert_normals)
    else:
        C.wind_step = int(getattr(C, "wind_step", 0)) + 1
    # ===== WIND ===== #

    # !!! The order of these matters !!!
    # Both drags are weighted by their own vertex group, default 1.0 so an
    # unpainted group behaves exactly as before.  group_data is (vc, 1), which
    # broadcasts against the (vc, 3) normals and velocity.
    if do_air_drag:
        compare_vel = U.compare_vecs(C.velocity, vert_normals)
        drag = air_drag * C.group_data["MC_air_drag"]
        C.velocity -= vert_normals * (compare_vel[:, None] * drag)

    if do_inverted_air_drag:
        compare_vel = U.compare_vecs(C.velocity, vert_normals)
        # the weight has to scale the damping as well as the part added back,
        # or a zero weight would still bleed velocity away
        inv = inverted_air_drag * C.group_data["MC_inverted_air_drag"]
        invert = vert_normals * (compare_vel[:, None] * inv)
        C.velocity *= (1 - inv)
        C.velocity += invert

    update_pin_select_sew(C, store_merge=True)
    
    if do_inflate:
        # MC_inflate weights it per vertex, defaulting to 1.0 so an unpainted
        # group inflates exactly as before.  group_data is (vc, 1), which
        # broadcasts against the (vc, 3) normals.
        C.velocity += vert_normals * (inflate * C.group_data["MC_inflate"])
 
    if C.EDIT:
        try:    
            C.obm.verts.ensure_lookup_table()
        except:
            C.obm = U.get_bmesh(C.ob)
        for i in range(C.vc):
            C.obm.verts[i].co = C.co[i]
        return    

    keys['MC_current'].data.foreach_set('co', C.co.ravel())
    C.ob.data.update()

    
# ===== Cloth Object ===== #
class cloth_data:
    pass


# refresh_target / check_target lived here.  They tried to recognise unchanged
# verts by comparing each one's ring of neighbours before and after an edit,
# but were never finished (their result was only printed), and both referred to
# a `target` that does not exist in their scope, so calling either one raised.
# map_target_verts does this job from the vert ids instead.


# ===== Cloth Object ===== #
def cloth_setup(ob):
    dprint("# ===== Ran Cloth Setup ===== #")
    if ob.type == "MESH":
        
        
        C = cloth_data()
        C.ob = ob
        C.name = "henry"
        C.skip_select_update = False
        C.recall_data = None # Used for object collision recollide

        C.EDIT = ob.data.is_editmode
        if C.EDIT:
            try:    
                bpy.ops.object.mode_set()
            except:
                return
            #ob.update_from_editmode()
        C.pco = None
        C.vc = len(ob.data.vertices)
        C.ec = len(ob.data.edges)
        C.fc = len(ob.data.polygons)
        C.key_switch = False
        C.self_collide_counts = np.zeros(C.vc, dtype=np.float32)
        
        # ===== OBJECT COLLISION ===== #
        C.start_local_co = None # joined collide object co
        C.start_local_normals = None # joined collide object normals
        C.start_local_vert_norms = None
        C.object_collide_counts = np.zeros(C.vc, dtype=np.float32)

        tridex_not_used, C.tridex_eidx, C.tridex_edges, C.tri_edge_idxer = U.get_tri_edges(C.ob)    
        C.tridex_edge_booler = np.zeros(C.tridex_eidx.shape[0], dtype=bool)
        
        # ===== OBJECT COLLISION ===== #        
        
        # broad-phase split depth tuners, created on demand by split_depth()
        C.sc_tuner = None
        C.ob_tuner = None
        
        # ids both meshes' verts are followed by through the user's edits:
        # the target's when the pattern is edited, the cloth's when the cloth
        # itself is
        target = C.ob.MC_props.target_object
        if target is not None:
            stamp_mana(target)
            stamp_mana(ob, C.vc)


        C.obm = U.get_bmesh(ob, refresh=True)
        G.setup_groups(C)
        
        #C.tridex = U.get_tridex_3(ob)
        C.tridex, t1, t2, t3 = U.get_tri_edges(C.ob)
        C.tri_booler = np.zeros(C.tridex.shape, dtype=bool)
        C.vert_booler = np.zeros(C.vc, dtype=bool)
        C.tc = C.tridex.shape[0]
        C.tidx = np.arange(C.tc)
        # used for self collision edge movement


        # ===== C ++ ===== #
        # Found rather than hard-coded, so this works both as loose text
        # datablocks and as an installed addon -- see U.find_dll.
        dll_path = U.find_dll()
        if dll_path is None:
            U.popup_error("Could not find the cloth solver DLL. Set its "
                          "location in the MC5 scene settings. See the console "
                          "for the folders that were searched.", icon='ERROR')
            return None

        C.solver = CPB.ClothSolverNative(dll_path)
        C.face_verts, C.face_starts = U.build_face_vert_csr(C.obm)
        C.out_centers = np.zeros(C.fc * 3, dtype=np.float32)
        C.out_normals = np.zeros(C.fc * 3, dtype=np.float32)
        
        C.centers_lib = U.load_compute_face_centers(dll_path)
        
        #U.compute_face_centers_cpp(C.centers_lib, C.co, C.face_verts, C.face_starts, C.fc, C.out_centers)        
        
        # ===== C ++ ===== #
                
        
        # ===== Edge to Edge data ===== #
        
        C.tri_edges = np.empty((C.tridex.shape[0], 3, 2), dtype=np.int32)
        C.tri_edges[:, :, 0] = C.tridex
        C.tri_edges[:, :, 1] = np.roll(C.tridex, -1, axis=1)
        C.t_eidx = C.tri_edges.reshape(C.tri_edges.size // 2, 2)
                
        sorted_tri_edges = np.sort(C.t_eidx, axis=1)
        C.tri_edges.ravel()[:] = sorted_tri_edges.ravel()
        C.uni_tri_edges, idxes, inverse = np.unique(sorted_tri_edges, return_index=True, return_inverse=True, axis=0)
        C.tri_edge_booler = np.zeros(C.uni_tri_edges.shape[0], dtype=bool)
        C.tri_inv_idx = np.arange(idxes.shape[0])[inverse].reshape(C.tridex.shape[0], 3)
        
        C.tri_eidx = t1
        
        C.eidx = np.sort(U.get_eidx(C.ob), axis=1)
        C.eidx_booler = np.zeros(C.eidx.shape, dtype=bool)
        C.eidxer = np.arange(C.eidx.shape[0])
        
        # ===== Edge to Edge data ===== #
        
        if False:
            # ===== Edge to Edge data ===== #
            C.EDGE_VERTS = np.array([[0,1],[1,2],[2,0]])          # (3,2) - vertex slots per edge
            C.tridex_edges = C.tridex[:, C.EDGE_VERTS] 
            C.ea_idx, C.eb_idx = np.divmod(np.arange(9), 3)         # all 9 edge-edge combo indices
            # ===== Edge to Edge data ===== #
        
        C.spring_edges = get_spring_edges(ob)
        C.spring_edges_flat = C.spring_edges.ravel()
        C.es_0 = C.spring_edges[:, 0]
        C.es_1 = C.spring_edges[:, 1]
        C.es_0_cont = np.ascontiguousarray(C.es_0, dtype=np.int32)
        C.es_1_cont = np.ascontiguousarray(C.es_1, dtype=np.int32)
        
        C.left_group_mult = C.group_data['MC_stretch'][C.es_0]
        C.right_group_mult = C.group_data['MC_stretch'][C.es_1]
        
        C.velocity_group = C.group_data['MC_velocity']
        C.friction_group = C.group_data['MC_friction']

        C.se_shape = C.spring_edges.shape[0]
        C.es_count = np.bincount(C.spring_edges.ravel(), minlength=C.vc)[:, None]
        C.es_mult = np.array(1.0 / C.es_count, dtype=np.float32)
        C.sum_positions = np.zeros((C.vc, 3), dtype=DTF)

        C.p0 = np.empty((C.se_shape, 3), dtype=DTF)
        C.p1 = np.empty((C.se_shape, 3), dtype=DTF)
        
        set_key_idx = True
        keys = ob.data.shape_keys
        if keys:
            if "MC_current" in ob.data.shape_keys.key_blocks:
                set_key_idx = False
        
        U.manage_shapes(ob, shapes=['Basis', 'MC_target', 'MC_current'], values=[0.0, 0.0, 1.0])
        shape_keys = ob.data.shape_keys.key_blocks
        index = next((i for i, key in enumerate(shape_keys) if key.name == 'MC_current'), -1)
        if set_key_idx:    
            ob.active_shape_key_index = index
        
        # !!! Changing active shape or setting it here kills the bmesh !!!
                
        # === Magnetic Data === #        
        boundary_mask = np.zeros(C.vc, dtype=bool)
        boundary_verts = [[e.verts[0].index, e.verts[1].index] for e in C.obm.edges if len(e.link_faces) == 1]
        boundary_mask[boundary_verts] = True
        C.boundary_mask = boundary_mask
        # === Magnetic Data === #
                
        C.start_co = np.empty((C.vc, 3), dtype=DTF)
        shape_keys['MC_current'].data.foreach_get('co', C.start_co.ravel())
        C.source_co = np.empty((C.vc, 3), dtype=DTF)
        shape_keys['MC_target'].data.foreach_get('co', C.source_co.ravel())
        C.selected = np.zeros(C.vc, dtype=bool)
        ob.data.vertices.foreach_get('select', C.selected)
        
        C.joined_co = np.empty((C.vc, 2, 3), dtype=DTF)
        C.joined_trico = np.empty((C.tc, 6, 3), dtype=DTF)
        
        C.start_normals = U.get_tri_normals(C.start_co[C.tridex], normalize=True)
        
        spring_vecs = C.source_co[C.spring_edges[:, 1]] - C.source_co[C.spring_edges[:, 0]]
        C.target_dists = np.linalg.norm(spring_vecs, axis=1) #[:, None]
        
        C.co = C.start_co.copy()
        C.velocity = np.zeros((C.vc, 3), dtype=DTF)
        C.vel_start = np.zeros((C.vc, 3), dtype=DTF)
        
        C.triobm = U.get_triobm(C.ob)

        # === Self Collision Data === #
        C.idxer = np.arange(C.co.shape[0])
        C.sc_vert_bool = np.zeros(C.co.shape[0], dtype=bool)
        C.sc_tri_bool = np.zeros_like(C.tridex, dtype=bool)
        C.sc_lookup = U.get_extended_tridex(C.ob, steps=1)
            
        if C.ob.MC_props.auto_radius:
            get_max_radius(C)
            if C.max_radius_found:
                C.ob.MC_props.sc_radius = np.min(C.max_radius)
            else:
                # nothing to measure: no faces, so the existing radius stands
                print("MC: auto radius skipped -- '%s' has no faces to measure "
                      "a self collision radius from." % C.ob.name)
            C.ob.MC_props.auto_radius = False
        # === Self Collision Data === #

        if False:
            # ===== Test Custom Attributes =====#
            C.bend_group = np.ones(C.vc, dtype=DTF)[:, None]
            U.set_named_attribute(C.ob, C.bend_group.ravel(), name="bend_group", type="FLOAT", domain="POINT")
            # ===== Test Custom Attributes =====#

        # ===== Sew Data =====#
        basic_sew_springs(C)
        build_sew_hinges(C) # misanthropic
        # ===== Sew Data =====#

        # ===== Matrix Bend =====#
        MCB.matrix_bend_data(C)
        MCB.get_hinge_weights(C)
        # regular + sew hinges share one incidence count (must follow both builds)
        combine_bend_multipliers(C)
        # ===== Matrix Bend =====#
        
        if C.EDIT:
            bpy.ops.object.mode_set(mode='EDIT')
        
    return C


# ===== Store Cloth Data ===== #
def is_live(ob):
    """Whether this reference still points at real data.

    Removing an object does not blank out the Python references to it -- they
    keep pointing at freed RNA, and touching one raises ReferenceError rather
    than returning None.  Trying is the only way to ask.
    """
    if ob is None:
        return False
    try:
        ob.name
    except ReferenceError:
        return False
    return True


def purge_dead():
    """Drop cached cloth whose object has been deleted.

    DATA is keyed by cloth_id, which is stored on the object, and a free id is
    picked by scanning the objects that still exist.  So a deleted object's id
    reads as free and gets handed to the next new cloth while its stale entry
    is still sitting in DATA -- the new object would inherit the dead one's
    cloth data, and the first C.ob access raises.  Deleting is not the only way
    in: joining removes the objects merged into the target.
    """
    dead = [cid for cid, C in DATA.items()
            if C is not None and not is_live(getattr(C, "ob", None))]
    for cid in dead:
        del(DATA[cid])
    return dead


def get_cloth(ob, clear=False, start=False):

    if start:
        purge_dead()
        ob.MC_props.cloth_id = -1
        ids = [ob.MC_props.cloth_id for ob in bpy.data.objects]
        max_id = max(ids) + 2
        booly = np.ones(max_id + 1, dtype=bool)
        idxer = np.arange(max_id + 1)
        booly[ids] = False
        new_id = idxer[booly][0]
        ob.MC_props.cloth_id = new_id
        # belt and braces: this id is meant to be unused, so nothing may be
        # left under it
        DATA.pop(new_id, None)

    cid = ob.MC_props.cloth_id
    if clear:
        ob.MC_props.cloth_id = -1
        if cid in DATA:
            del(DATA[cid])
        return    
            
    if cid == -1:
        max_id = max([ob.MC_props.cloth_id for ob in bpy.data.objects])
        cid = max_id + 1
        ob.MC_props.cloth_id = cid
        dprint("MC: returning cid == -1")
        
    if cid in DATA:
        C = DATA[cid]
        if C: # in case C is None like during loop cut or extrude
            owner = getattr(C, "ob", None)
            if is_live(owner):
                if owner == ob:
                    return C
                # another live object is already using this id -- a duplicate
                # carries cloth_id over -- so this one needs its own rather
                # than the two of them fighting over one cache entry
                return get_cloth(ob, start=True)
            else:
                # the object this belonged to is gone and the id came back
                # round to us
                del(DATA[cid])



    C = cloth_setup(ob)
    DATA[cid] = C
    
    if C:
        C.obm = U.get_bmesh(C.ob)

    return C


# ===== Magnetic Object ===== #
class magnetic_setup:
    pass


# ===== Ob Collision Object ===== #
class ob_collision_setup:
    pass


# ===== Store Magnetic Data ===== #
def get_magnetic(ob, clear=False, start=False):
    if start:
        ob.MC_props.magnetic_id = -1
        ids = [ob.MC_props.magnetic_id for ob in bpy.data.objects]
        max_id = max(ids) + 2
        booly = np.ones(max_id + 1, dtype=bool)
        idxer = np.arange(max_id + 1)
        booly[ids] = False
        new_id = idxer[booly][0]
        ob.MC_props.magnetic_id = new_id
    
    mid = ob.MC_props.magnetic_id
    if clear:
        ob.MC_props.magnetic_id = 0
        if mid in MAGNETIC_DATA:
            del(MAGNETIC_DATA[mid])
        return
    
    if mid == -1:
        max_id = max([ob.MC_props.magnetic_id for ob in bpy.data.objects])
        mid = max_id + 1
        ob.MC_props.magnetic_id = mid
        
    if mid in MAGNETIC_DATA:
        return MAGNETIC_DATA[mid]
        
    M = magnetic_setup()
    M.ob = ob
    MAGNETIC_DATA[mid] = M
    
    
    dprint("MC: magnetic setup")
    return M


# ===== Store Object Collision Data ===== #
def get_ob_collision(ob, clear=False, start=False):
    if start:
        ob.MC_props.ob_collision_id = -1
        ids = [ob.MC_props.ob_collision_id for ob in bpy.data.objects]
        max_id = max(ids) + 2
        booly = np.ones(max_id + 1, dtype=bool)
        idxer = np.arange(max_id + 1)
        booly[ids] = False
        new_id = idxer[booly][0]
        ob.MC_props.ob_collision_id = new_id
    
    mid = ob.MC_props.ob_collision_id
    if clear:
        ob.MC_props.ob_collision_id = 0
        if mid in COLLISION_DATA:
            del(COLLISION_DATA[mid])
        return
    
    if mid == -1:
        max_id = max([ob.MC_props.ob_collision_id for ob in bpy.data.objects])
        mid = max_id + 1
        ob.MC_props.ob_collision_id = mid
        
    if mid in COLLISION_DATA:
        return COLLISION_DATA[mid]
    
    OC = ob_collision_setup()
    OC.ob = ob
    COLLISION_DATA[mid] = OC
    
    dprint("MC: object collision setup")
    return OC


# ===== Handlers ===== #
def mc_handler(self=None, contxt=None):

    cloths = [ob for ob in bpy.data.objects if ob.MC_props.cloth]
    animated = [ob for ob in cloths if ob.MC_props.animated or ob.MC_props.cache_playback]

    for ob in animated:
        C = get_cloth(ob)

        if C: # C can be None if we are in the middle of an action like loop cut
            try:    
                C.ob.MC_props
            except ReferenceError:
                ob.MC_props.cloth = False
                ob.MC_props.cloth = True
                C = get_cloth(ob)
                
            physics(C)
            

@bpy.app.handlers.persistent
def mc_load_post(*_args):
    """After a .blend is opened: what register() does when the addon is
    enabled.  Blender drops every non-persistent handler on a file load, and
    nothing re-installed ours, so in a reopened file neither cache playback nor
    animated runs did anything until a cloth setting was toggled.  The cloth
    data of the previous file is dropped with it."""
    purge_dead()
    install_handler()
    # no bake survives a file load; a file saved mid-bake would otherwise
    # open with only the Pause / Resume button showing, wired to nothing
    CACHE.bake_ended()
    CACHE.clear_stale_bake()
    for ob in bpy.data.objects:
        p = ob.MC_props
        if p.ob_collision:
            p.ob_collision = True      # rebuilds the collider data
        if p.cloth:
            p.cloth = True             # rebuilds the cloth data
        if p.magnetic_target:
            p.magnetic_target = True


def install_file_handlers(clear=False):
    """load_post / save_post handlers, replaced by name so reloading the
    addon's text blocks never stacks them."""
    for hlist, fn in ((bpy.app.handlers.load_post, mc_load_post),
                      (bpy.app.handlers.save_post, CACHE.on_save_post)):
        for h in [h for h in hlist if getattr(h, "__name__", "") == fn.__name__]:
            hlist.remove(h)
        if not clear:
            hlist.append(fn)


@bpy.app.handlers.persistent
def MC_depsgraph_update_cloth(scene, depsgraph):
    
    T = time.time()
    for update in depsgraph.updates:
        if isinstance(update.id, bpy.types.Object):
            if update.id.type == 'MESH' and update.is_updated_geometry:
                dprint("MC: running deps")
                #update.id.MC_props.cloth = update.id.MC_props.cloth
                #print(f"Mesh geometry updated: {update.id.name}")
                # Update your numpy arrays here
                #update_numpy_arrays(update.id)
    dprint("%.4f deps" % (time.time() - T))


# ===== Handlers ===== #
# Which mc_handler_continuous is actually registered, parked somewhere that
# survives re-running this module.  See install_handler().
TIMER_KEY = "MC_continuous_timer"


def mc_handler_continuous():
    # Belt and braces for the timer-stacking bug: a timer left behind by an
    # earlier load of this text retires itself on its next tick, so stale ones
    # cannot pile up extra physics steps even if install_handler never sees
    # them (a persistent timer outlives a file load, driver_namespace does not).
    live = bpy.app.driver_namespace.get(TIMER_KEY)
    if live is not None and live is not mc_handler_continuous:
        return None                 # returning None unregisters this timer

    cloths = [ob for ob in bpy.data.objects if ob.MC_props.cloth]
    continuous = [ob for ob in cloths if ob.MC_props.continuous]
    
    # Returning None should delete the handler
    if len(continuous) == 0:
        return
    
    for ob in continuous:
        C = get_cloth(ob)
        if C is None:
            ob.MC_props.cloth = False
            ob.MC_props.cloth = True
        else:
            # An exception raised out of a timer makes Blender unregister it,
            # so one bad cloth would stop continuous for every object with no
            # message beyond the traceback.  Keep the timer alive and deal with
            # the offender on its own.
            try:
                physics(C)
            except ReferenceError:
                # the cached cloth outlived its object; rebuild it
                purge_dead()
                ob.MC_props.cloth = False
                ob.MC_props.cloth = True
            except Exception:
                traceback.print_exc()
                ob.MC_props.continuous = False
                print("MC: continuous turned off for '%s' after the error "
                      "above. The other cloth objects keep running." % ob.name)

    return 0.0
    

# ===== Install Handlers ===== #
def install_handler(clear=False):
    handler_names = np.array([i.__name__ for i in bpy.app.handlers.frame_change_post])
    
    booly = [i == 'mc_handler' for i in handler_names]
    idx = np.arange(handler_names.shape[0])
    idx_to_kill = idx[booly]
    for i in idx_to_kill[::-1]:
        del(bpy.app.handlers.frame_change_post[i])
        dprint("MC: deleted handler", i)

    #deps_handler_names = np.array([i.__name__ for i in bpy.app.handlers.depsgraph_update_post])
    deps_handler_names = np.array([i.__name__ for i in bpy.app.handlers.undo_post])
    booly = [i == 'MC_depsgraph_update_cloth' for i in deps_handler_names]
    idx = np.arange(deps_handler_names.shape[0])
    idx_to_kill = idx[booly]
    for i in idx_to_kill[::-1]:
        del(bpy.app.handlers.undo_post[i])
        dprint("MC: deleted deps handler", i)


    # Frame handlers above are cleared by __name__, which catches stale ones
    # from any earlier load of this text.  Timers cannot work that way: they
    # are unregistered by the exact function object that was registered, and
    # re-running this text builds a brand new one, so the PREVIOUS module's
    # timer is invisible to is_registered() here and stays alive.  Every reload
    # added another, and every extra timer is another physics() call per tick
    # -- which is why continuous ran the sim faster than animated while
    # animated, cleared by name, stayed correct at one step per frame.
    #
    # driver_namespace is a plain session dict, so it survives re-running this
    # module and can hold on to whichever function is actually registered.
    ns = bpy.app.driver_namespace
    old = ns.get(TIMER_KEY)
    if old is not None and bpy.app.timers.is_registered(old):
        bpy.app.timers.unregister(old)
    ns.pop(TIMER_KEY, None)

    if bpy.app.timers.is_registered(mc_handler_continuous):
        bpy.app.timers.unregister(mc_handler_continuous)

    if clear:
        return

    bpy.app.timers.register(mc_handler_continuous, persistent=True)
    ns[TIMER_KEY] = mc_handler_continuous
    bpy.app.handlers.frame_change_post.append(mc_handler)
    #bpy.app.handlers.undo_post.append(MC_depsgraph_update_cloth)


'''
!!! For friction: We want the bend and stretch to still work so we don't get weird 
                    sticking. I think replacing the velocity with the movement required
                    by the friction (from barycentric) and allowing bend and stretch to
                    move the vertices could be the way. !!!


Create an object list for collision objects so each cloth
    Can collide with specific objects.    
    
Could create wrinkles based on a normal map or bit map
    Bake the normal map vectors to bend spring scalers
    

'''


