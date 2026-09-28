

# This will assume that all n-gons are planar. It will have the effect of making the n_gons planar
#   unless I get a scale for moving the n-gons on the normal
#   Maybe I can add a feature that flattens n-gons?


import bpy
import numpy as np


try:    
    U = bpy.data.texts['utils.py'].as_module()
    CPB = bpy.data.texts['c_plus_bridge.py'].as_module()
    
except:
    from . import utils as U
    from . import c_plus_bridge as CPB


class MatrixBend():
    pass


def get_hinge_weights(C):
    """For using the vertex group."""
    w_eidx = C.group_data["MC_bend"][C.MB.hinge_eidx]
    C.hinge_weights = np.mean(w_eidx, axis=1).ravel()
    C.lf_hinge_weights = C.hinge_weights[C.MB.lf_matrix_tiler]
    C.rf_hinge_weights = C.hinge_weights[C.MB.rf_matrix_tiler]


def matrix_bend_data(C):
    
    MB = MatrixBend()
    
    ob = C.ob # the blender object
    obm = C.obm # the bmesh from the blender object
    
    # ===== FACE CENTER & NORMAL ===== #
    tco = np.empty((len(ob.data.vertices), 3), dtype=np.float32)
    ob.data.shape_keys.key_blocks["MC_target"].data.foreach_get('co', tco.ravel()) # the starting state of the mesh for deriving rest angles and such

    C.out_centers[:] = 0.0
    tco_cont = np.ascontiguousarray(tco.reshape(-1), dtype=np.float32)
    U.compute_face_centers_cpp(C.centers_lib, tco_cont, C.face_verts, C.face_starts, C.fc, C.out_centers)

    C.out_normals[:] = 0.0
    face_normals = U.compute_face_normals_cpp(C.centers_lib, tco_cont, C.face_verts, C.face_starts, C.fc, C.out_normals)
    
    

    # ===== FACE CENTER & NORMAL ===== #    
    eidx = U.get_eidx(ob)
    
    hinge_edges = [e for e in obm.edges if len(e.link_faces) == 2]
    hc = len(hinge_edges)
    
    hinge_idx = np.empty(hc, dtype=np.int32)
    hinge_eidx = np.empty((hc, 2), dtype=np.int32)
    
    lf_idx = np.empty(hc, dtype=np.int32)
    rf_idx = np.empty(hc, dtype=np.int32)
    
    lf_tiler = []
    rf_tiler = []
    
    lf_verts = []
    rf_verts = []
    
    lf_matrix_tiler = []
    rf_matrix_tiler = []
    
    for i, ed in enumerate(hinge_edges):
        hinge_idx[i] = ed.index
        hinge_eidx[i] = eidx[ed.index]
        
        lf = ed.link_faces[0]
        rf = ed.link_faces[1]
        
        lf_idx[i] = lf.index
        rf_idx[i] = rf.index
        
        lf_vidx = [v.index for v in lf.verts]
        rf_vidx = [v.index for v in rf.verts]
        
        lfc = len(lf_vidx)
        rfc = len(rf_vidx)
        
        lf_verts += lf_vidx
        rf_verts += rf_vidx
        
        lf_tiler += [lf.index] * lfc
        rf_tiler += [rf.index] * rfc
        
        lf_matrix_tiler += [i] * lfc
        rf_matrix_tiler += [i] * rfc
    
    # Convert to numpy arrays
    lf_verts = np.array(lf_verts, dtype=np.int32)
    rf_verts = np.array(rf_verts, dtype=np.int32)
    lf_tiler = np.array(lf_tiler, dtype=np.int32)
    rf_tiler = np.array(rf_tiler, dtype=np.int32)
    lf_matrix_tiler = np.array(lf_matrix_tiler, dtype=np.int32)
    rf_matrix_tiler = np.array(rf_matrix_tiler, dtype=np.int32)
    
    hinge_co = C.source_co[hinge_eidx]
    hinge_origins = hinge_co[:, 0]
    hinge_vecs = hinge_co[:, 1] - hinge_origins
    u_hinge_vecs = U.u_vecs_out(hinge_vecs)
    
    lf_normals = face_normals[lf_idx]
    rf_normals = face_normals[rf_idx]    
    
    lf_matrix = np.empty((hc, 3, 3), dtype=np.float32)
    rf_matrix = np.empty((hc, 3, 3), dtype=np.float32)
    
    # !!! this assumes the edge and the face normal are orthagonal, (it might be a non-planar n-gon)
    lf_matrix[:, 0] = u_hinge_vecs # matrix derived from left face data
    rf_matrix[:, 0] = u_hinge_vecs # matrix derived from right face data
    
    lf_matrix[:, 1] = lf_normals
    rf_matrix[:, 1] = rf_normals
    
    lf_matrix[:, 2] = U.fastest_cross_product(u_hinge_vecs, lf_normals)
    rf_matrix[:, 2] = U.fastest_cross_product(u_hinge_vecs, rf_normals)
    
    MB.lf_normal_scalers = np.einsum('vij,vj->vi', rf_matrix, lf_normals)
    MB.rf_normal_scalers = np.einsum('vij,vj->vi', lf_matrix, rf_normals)
    
    MB.hc = hc

    MB.hinge_idx = hinge_idx
    MB.hinge_eidx = hinge_eidx
    
    MB.lf_idx = lf_idx
    MB.rf_idx = rf_idx
    
    MB.lf_tiler = lf_tiler
    MB.rf_tiler = rf_tiler
    
    MB.lf_verts = lf_verts
    MB.rf_verts = rf_verts        
    
    v_counts = np.bincount(np.append(lf_verts, rf_verts))
    MB.force_multiplier = np.nan_to_num(1 / v_counts)[:, None]
    
    MB.lf_matrix = np.empty((hc, 3, 3), dtype=np.float32)
    MB.rf_matrix = np.empty((hc, 3, 3), dtype=np.float32)

    MB.lf_matrix_tiler = lf_matrix_tiler
    MB.rf_matrix_tiler = rf_matrix_tiler

    MB.force_accumulator = np.zeros((C.vc, 3), dtype=np.float32)
    
    # Pre-allocate arrays for mb_bend
    lf_vert_count = len(lf_verts)
    rf_vert_count = len(rf_verts)
    
    MB.hinge_co = np.empty((hc, 2, 3), dtype=np.float32)
    MB.hinge_origins = np.empty((hc, 3), dtype=np.float32)
    MB.hinge_vecs = np.empty((hc, 3), dtype=np.float32)
    MB.u_hinge_vecs = np.empty((hc, 3), dtype=np.float32)
    
    MB.lf_centers = np.empty((hc, 3), dtype=np.float32)
    MB.rf_centers = np.empty((hc, 3), dtype=np.float32)
    MB.lf_center_vecs = np.empty((hc, 3), dtype=np.float32)
    MB.rf_center_vecs = np.empty((hc, 3), dtype=np.float32)
    
    MB.lf_normals = np.empty((hc, 3), dtype=np.float32)
    MB.rf_normals = np.empty((hc, 3), dtype=np.float32)
    
    MB.lf_target_normals = np.empty((hc, 3), dtype=np.float32)
    MB.rf_target_normals = np.empty((hc, 3), dtype=np.float32)
    
    MB.lf_current_center_vecs = np.empty((lf_vert_count, 3), dtype=np.float32)
    MB.rf_current_center_vecs = np.empty((rf_vert_count, 3), dtype=np.float32)
    
    MB.lf_dots = np.empty(lf_vert_count, dtype=np.float32)
    MB.rf_dots = np.empty(rf_vert_count, dtype=np.float32)
    
    MB.lf_force = np.empty((lf_vert_count, 3), dtype=np.float32)
    MB.rf_force = np.empty((rf_vert_count, 3), dtype=np.float32)
    
    MB.move = np.empty((C.vc, 3), dtype=np.float32)
    MB.u_hinge_vecs_length = np.empty(hc, dtype=np.float32)
    
    # Pre-allocate for fancy indexing
    MB.lf_tiled_normals = np.empty((lf_vert_count, 3), dtype=np.float32)
    MB.rf_tiled_normals = np.empty((rf_vert_count, 3), dtype=np.float32)
    MB.lf_tiled_target_normals = np.empty((lf_vert_count, 3), dtype=np.float32)
    MB.rf_tiled_target_normals = np.empty((rf_vert_count, 3), dtype=np.float32)
    MB.lf_tiled_face_centers = np.empty((lf_vert_count, 3), dtype=np.float32)
    MB.rf_tiled_face_centers = np.empty((rf_vert_count, 3), dtype=np.float32)
    MB.lf_co = np.empty((lf_vert_count, 3), dtype=np.float32)
    MB.rf_co = np.empty((rf_vert_count, 3), dtype=np.float32)
    C.MB = MB
    
    # ===== C++ ===== #
    C.MB.face_verts, C.MB.face_starts = CPB.ClothSolverNative.build_face_csr(C.obm)
    C.MB.num_faces = len(C.obm.faces)
    C.MB.face_centers_scratch = np.zeros(C.MB.num_faces * 3, dtype=np.float32)
    C.MB.face_normals_scratch = np.zeros(C.MB.num_faces * 3, dtype=np.float32)
    C.MB.bend_force_accumulator = np.zeros(C.vc * 3, dtype=np.float32)
    # ===== C++ ===== #
    
    

def mb_bend(C, bv):
    MB = C.MB

    # ===== FACE CENTER & NORMAL ===== #
    # ===== C++ ===== #
    co_cont = np.ascontiguousarray(C.co.reshape(-1), dtype=np.float32)
    face_centers = U.compute_face_centers_cpp(C.centers_lib, co_cont, C.face_verts, C.face_starts, C.fc, C.out_centers)
    face_normals = U.compute_face_normals_cpp(C.centers_lib, co_cont, C.face_verts, C.face_starts, C.fc, C.out_normals)
    # ===== C++ ===== #

    np.take(C.co, MB.hinge_eidx, axis=0, out=MB.hinge_co)
    MB.hinge_origins[:] = MB.hinge_co[:, 0]
    np.subtract(MB.hinge_co[:, 1], MB.hinge_origins, out=MB.hinge_vecs)

    U.u_vecs_out(MB.hinge_vecs, out=MB.u_hinge_vecs, length_out=MB.u_hinge_vecs_length)

    np.take(face_centers, MB.lf_idx, axis=0, out=MB.lf_centers)
    np.take(face_centers, MB.rf_idx, axis=0, out=MB.rf_centers)
    
    np.take(face_normals, MB.lf_idx, axis=0, out=MB.lf_normals)
    np.take(face_normals, MB.rf_idx, axis=0, out=MB.rf_normals)
    
    MB.lf_matrix[:, 0] = MB.u_hinge_vecs
    MB.rf_matrix[:, 0] = MB.u_hinge_vecs
    
    MB.lf_matrix[:, 1] = MB.lf_normals
    MB.rf_matrix[:, 1] = MB.rf_normals
    
    U.fastest_cross_product(MB.u_hinge_vecs, MB.lf_normals, c=MB.lf_matrix[:, 2])
    U.fastest_cross_product(MB.u_hinge_vecs, MB.rf_normals, c=MB.rf_matrix[:, 2])
    
    np.einsum('vi,vij->vj', MB.lf_normal_scalers, MB.rf_matrix, out=MB.lf_target_normals)
    np.einsum('vi,vij->vj', MB.rf_normal_scalers, MB.lf_matrix, out=MB.rf_target_normals)
    
    # --- NEW: monotonic bend error ---
    lf_ndot = np.einsum('ij,ij->i', MB.lf_normals, MB.lf_target_normals)
    rf_ndot = np.einsum('ij,ij->i', MB.rf_normals, MB.rf_target_normals)

    lf_bend_error = 1.0 - lf_ndot
    rf_bend_error = 1.0 - rf_ndot
    
    # Get current center vecs using pre-allocated arrays
    np.take(C.co, MB.lf_verts, axis=0, out=MB.lf_co)
    np.take(face_centers, MB.lf_tiler, axis=0, out=MB.lf_tiled_face_centers)
    np.subtract(MB.lf_co, MB.lf_tiled_face_centers, out=MB.lf_current_center_vecs)
    
    np.take(C.co, MB.rf_verts, axis=0, out=MB.rf_co)
    np.take(face_centers, MB.rf_tiler, axis=0, out=MB.rf_tiled_face_centers)
    np.subtract(MB.rf_co, MB.rf_tiled_face_centers, out=MB.rf_current_center_vecs)
    
    # Get tiled target normals
    np.take(MB.lf_target_normals, MB.lf_matrix_tiler, axis=0, out=MB.lf_tiled_target_normals)
    np.take(MB.rf_target_normals, MB.rf_matrix_tiler, axis=0, out=MB.rf_tiled_target_normals)
    
    # Compute dots using einsum with out parameter
    np.einsum('ij,ij->i', MB.lf_current_center_vecs, MB.lf_tiled_target_normals, out=MB.lf_dots)
    np.einsum('ij,ij->i', MB.rf_current_center_vecs, MB.rf_tiled_target_normals, out=MB.rf_dots)
    
    # Scale and negate dots
    MB.lf_dots *= -C.ob.MC_props.e_bend_force * C.lf_hinge_weights * lf_bend_error[MB.lf_matrix_tiler]
    MB.rf_dots *= -C.ob.MC_props.e_bend_force * C.rf_hinge_weights * rf_bend_error[MB.rf_matrix_tiler]
    
    # Get tiled face normals and compute forces
    np.take(face_normals, MB.lf_tiler, axis=0, out=MB.lf_tiled_normals)
    np.take(face_normals, MB.rf_tiler, axis=0, out=MB.rf_tiled_normals)
    
    np.multiply(MB.lf_tiled_normals, MB.lf_dots[:, None], out=MB.lf_force)
    np.multiply(MB.rf_tiled_normals, MB.rf_dots[:, None], out=MB.rf_force)

    # Accumulate forces
    MB.force_accumulator[:] = 0.0
    np.add.at(MB.force_accumulator, MB.lf_verts, MB.lf_force)
    np.add.at(MB.force_accumulator, MB.rf_verts, MB.rf_force)
    
    # Apply force multiplier and move
    force_scalar = MB.force_multiplier * C.ob.MC_props.bend_stabilize * bv
    np.multiply(MB.force_accumulator, force_scalar, out=MB.move)
    C.co += np.nan_to_num(MB.move)