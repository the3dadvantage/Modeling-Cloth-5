import bpy
import numpy as np
import time

U = bpy.data.texts['utils.py'].as_module()


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
    
    ob = C.ob
    obm = C.obm
    
    # ===== FACE CENTER & NORMAL ===== #
    tco = np.empty((len(ob.data.vertices), 3), dtype=np.float32)
    ob.data.shape_keys.key_blocks["MC_target"].data.foreach_get('co', tco.ravel())
    cmesh = ob.to_mesh(preserve_all_data_layers=True)
    cmesh.vertices.foreach_set('co', tco.ravel())
    cmesh.update()
    face_centers = U.get_face_centers_mesh(cmesh)
    face_normals = U.get_poly_normals_mesh(cmesh)
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


def mb_bend(C, bv):
    MB = C.MB
    ob = C.ob
    co = C.co

    cmesh = ob.to_mesh(preserve_all_data_layers=True)
    cmesh.vertices.foreach_set('co', C.co.ravel())
    cmesh.update()
    face_centers = U.get_face_centers_mesh(cmesh)
    face_normals = U.get_poly_normals_mesh(cmesh)

    # ===== FACE CENTER & NORMAL ===== #
    
    # Use pre-allocated arrays and out parameters
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
    MB.lf_dots *= -C.ob.MC_props.e_bend_force * C.lf_hinge_weights
    MB.rf_dots *= -C.ob.MC_props.e_bend_force * C.rf_hinge_weights
    
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
