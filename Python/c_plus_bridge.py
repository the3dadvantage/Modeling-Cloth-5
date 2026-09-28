import ctypes
import os
import sys
import numpy as np


class ClothSolverNative:
    """
    Thin wrapper around cloth_solver.dll.
    Create once per cloth object (at the same point C.MB is built),
    reuse every frame.
    """

    def __init__(self, dll_path=None):
        if dll_path is None:
            # MC5 finds the library and passes it in (U.find_dll); this is only
            # a fallback, named for whichever platform we are on
            suffix = (".dll" if sys.platform.startswith("win")
                      else ".dylib" if sys.platform == "darwin" else ".so")
            dll_path = os.path.join(os.path.dirname(__file__),
                                    "mc_cloth_solver" + suffix)
        self.lib = ctypes.CDLL(dll_path)
        self._setup_argtypes()

    # -------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------

    def _setup_argtypes(self):
        fp = ctypes.POINTER(ctypes.c_float)
        ip = ctypes.POINTER(ctypes.c_int)
        bp = ctypes.POINTER(ctypes.c_ubyte)
        i  = ctypes.c_int
        f  = ctypes.c_float

        self.lib.run_bend_solver.argtypes = [
            fp,               # co
            fp, fp,           # face_centers_scratch, face_normals_scratch
            ip, ip, i,        # face_verts, face_starts, num_faces
            fp,               # bend_force_accumulator
            ip, ip,           # hinge_v0, hinge_v1
            ip, ip,           # lf_idx, rf_idx
            fp, fp,           # lf_normal_scalers, rf_normal_scalers
            ip, ip, ip, fp, i,  # lf_verts, lf_tiler, lf_matrix_tiler, lf_hinge_weights, lf_total
            ip, ip, ip, fp, i,  # rf_verts, rf_tiler, rf_matrix_tiler, rf_hinge_weights, rf_total
            fp, i,            # force_multiplier, hc
            f, f,             # e_bend_force, bend_stabilize

            # ----- sew bend (updated) -----
            fp,               # sew_bend_force_accumulator
            ip, ip,           # sew_panel1_v0, sew_panel1_v1
            ip, ip,           # sew_lf_idx, sew_rf_idx
            fp, fp,           # sew_lf_normal_scalers, sew_rf_normal_scalers
            ip, ip, ip, i,    # sew_lf_verts, sew_lf_tiler, sew_lf_matrix_tiler, sew_lf_total
            ip, ip, ip, i,    # sew_rf_verts, sew_rf_tiler, sew_rf_matrix_tiler, sew_rf_total
            ip,               # sew_merged_idx
            bp,               # merged
            fp,               # sew_force_multiplier
            i,                # sew_hc
            f,                # sew_bend_force

            # ----- spring stabilization -----
            fp,               # sum_positions
            ip, ip,           # es_0, es_1
            fp, fp, fp, fp,   # target_dists, right_group_mult, left_group_mult, es_mult
            i,                # num_springs

            # ----- pin/sew fixup -----
            fp, fp, bp,       # start_co, mc_pin, selected
            i,                # num_verts
            fp, i,            # bend_values, num_bend_iters

            # ----- sew force params -----
            ip, ip, fp, fp, i, i, f, f, f,
        ]
        self.lib.run_bend_solver.restype = None

    @staticmethod
    def _ptr_inout(arr, ctype):
        """Strict pointer for arrays C++ mutates in place. Asserts contiguity
        so a silent copy never causes the mutation to be lost."""
        assert arr.flags['C_CONTIGUOUS'], (
            f"in/out array must be C-contiguous, shape={arr.shape} strides={arr.strides}"
        )
        return arr.ctypes.data_as(ctypes.POINTER(ctype))

    @staticmethod
    def _ptr_in(arr, ctype, dtype):
        """Safe pointer for read-only inputs. Copies if non-contiguous/wrong
        dtype. Returns (ptr, keep_alive) -- caller must hold keep_alive until
        after the lib call."""
        arr = np.ascontiguousarray(arr, dtype=dtype)
        return arr.ctypes.data_as(ctypes.POINTER(ctype)), arr

    def _sew_args(self, C):
        """
        Build the 9 ctypes sew arguments shared by run_spring_solver and
        run_bend_solver. Pass num_sew_verts=0 when mesh has no sew edges and
        C++ will skip the sew pass entirely.
        """
        has_sew = (hasattr(C, 'basic_sew_verts') and
                   C.basic_sew_verts is not None and
                   len(C.basic_sew_verts) > 0)

        if has_sew:
            sv_p,  self._k_sv  = self._ptr_in(C.basic_sew_verts,             ctypes.c_int,   np.int32)
            si_p,  self._k_si  = self._ptr_in(C.sew_key_idxer,               ctypes.c_int,   np.int32)
            sm_p,  self._k_sm  = self._ptr_in(C.sew_key_multiplier.reshape(-1), ctypes.c_float, np.float32)
            ga_p               = self._ptr_inout(C.sew_group_adder,           ctypes.c_float)

            return (
                sv_p, si_p, sm_p, ga_p,
                ctypes.c_int(len(C.basic_sew_verts)),
                ctypes.c_int(len(C.sew_edge_groups)),
                ctypes.c_float(C.basic_merge_limit),
                ctypes.c_float(C.ob.MC_props.butt_sew_force),
                ctypes.c_float(C.ob.MC_props.sew_force),
            )
        else:
            # Empty placeholders -- C++ checks num_sew_verts == 0 and skips
            _ei = np.zeros(1, dtype=np.int32)
            _ef = np.zeros(1, dtype=np.float32)
            _eg = np.zeros(3,  dtype=np.float32)
            ei_p, _ = self._ptr_in(_ei, ctypes.c_int,   np.int32)
            ef_p, _ = self._ptr_in(_ef, ctypes.c_float, np.float32)
            ga_p    = self._ptr_inout(_eg, ctypes.c_float)
            return (
                ei_p, ei_p, ef_p, ga_p,
                ctypes.c_int(0), ctypes.c_int(0),
                ctypes.c_float(0.0), ctypes.c_float(0.0), ctypes.c_float(0.0),
            )

    def _sew_bend_args(self, C, merged_u8):
        """
        Build the sew-bend arguments for run_bend_solver (matrix version).
        Pass sew_hc=0 when there are no sew hinges and C++ skips the block.
        """
        SH = getattr(C, 'sew_hinges', None)
        has_sew_bend = SH is not None and SH.count > 0

        if has_sew_bend:
            sba_p  = self._ptr_inout(SH.force_accumulator,          ctypes.c_float)

            p1v0_p, _b0  = self._ptr_in(SH.panel1_v0,               ctypes.c_int,   np.int32)
            p1v1_p, _b1  = self._ptr_in(SH.panel1_v1,               ctypes.c_int,   np.int32)
            lfi_p,  _b2  = self._ptr_in(SH.face1_idx,               ctypes.c_int,   np.int32)
            rfi_p,  _b3  = self._ptr_in(SH.face2_idx,               ctypes.c_int,   np.int32)

            # NEW – scalers
            lns_p,  _b4  = self._ptr_in(SH.lf_normal_scalers.reshape(-1), ctypes.c_float, np.float32)
            rns_p,  _b5  = self._ptr_in(SH.rf_normal_scalers.reshape(-1), ctypes.c_float, np.float32)

            slv_p,  _b6  = self._ptr_in(SH.lf_verts,                ctypes.c_int,   np.int32)
            slt_p,  _b7  = self._ptr_in(SH.lf_tiler,                ctypes.c_int,   np.int32)
            slm_p,  _b8  = self._ptr_in(SH.lf_matrix_tiler,         ctypes.c_int,   np.int32)
            srv_p,  _b9  = self._ptr_in(SH.rf_verts,                ctypes.c_int,   np.int32)
            srt_p,  _b10 = self._ptr_in(SH.rf_tiler,                ctypes.c_int,   np.int32)
            srm_p,  _b11 = self._ptr_in(SH.rf_matrix_tiler,         ctypes.c_int,   np.int32)

            mid_p,  _b12 = self._ptr_in(SH.merged_idx.reshape(-1),  ctypes.c_int,   np.int32)
            mg_p,   _b13 = self._ptr_in(merged_u8,                  ctypes.c_ubyte, np.uint8)
            sfm_p,  _b14 = self._ptr_in(SH.force_multiplier.reshape(-1), ctypes.c_float, np.float32)

            # keep-alives
            self._sew_bend_keep = (_b0,_b1,_b2,_b3,_b4,_b5,_b6,_b7,_b8,_b9,_b10,_b11,_b12,_b13,_b14)

            return (
                sba_p,
                p1v0_p, p1v1_p,
                lfi_p, rfi_p,
                lns_p, rns_p,                          # scalers
                slv_p, slt_p, slm_p, ctypes.c_int(SH.lf_total),
                srv_p, srt_p, srm_p, ctypes.c_int(SH.rf_total),
                mid_p, mg_p,
                sfm_p,
                ctypes.c_int(SH.count),
                ctypes.c_float(getattr(SH, 'sew_bend_force', C.ob.MC_props.sew_bend_force)),
            )
        else:
            # dummy placeholders – C++ sees sew_hc == 0 and skips
            _ei = np.zeros(1, dtype=np.int32)
            _ef = np.zeros(3, dtype=np.float32)
            _eu = np.zeros(1, dtype=np.uint8)

            ei_p, _ = self._ptr_in(_ei, ctypes.c_int,   np.int32)
            ef_p, _ = self._ptr_in(_ef, ctypes.c_float, np.float32)
            eu_p, _ = self._ptr_in(_eu, ctypes.c_ubyte, np.uint8)

            return (
                ef_p,                       # force_accumulator dummy
                ei_p, ei_p,                 # panel1_v0, panel1_v1
                ei_p, ei_p,                 # lf_idx, rf_idx
                ef_p, ef_p,                 # lf_normal_scalers, rf_normal_scalers (dummy)
                ei_p, ei_p, ei_p, ctypes.c_int(0),  # lf arrays + total
                ei_p, ei_p, ei_p, ctypes.c_int(0),  # rf arrays + total
                ei_p, eu_p,                 # merged_idx, merged
                ef_p,                       # force_multiplier dummy
                ctypes.c_int(0),            # sew_hc = 0
                ctypes.c_float(C.ob.MC_props.sew_bend_force if C.ob.MC_props.sew_bend else 0.0),        # sew_bend_force
            )

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def run_spring_solver(self, C, stretch_values, shrink_values):
        """
        One-time setup required (add to wherever C is first built):
            C.sew_group_adder = np.zeros(len(C.sew_edge_groups)*3, dtype=np.float32)
            # or if no sew edges: C.basic_sew_verts = np.zeros(0, dtype=np.int32) etc.
        """
        num_verts   = C.co.shape[0]
        num_springs = C.es_0.shape[0]

        selected_u8  = np.ascontiguousarray(C.selected, dtype=np.uint8)
        stretch_arr  = np.ascontiguousarray(stretch_values, dtype=np.float32)
        shrink_arr   = np.ascontiguousarray(shrink_values,  dtype=np.float32)

        co_p    = self._ptr_inout(C.co.reshape(-1),           ctypes.c_float)
        sp_p    = self._ptr_inout(C.sum_positions.reshape(-1), ctypes.c_float)

        es0_p,  _k0  = self._ptr_in(C.es_0,                            ctypes.c_int,   np.int32)
        es1_p,  _k1  = self._ptr_in(C.es_1,                            ctypes.c_int,   np.int32)
        td_p,   _k2  = self._ptr_in(C.target_dists,                    ctypes.c_float, np.float32)
        rm_p,   _k3  = self._ptr_in(C.right_group_mult.reshape(-1),    ctypes.c_float, np.float32)
        lm_p,   _k4  = self._ptr_in(C.left_group_mult.reshape(-1),     ctypes.c_float, np.float32)
        em_p,   _k5  = self._ptr_in(C.es_mult.reshape(-1),             ctypes.c_float, np.float32)
        sc_p,   _k6  = self._ptr_in(C.start_co.reshape(-1),            ctypes.c_float, np.float32)
        pin_p,  _k7  = self._ptr_in(C.group_data['MC_pin'].reshape(-1), ctypes.c_float, np.float32)
        sel_p,  _k8  = self._ptr_in(selected_u8,                        ctypes.c_ubyte, np.uint8)
        stv_p,  _k9  = self._ptr_in(stretch_arr,                        ctypes.c_float, np.float32)
        shv_p,  _k10 = self._ptr_in(shrink_arr,                         ctypes.c_float, np.float32)

        sew = self._sew_args(C)

        self.lib.run_spring_solver(
            co_p, sp_p,
            es0_p, es1_p, td_p, rm_p, lm_p, em_p,
            sc_p, pin_p, sel_p,
            ctypes.c_int(num_verts), ctypes.c_int(num_springs),
            stv_p, shv_p, ctypes.c_int(stretch_arr.shape[0]),
            *sew,
        )

    def run_bend_solver(self, C, bend_values):
        """
        One-time setup required (after matrix_bend_data and build_sew_hinges):
            C.MB.face_verts, C.MB.face_starts = ClothSolverNative.build_face_csr(C.obm)
            C.MB.num_faces = len(C.obm.faces)
            C.MB.face_centers_scratch  = np.zeros(C.MB.num_faces * 3, dtype=np.float32)
            C.MB.face_normals_scratch  = np.zeros(C.MB.num_faces * 3, dtype=np.float32)
            C.MB.bend_force_accumulator = np.zeros(C.vc * 3, dtype=np.float32)
            # sew hinges (if any):
            # SH.force_accumulator = np.zeros(C.vc * 3, dtype=np.float32)  (done in build_sew_hinges)

        C.merged must be set (returned from basic_sew_force) before calling this.
        Pass C.merged=None or empty if no sew hinges.
        """
        MB = C.MB
        num_verts   = C.co.shape[0]
        num_springs = C.es_0.shape[0]

        bend_arr    = np.ascontiguousarray(bend_values, dtype=np.float32)
        selected_u8 = np.ascontiguousarray(C.selected,  dtype=np.uint8)

        # merged array for sew bend: parallel to basic_sew_verts
        merged = getattr(C, 'merged', None)
        if merged is not None and len(merged) > 0:
            merged_u8 = np.ascontiguousarray(merged, dtype=np.uint8)
        else:
            merged_u8 = np.zeros(1, dtype=np.uint8)

        # in/out buffers
        co_p    = self._ptr_inout(C.co.reshape(-1),              ctypes.c_float)
        fc_p    = self._ptr_inout(MB.face_centers_scratch,        ctypes.c_float)
        fn_p    = self._ptr_inout(MB.face_normals_scratch,        ctypes.c_float)
        bfa_p   = self._ptr_inout(MB.bend_force_accumulator,      ctypes.c_float)
        sp_p    = self._ptr_inout(C.sum_positions.reshape(-1),    ctypes.c_float)

        # read-only: face topology
        fv_p,  _f0 = self._ptr_in(MB.face_verts,  ctypes.c_int,   np.int32)
        fs_p,  _f1 = self._ptr_in(MB.face_starts, ctypes.c_int,   np.int32)

        # read-only: regular bend
        hv0_p, _h0 = self._ptr_in(MB.hinge_eidx[:,0],             ctypes.c_int,   np.int32)
        hv1_p, _h1 = self._ptr_in(MB.hinge_eidx[:,1],             ctypes.c_int,   np.int32)
        lfi_p, _h2 = self._ptr_in(MB.lf_idx,                      ctypes.c_int,   np.int32)
        rfi_p, _h3 = self._ptr_in(MB.rf_idx,                      ctypes.c_int,   np.int32)
        lns_p, _h4 = self._ptr_in(MB.lf_normal_scalers.reshape(-1), ctypes.c_float, np.float32)
        rns_p, _h5 = self._ptr_in(MB.rf_normal_scalers.reshape(-1), ctypes.c_float, np.float32)
        lv_p,  _h6 = self._ptr_in(MB.lf_verts,                    ctypes.c_int,   np.int32)
        lt_p,  _h7 = self._ptr_in(MB.lf_tiler,                    ctypes.c_int,   np.int32)
        lmt_p, _h8 = self._ptr_in(MB.lf_matrix_tiler,             ctypes.c_int,   np.int32)
        lhw_p, _h9 = self._ptr_in(C.lf_hinge_weights,             ctypes.c_float, np.float32)
        rv_p, _h10 = self._ptr_in(MB.rf_verts,                    ctypes.c_int,   np.int32)
        rt_p, _h11 = self._ptr_in(MB.rf_tiler,                    ctypes.c_int,   np.int32)
        rmt_p,_h12 = self._ptr_in(MB.rf_matrix_tiler,             ctypes.c_int,   np.int32)
        rhw_p,_h13 = self._ptr_in(C.rf_hinge_weights,             ctypes.c_float, np.float32)
        fm_p, _h14 = self._ptr_in(MB.force_multiplier.reshape(-1), ctypes.c_float, np.float32)

        # read-only: spring stabilization
        es0_p, _s0 = self._ptr_in(C.es_0,                           ctypes.c_int,   np.int32)
        es1_p, _s1 = self._ptr_in(C.es_1,                           ctypes.c_int,   np.int32)
        td_p,  _s2 = self._ptr_in(C.target_dists,                   ctypes.c_float, np.float32)
        rm_p,  _s3 = self._ptr_in(C.right_group_mult.reshape(-1),   ctypes.c_float, np.float32)
        lm_p,  _s4 = self._ptr_in(C.left_group_mult.reshape(-1),    ctypes.c_float, np.float32)
        em_p,  _s5 = self._ptr_in(C.es_mult.reshape(-1),            ctypes.c_float, np.float32)

        # read-only: pin/sew fixup
        sc_p,  _p0 = self._ptr_in(C.start_co.reshape(-1),           ctypes.c_float, np.float32)
        pin_p, _p1 = self._ptr_in(C.group_data['MC_pin'].reshape(-1), ctypes.c_float, np.float32)
        sel_p, _p2 = self._ptr_in(selected_u8,                       ctypes.c_ubyte, np.uint8)
        bv_p,  _p3 = self._ptr_in(bend_arr,                          ctypes.c_float, np.float32)

        sew_bend = self._sew_bend_args(C, merged_u8)
        sew      = self._sew_args(C)

        self.lib.run_bend_solver(
            co_p,
            fc_p, fn_p, fv_p, fs_p, ctypes.c_int(MB.num_faces),
            bfa_p,
            hv0_p, hv1_p, lfi_p, rfi_p,
            lns_p, rns_p,
            lv_p, lt_p, lmt_p, lhw_p, ctypes.c_int(MB.lf_verts.shape[0]),
            rv_p, rt_p, rmt_p, rhw_p, ctypes.c_int(MB.rf_verts.shape[0]),
            fm_p, ctypes.c_int(MB.hc),
            ctypes.c_float(C.ob.MC_props.e_bend_force),
            ctypes.c_float(C.ob.MC_props.bend_stabilize),
            *sew_bend,
            sp_p,
            es0_p, es1_p, td_p, rm_p, lm_p, em_p,
            ctypes.c_int(num_springs),
            sc_p, pin_p, sel_p,
            ctypes.c_int(num_verts),
            bv_p, ctypes.c_int(bend_arr.shape[0]),
            *sew,
        )

    @staticmethod
    def build_face_csr(obm):
        """One-time setup: flatten obm.faces into CSR layout."""
        face_verts_list = []
        face_starts = [0]
        for face in obm.faces:
            vidx = [v.index for v in face.verts]
            face_verts_list.extend(vidx)
            face_starts.append(face_starts[-1] + len(vidx))
        return (np.array(face_verts_list, dtype=np.int32),
                np.array(face_starts,     dtype=np.int32))