// The Visual Studio project compiles this with a precompiled header sitting
// next to it; a copy built anywhere else (cpp/solver, CI) has no pch.h and does
// not need one -- nothing in here uses windows.h.
#if defined(_MSC_VER) && __has_include("pch.h")
#include "pch.h"
#endif
#include <cmath>
#include <cstring>
#include <vector>

// Exporting a symbol from a shared library is spelled differently by each
// compiler.  Everything else in this file is plain C++17, so this macro is all
// that stands between it and a mac (.dylib) or linux (.so) build.
#if defined(_WIN32)
  #define MC_API __declspec(dllexport)
#else
  #define MC_API __attribute__((visibility("default")))
#endif

extern "C" {

    // =============================================================================
    // SPRING SOLVER
    // =============================================================================
    MC_API void linear_springs(
        float* co,
        float* sum_positions,
        const int* es_0,
        const int* es_1,
        const float* target_dists,
        const float* right_group_mult,
        const float* left_group_mult,
        const float* es_mult,
        int num_verts,
        int num_springs,
        float shrink_grow,
        float stretch)
    {
        std::memset(sum_positions, 0, sizeof(float) * num_verts * 3);
        for (int s = 0; s < num_springs; ++s) {
            int i0 = es_0[s] * 3;
            int i1 = es_1[s] * 3;

            float vx = co[i0 + 0] - co[i1 + 0];
            float vy = co[i0 + 1] - co[i1 + 1];
            float vz = co[i0 + 2] - co[i1 + 2];

            float d = std::sqrt(vx * vx + vy * vy + vz * vz);
            if (d < 1e-12f) continue;

            float scale = (target_dists[s] * shrink_grow) / d - 1.0f;
            float rmult = right_group_mult[s];
            float lmult = left_group_mult[s];

            sum_positions[i0 + 0] += scale * vx * rmult;
            sum_positions[i0 + 1] += scale * vy * rmult;
            sum_positions[i0 + 2] += scale * vz * rmult;

            sum_positions[i1 + 0] -= scale * vx * lmult;
            sum_positions[i1 + 1] -= scale * vy * lmult;
            sum_positions[i1 + 2] -= scale * vz * lmult;
        }
        for (int v = 0; v < num_verts; ++v) {
            float m = es_mult[v] * stretch;
            for (int axis = 0; axis < 3; ++axis) {
                float val = sum_positions[v * 3 + axis] * m;
                if (std::isnan(val) || std::isinf(val)) val = 0.0f;
                co[v * 3 + axis] += val;
            }
        }
    }

    // =============================================================================
    // PIN / SELECT / SEW FIXUP PASS
    // =============================================================================
    MC_API void update_pin_select_sew(
        float* co,
        const float* start_co,
        const float* mc_pin,
        const unsigned char* selected,
        int num_verts,
        const int* basic_sew_verts,
        const int* sew_key_idxer,
        const float* sew_key_multiplier,
        float* group_adder,
        int num_sew_verts,
        int num_groups,
        float basic_merge_limit,
        float butt_sew_force,
        float sew_force)
    {
        if (num_sew_verts > 0) {
            std::memset(group_adder, 0, sizeof(float) * num_groups * 3);
            for (int k = 0; k < num_sew_verts; ++k) {
                int vert = basic_sew_verts[k];
                int group = sew_key_idxer[k];
                group_adder[group * 3 + 0] += co[vert * 3 + 0];
                group_adder[group * 3 + 1] += co[vert * 3 + 1];
                group_adder[group * 3 + 2] += co[vert * 3 + 2];
            }
            for (int g = 0; g < num_groups; ++g) {
                float m = sew_key_multiplier[g];
                group_adder[g * 3 + 0] *= m;
                group_adder[g * 3 + 1] *= m;
                group_adder[g * 3 + 2] *= m;
            }
            float merge_threshold = basic_merge_limit * butt_sew_force;
            for (int k = 0; k < num_sew_verts; ++k) {
                int vert = basic_sew_verts[k];
                int group = sew_key_idxer[k];
                int vidx = vert * 3;
                float dx = group_adder[group * 3 + 0] - co[vidx + 0];
                float dy = group_adder[group * 3 + 1] - co[vidx + 1];
                float dz = group_adder[group * 3 + 2] - co[vidx + 2];
                float dist = std::sqrt(dx * dx + dy * dy + dz * dz);
                if (dist < merge_threshold) {
                    co[vidx + 0] += dx;
                    co[vidx + 1] += dy;
                    co[vidx + 2] += dz;
                }
                else {
                    co[vidx + 0] += dx * sew_force;
                    co[vidx + 1] += dy * sew_force;
                    co[vidx + 2] += dz * sew_force;
                }
            }
        }
        for (int v = 0; v < num_verts; ++v) {
            float pin = mc_pin[v];
            for (int axis = 0; axis < 3; ++axis) {
                int idx = v * 3 + axis;
                co[idx] += (start_co[idx] - co[idx]) * pin;
            }
            if (selected[v]) {
                co[v * 3 + 0] = start_co[v * 3 + 0];
                co[v * 3 + 1] = start_co[v * 3 + 1];
                co[v * 3 + 2] = start_co[v * 3 + 2];
            }
        }
    }

    // =============================================================================
    // STRETCH ITERATION LOOP
    // =============================================================================
    MC_API void run_spring_solver(
        float* co,
        float* sum_positions,
        const int* es_0,
        const int* es_1,
        const float* target_dists,
        const float* right_group_mult,
        const float* left_group_mult,
        const float* es_mult,
        const float* start_co,
        const float* mc_pin,
        const unsigned char* selected,
        int num_verts,
        int num_springs,
        const float* stretch_values,
        const float* shrink_values,
        int num_iters,
        const int* basic_sew_verts,
        const int* sew_key_idxer,
        const float* sew_key_multiplier,
        float* group_adder,
        int num_sew_verts,
        int num_groups,
        float basic_merge_limit,
        float butt_sew_force,
        float sew_force)
    {
        for (int it = 0; it < num_iters; ++it) {
            linear_springs(
                co, sum_positions,
                es_0, es_1, target_dists,
                right_group_mult, left_group_mult, es_mult,
                num_verts, num_springs,
                shrink_values[it], stretch_values[it]
            );
            update_pin_select_sew(
                co, start_co, mc_pin, selected, num_verts,
                basic_sew_verts, sew_key_idxer, sew_key_multiplier,
                group_adder, num_sew_verts, num_groups,
                basic_merge_limit, butt_sew_force, sew_force
            );
        }
    }

    // =============================================================================
    // FACE GEOMETRY
    // =============================================================================
    MC_API void compute_face_centers(
        const float* co,
        const int* face_verts,
        const int* face_starts,
        int num_faces,
        float* out_centers)
    {
        for (int f = 0; f < num_faces; ++f) {
            int start = face_starts[f];
            int end = face_starts[f + 1];
            int count = end - start;
            float sx = 0.0f, sy = 0.0f, sz = 0.0f;
            for (int k = start; k < end; ++k) {
                int v = face_verts[k] * 3;
                sx += co[v + 0]; sy += co[v + 1]; sz += co[v + 2];
            }
            float inv = 1.0f / (float)count;
            out_centers[f * 3 + 0] = sx * inv;
            out_centers[f * 3 + 1] = sy * inv;
            out_centers[f * 3 + 2] = sz * inv;
        }
    }

    MC_API void compute_face_normals(
        const float* co,
        const int* face_verts,
        const int* face_starts,
        int num_faces,
        float* out_normals)
    {
        for (int f = 0; f < num_faces; ++f) {
            int start = face_starts[f];
            int end = face_starts[f + 1];
            int count = end - start;
            float nx = 0.0f, ny = 0.0f, nz = 0.0f;
            for (int k = 0; k < count; ++k) {
                int ic = face_verts[start + k] * 3;
                int in_ = face_verts[start + (k + 1) % count] * 3;
                float xc = co[ic + 0], yc = co[ic + 1], zc = co[ic + 2];
                float xn = co[in_ + 0], yn = co[in_ + 1], zn = co[in_ + 2];
                nx += (yc - yn) * (zc + zn);
                ny += (zc - zn) * (xc + xn);
                nz += (xc - xn) * (yc + yn);
            }
            float len = std::sqrt(nx * nx + ny * ny + nz * nz);
            if (len < 1e-12f) {
                out_normals[f * 3 + 0] = out_normals[f * 3 + 1] = out_normals[f * 3 + 2] = 0.0f;
            }
            else {
                float inv = 1.0f / len;
                out_normals[f * 3 + 0] = nx * inv;
                out_normals[f * 3 + 1] = ny * inv;
                out_normals[f * 3 + 2] = nz * inv;
            }
        }
    }

    // =============================================================================
    // REGULAR BEND FORCE
    // =============================================================================
    MC_API void compute_bend_forces(
        float* co,
        float* force_accumulator,
        const int* hinge_v0,
        const int* hinge_v1,
        const int* lf_idx,
        const int* rf_idx,
        const float* lf_normal_scalers,
        const float* rf_normal_scalers,
        const int* lf_verts,
        const int* lf_tiler,
        const int* lf_matrix_tiler,
        const float* lf_hinge_weights,
        int lf_total,
        const int* rf_verts,
        const int* rf_tiler,
        const int* rf_matrix_tiler,
        const float* rf_hinge_weights,
        int rf_total,
        const float* face_centers,
        const float* face_normals,
        const float* force_multiplier,
        int num_verts,
        int hc,
        float e_bend_force,
        float bend_stabilize,
        float bv)
    {
        std::memset(force_accumulator, 0, sizeof(float) * num_verts * 3);
        std::vector<float> lf_matrix(hc * 9);
        std::vector<float> rf_matrix(hc * 9);
        std::vector<float> lf_target_normal(hc * 3);
        std::vector<float> rf_target_normal(hc * 3);
        std::vector<float> lf_bend_error(hc);
        std::vector<float> rf_bend_error(hc);

        for (int h = 0; h < hc; ++h) {
            int v0 = hinge_v0[h] * 3, v1 = hinge_v1[h] * 3;
            float hx = co[v1 + 0] - co[v0 + 0], hy = co[v1 + 1] - co[v0 + 1], hz = co[v1 + 2] - co[v0 + 2];
            float hlen = std::sqrt(hx * hx + hy * hy + hz * hz);
            float inv_h = (hlen > 1e-12f) ? 1.0f / hlen : 0.0f;
            hx *= inv_h; hy *= inv_h; hz *= inv_h;

            int lfi = lf_idx[h] * 3, rfi = rf_idx[h] * 3;
            float lnx = face_normals[lfi + 0], lny = face_normals[lfi + 1], lnz = face_normals[lfi + 2];
            float rnx = face_normals[rfi + 0], rny = face_normals[rfi + 1], rnz = face_normals[rfi + 2];

            lf_matrix[h * 9 + 0] = hx; lf_matrix[h * 9 + 1] = hy; lf_matrix[h * 9 + 2] = hz;
            lf_matrix[h * 9 + 3] = lnx; lf_matrix[h * 9 + 4] = lny; lf_matrix[h * 9 + 5] = lnz;
            lf_matrix[h * 9 + 6] = hy * lnz - hz * lny;
            lf_matrix[h * 9 + 7] = hz * lnx - hx * lnz;
            lf_matrix[h * 9 + 8] = hx * lny - hy * lnx;

            rf_matrix[h * 9 + 0] = hx; rf_matrix[h * 9 + 1] = hy; rf_matrix[h * 9 + 2] = hz;
            rf_matrix[h * 9 + 3] = rnx; rf_matrix[h * 9 + 4] = rny; rf_matrix[h * 9 + 5] = rnz;
            rf_matrix[h * 9 + 6] = hy * rnz - hz * rny;
            rf_matrix[h * 9 + 7] = hz * rnx - hx * rnz;
            rf_matrix[h * 9 + 8] = hx * rny - hy * rnx;

            float ls0 = lf_normal_scalers[h * 3 + 0], ls1 = lf_normal_scalers[h * 3 + 1], ls2 = lf_normal_scalers[h * 3 + 2];
            lf_target_normal[h * 3 + 0] = ls0 * rf_matrix[h * 9 + 0] + ls1 * rf_matrix[h * 9 + 3] + ls2 * rf_matrix[h * 9 + 6];
            lf_target_normal[h * 3 + 1] = ls0 * rf_matrix[h * 9 + 1] + ls1 * rf_matrix[h * 9 + 4] + ls2 * rf_matrix[h * 9 + 7];
            lf_target_normal[h * 3 + 2] = ls0 * rf_matrix[h * 9 + 2] + ls1 * rf_matrix[h * 9 + 5] + ls2 * rf_matrix[h * 9 + 8];

            float rs0 = rf_normal_scalers[h * 3 + 0], rs1 = rf_normal_scalers[h * 3 + 1], rs2 = rf_normal_scalers[h * 3 + 2];
            rf_target_normal[h * 3 + 0] = rs0 * lf_matrix[h * 9 + 0] + rs1 * lf_matrix[h * 9 + 3] + rs2 * lf_matrix[h * 9 + 6];
            rf_target_normal[h * 3 + 1] = rs0 * lf_matrix[h * 9 + 1] + rs1 * lf_matrix[h * 9 + 4] + rs2 * lf_matrix[h * 9 + 7];
            rf_target_normal[h * 3 + 2] = rs0 * lf_matrix[h * 9 + 2] + rs1 * lf_matrix[h * 9 + 5] + rs2 * lf_matrix[h * 9 + 8];

            float ld = lnx * lf_target_normal[h * 3 + 0] + lny * lf_target_normal[h * 3 + 1] + lnz * lf_target_normal[h * 3 + 2];
            ld = (ld < -1.0f) ? -1.0f : (ld > 1.0f) ? 1.0f : ld;
            lf_bend_error[h] = std::sqrt(2.0f * (1.0f - ld));

            float rd = rnx * rf_target_normal[h * 3 + 0] + rny * rf_target_normal[h * 3 + 1] + rnz * rf_target_normal[h * 3 + 2];
            rd = (rd < -1.0f) ? -1.0f : (rd > 1.0f) ? 1.0f : rd;
            rf_bend_error[h] = std::sqrt(2.0f * (1.0f - rd));
        }

        for (int k = 0; k < lf_total; ++k) {
            int vert = lf_verts[k], face = lf_tiler[k], hinge = lf_matrix_tiler[k];
            int vidx = vert * 3, fidx = face * 3;
            float cvx = co[vidx + 0] - face_centers[fidx + 0];
            float cvy = co[vidx + 1] - face_centers[fidx + 1];
            float cvz = co[vidx + 2] - face_centers[fidx + 2];
            float dot = cvx * lf_target_normal[hinge * 3 + 0] + cvy * lf_target_normal[hinge * 3 + 1] + cvz * lf_target_normal[hinge * 3 + 2];
            float sd = -e_bend_force * lf_hinge_weights[k] * lf_bend_error[hinge] * dot;
            force_accumulator[vidx + 0] += face_normals[fidx + 0] * sd;
            force_accumulator[vidx + 1] += face_normals[fidx + 1] * sd;
            force_accumulator[vidx + 2] += face_normals[fidx + 2] * sd;
        }
        for (int k = 0; k < rf_total; ++k) {
            int vert = rf_verts[k], face = rf_tiler[k], hinge = rf_matrix_tiler[k];
            int vidx = vert * 3, fidx = face * 3;
            float cvx = co[vidx + 0] - face_centers[fidx + 0];
            float cvy = co[vidx + 1] - face_centers[fidx + 1];
            float cvz = co[vidx + 2] - face_centers[fidx + 2];
            float dot = cvx * rf_target_normal[hinge * 3 + 0] + cvy * rf_target_normal[hinge * 3 + 1] + cvz * rf_target_normal[hinge * 3 + 2];
            float sd = -e_bend_force * rf_hinge_weights[k] * rf_bend_error[hinge] * dot;
            force_accumulator[vidx + 0] += face_normals[fidx + 0] * sd;
            force_accumulator[vidx + 1] += face_normals[fidx + 1] * sd;
            force_accumulator[vidx + 2] += face_normals[fidx + 2] * sd;
        }

        float scale = bend_stabilize * bv;
        for (int v = 0; v < num_verts; ++v) {
            float m = force_multiplier[v] * scale;
            for (int axis = 0; axis < 3; ++axis) {
                float val = force_accumulator[v * 3 + axis] * m;
                if (std::isnan(val) || std::isinf(val)) val = 0.0f;
                co[v * 3 + axis] += val;
            }
        }
    }

    // =============================================================================
    // SEW BEND FORCE (matrix version with target angle support)
    // =============================================================================
    MC_API void compute_sew_bend_forces(
        float* co,
        float* force_accumulator,
        const int* panel1_v0,
        const int* panel1_v1,
        const int* lf_idx,
        const int* rf_idx,
        const float* lf_normal_scalers,
        const float* rf_normal_scalers,
        const int* lf_verts,
        const int* lf_tiler,
        const int* lf_matrix_tiler,
        int lf_total,
        const int* rf_verts,
        const int* rf_tiler,
        const int* rf_matrix_tiler,
        int rf_total,
        const int* merged_idx,
        const unsigned char* merged,
        const float* face_centers,
        const float* face_normals,
        const float* force_multiplier,
        int num_verts,
        int hc,
        float sew_bend_force,
        float bend_stabilize,
        float bv)
    {
        if (sew_bend_force == 0.0f || hc == 0) return;

        std::memset(force_accumulator, 0, sizeof(float) * num_verts * 3);

        std::vector<float> lf_matrix(hc * 9);
        std::vector<float> rf_matrix(hc * 9);
        std::vector<float> lf_target_normal(hc * 3);
        std::vector<float> rf_target_normal(hc * 3);
        std::vector<float> lf_bend_error(hc);
        std::vector<float> rf_bend_error(hc);
        std::vector<int>   hinge_active(hc, 0);

        for (int h = 0; h < hc; ++h) {
            if (!merged[merged_idx[h * 4 + 0]]) continue;
            if (!merged[merged_idx[h * 4 + 1]]) continue;
            if (!merged[merged_idx[h * 4 + 2]]) continue;
            if (!merged[merged_idx[h * 4 + 3]]) continue;
            hinge_active[h] = 1;

            int v0 = panel1_v0[h] * 3, v1 = panel1_v1[h] * 3;
            float hx = co[v1 + 0] - co[v0 + 0];
            float hy = co[v1 + 1] - co[v0 + 1];
            float hz = co[v1 + 2] - co[v0 + 2];
            float hlen = std::sqrt(hx * hx + hy * hy + hz * hz);
            float inv_h = (hlen > 1e-12f) ? 1.0f / hlen : 0.0f;
            hx *= inv_h; hy *= inv_h; hz *= inv_h;

            int lfi = lf_idx[h] * 3, rfi = rf_idx[h] * 3;
            float lnx = face_normals[lfi + 0], lny = face_normals[lfi + 1], lnz = face_normals[lfi + 2];
            float rnx = face_normals[rfi + 0], rny = face_normals[rfi + 1], rnz = face_normals[rfi + 2];

            // frames
            lf_matrix[h * 9 + 0] = hx; lf_matrix[h * 9 + 1] = hy; lf_matrix[h * 9 + 2] = hz;
            lf_matrix[h * 9 + 3] = lnx; lf_matrix[h * 9 + 4] = lny; lf_matrix[h * 9 + 5] = lnz;
            lf_matrix[h * 9 + 6] = hy * lnz - hz * lny;
            lf_matrix[h * 9 + 7] = hz * lnx - hx * lnz;
            lf_matrix[h * 9 + 8] = hx * lny - hy * lnx;

            rf_matrix[h * 9 + 0] = hx; rf_matrix[h * 9 + 1] = hy; rf_matrix[h * 9 + 2] = hz;
            rf_matrix[h * 9 + 3] = rnx; rf_matrix[h * 9 + 4] = rny; rf_matrix[h * 9 + 5] = rnz;
            rf_matrix[h * 9 + 6] = hy * rnz - hz * rny;
            rf_matrix[h * 9 + 7] = hz * rnx - hx * rnz;
            rf_matrix[h * 9 + 8] = hx * rny - hy * rnx;

            // target normals from scalers
            float ls0 = lf_normal_scalers[h * 3 + 0], ls1 = lf_normal_scalers[h * 3 + 1], ls2 = lf_normal_scalers[h * 3 + 2];
            lf_target_normal[h * 3 + 0] = ls0 * rf_matrix[h * 9 + 0] + ls1 * rf_matrix[h * 9 + 3] + ls2 * rf_matrix[h * 9 + 6];
            lf_target_normal[h * 3 + 1] = ls0 * rf_matrix[h * 9 + 1] + ls1 * rf_matrix[h * 9 + 4] + ls2 * rf_matrix[h * 9 + 7];
            lf_target_normal[h * 3 + 2] = ls0 * rf_matrix[h * 9 + 2] + ls1 * rf_matrix[h * 9 + 5] + ls2 * rf_matrix[h * 9 + 8];

            float rs0 = rf_normal_scalers[h * 3 + 0], rs1 = rf_normal_scalers[h * 3 + 1], rs2 = rf_normal_scalers[h * 3 + 2];
            rf_target_normal[h * 3 + 0] = rs0 * lf_matrix[h * 9 + 0] + rs1 * lf_matrix[h * 9 + 3] + rs2 * lf_matrix[h * 9 + 6];
            rf_target_normal[h * 3 + 1] = rs0 * lf_matrix[h * 9 + 1] + rs1 * lf_matrix[h * 9 + 4] + rs2 * lf_matrix[h * 9 + 7];
            rf_target_normal[h * 3 + 2] = rs0 * lf_matrix[h * 9 + 2] + rs1 * lf_matrix[h * 9 + 5] + rs2 * lf_matrix[h * 9 + 8];

            float ld = lnx * lf_target_normal[h * 3 + 0] + lny * lf_target_normal[h * 3 + 1] + lnz * lf_target_normal[h * 3 + 2];
            ld = (ld < -1.0f) ? -1.0f : (ld > 1.0f) ? 1.0f : ld;
            lf_bend_error[h] = std::sqrt(2.0f * (1.0f - ld));

            float rd = rnx * rf_target_normal[h * 3 + 0] + rny * rf_target_normal[h * 3 + 1] + rnz * rf_target_normal[h * 3 + 2];
            rd = (rd < -1.0f) ? -1.0f : (rd > 1.0f) ? 1.0f : rd;
            rf_bend_error[h] = std::sqrt(2.0f * (1.0f - rd));
        }

        for (int k = 0; k < lf_total; ++k) {
            int hinge = lf_matrix_tiler[k];
            if (!hinge_active[hinge]) continue;
            int vert = lf_verts[k], face = lf_tiler[k];
            int vidx = vert * 3, fidx = face * 3;
            float cvx = co[vidx + 0] - face_centers[fidx + 0];
            float cvy = co[vidx + 1] - face_centers[fidx + 1];
            float cvz = co[vidx + 2] - face_centers[fidx + 2];
            float dot = cvx * lf_target_normal[hinge * 3 + 0] + cvy * lf_target_normal[hinge * 3 + 1] + cvz * lf_target_normal[hinge * 3 + 2];
            float sd = -sew_bend_force * lf_bend_error[hinge] * dot;
            force_accumulator[vidx + 0] += face_normals[fidx + 0] * sd;
            force_accumulator[vidx + 1] += face_normals[fidx + 1] * sd;
            force_accumulator[vidx + 2] += face_normals[fidx + 2] * sd;
        }
        for (int k = 0; k < rf_total; ++k) {
            int hinge = rf_matrix_tiler[k];
            if (!hinge_active[hinge]) continue;
            int vert = rf_verts[k], face = rf_tiler[k];
            int vidx = vert * 3, fidx = face * 3;
            float cvx = co[vidx + 0] - face_centers[fidx + 0];
            float cvy = co[vidx + 1] - face_centers[fidx + 1];
            float cvz = co[vidx + 2] - face_centers[fidx + 2];
            float dot = cvx * rf_target_normal[hinge * 3 + 0] + cvy * rf_target_normal[hinge * 3 + 1] + cvz * rf_target_normal[hinge * 3 + 2];
            float sd = -sew_bend_force * rf_bend_error[hinge] * dot;
            force_accumulator[vidx + 0] += face_normals[fidx + 0] * sd;
            force_accumulator[vidx + 1] += face_normals[fidx + 1] * sd;
            force_accumulator[vidx + 2] += face_normals[fidx + 2] * sd;
        }

        float scale = bend_stabilize * bv;
        for (int v = 0; v < num_verts; ++v) {
            float m = force_multiplier[v] * scale;
            for (int axis = 0; axis < 3; ++axis) {
                float val = force_accumulator[v * 3 + axis] * m;
                if (std::isnan(val) || std::isinf(val)) val = 0.0f;
                co[v * 3 + axis] += val;
            }
        }
    }

    // =============================================================================
    // BEND ITERATION LOOP
    // =============================================================================
    MC_API void run_bend_solver(
        float* co,
        // face geometry
        float* face_centers_scratch,
        float* face_normals_scratch,
        const int* face_verts,
        const int* face_starts,
        int num_faces,
        // regular bend
        float* bend_force_accumulator,
        const int* hinge_v0,
        const int* hinge_v1,
        const int* lf_idx,
        const int* rf_idx,
        const float* lf_normal_scalers,
        const float* rf_normal_scalers,
        const int* lf_verts,
        const int* lf_tiler,
        const int* lf_matrix_tiler,
        const float* lf_hinge_weights,
        int lf_total,
        const int* rf_verts,
        const int* rf_tiler,
        const int* rf_matrix_tiler,
        const float* rf_hinge_weights,
        int rf_total,
        const float* force_multiplier,
        int hc,
        float e_bend_force,
        float bend_stabilize,
        // sew bend
        float* sew_bend_force_accumulator,
        const int* sew_panel1_v0,
        const int* sew_panel1_v1,
        const int* sew_lf_idx,
        const int* sew_rf_idx,
        const float* sew_lf_normal_scalers,
        const float* sew_rf_normal_scalers,
        const int* sew_lf_verts,
        const int* sew_lf_tiler,
        const int* sew_lf_matrix_tiler,
        int sew_lf_total,
        const int* sew_rf_verts,
        const int* sew_rf_tiler,
        const int* sew_rf_matrix_tiler,
        int sew_rf_total,
        const int* sew_merged_idx,
        const unsigned char* merged,
        const float* sew_force_multiplier,
        int sew_hc,
        float sew_bend_force,
        // spring stabilization
        float* sum_positions,
        const int* es_0,
        const int* es_1,
        const float* target_dists,
        const float* right_group_mult,
        const float* left_group_mult,
        const float* es_mult,
        int num_springs,
        // pin/sew fixup
        const float* start_co,
        const float* mc_pin,
        const unsigned char* selected,
        int num_verts,
        // iteration values
        const float* bend_values,
        int num_bend_iters,
        // sew force params
        const int* basic_sew_verts,
        const int* sew_key_idxer,
        const float* sew_key_multiplier,
        float* group_adder,
        int num_sew_verts,
        int num_groups,
        float basic_merge_limit,
        float butt_sew_force,
        float sew_force)
    {
        for (int it = 0; it < num_bend_iters; ++it) {
            compute_face_centers(co, face_verts, face_starts, num_faces, face_centers_scratch);
            compute_face_normals(co, face_verts, face_starts, num_faces, face_normals_scratch);

            compute_bend_forces(
                co, bend_force_accumulator,
                hinge_v0, hinge_v1, lf_idx, rf_idx,
                lf_normal_scalers, rf_normal_scalers,
                lf_verts, lf_tiler, lf_matrix_tiler, lf_hinge_weights, lf_total,
                rf_verts, rf_tiler, rf_matrix_tiler, rf_hinge_weights, rf_total,
                face_centers_scratch, face_normals_scratch, force_multiplier,
                num_verts, hc, e_bend_force, bend_stabilize, bend_values[it]
            );

            if (sew_hc > 0) {
                compute_sew_bend_forces(
                    co, sew_bend_force_accumulator,
                    sew_panel1_v0, sew_panel1_v1,
                    sew_lf_idx, sew_rf_idx,
                    sew_lf_normal_scalers, sew_rf_normal_scalers,
                    sew_lf_verts, sew_lf_tiler, sew_lf_matrix_tiler, sew_lf_total,
                    sew_rf_verts, sew_rf_tiler, sew_rf_matrix_tiler, sew_rf_total,
                    sew_merged_idx, merged,
                    face_centers_scratch, face_normals_scratch, sew_force_multiplier,
                    num_verts, sew_hc, sew_bend_force, bend_stabilize, bend_values[it]
                );
            }

            linear_springs(
                co, sum_positions,
                es_0, es_1, target_dists,
                right_group_mult, left_group_mult, es_mult,
                num_verts, num_springs,
                1.0f, 1.0f
            );

            update_pin_select_sew(
                co, start_co, mc_pin, selected, num_verts,
                basic_sew_verts, sew_key_idxer, sew_key_multiplier,
                group_adder, num_sew_verts, num_groups,
                basic_merge_limit, butt_sew_force, sew_force
            );
        }
    }

} // extern "C"