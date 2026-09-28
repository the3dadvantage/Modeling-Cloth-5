// mc_collide -- native collision for Modeling Cloth.
//
// A port of the collision code: object_collide.py's broad phase and narrow
// resolve, and self_collide_2.py's candidate search and relaxation loop.  The
// Python modules stay the reference and the single source of every tunable:
// they arrive in OcParams / ScParams, so the two cannot drift apart.  The
// Python bridge is collide_native.py; tests/shadow_compare.py checks the two
// agree call by call.
//
// Conventions
//   * All geometry is computed in double.  Cloth positions live in the
//     caller's float32 array and are rounded back to float after every pass,
//     exactly as the Python does (C.co += step.astype(float32)).
//   * Contacts are generated and combined in the same order as the Python so
//     ties break the same way.
//   * Pointers documented as optional may be null.
//
// Build: build.bat (MSVC, /O2).  Output: ..\..\Python\mc_collide.dll

#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>
#include <algorithm>
#include <limits>

// Exporting a symbol is spelled differently per compiler; everything else here
// is plain C++17, so this is all that stands between it and a mac / linux build.
#if defined(_WIN32)
  #define API extern "C" __declspec(dllexport)
#else
  #define API extern "C" __attribute__((visibility("default")))
#endif

// Bump whenever any exported signature or struct layout changes.  The bridge
// refuses a DLL whose version or struct sizes differ from what it expects.
static const int32_t MC_COLLIDE_ABI = 4;

// ---------------------------------------------------------------- vectors
struct V3 {
    double x, y, z;
    V3() : x(0), y(0), z(0) {}
    V3(double a, double b, double c) : x(a), y(b), z(c) {}
};
static inline V3 operator+(V3 a, V3 b) { return V3(a.x + b.x, a.y + b.y, a.z + b.z); }
static inline V3 operator-(V3 a, V3 b) { return V3(a.x - b.x, a.y - b.y, a.z - b.z); }
static inline V3 operator*(V3 a, double s) { return V3(a.x * s, a.y * s, a.z * s); }
static inline double dot(V3 a, V3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
static inline double len(V3 a) { return std::sqrt(dot(a, a)); }
static inline V3 ldf(const float* p, int64_t i) { return V3(p[3 * i], p[3 * i + 1], p[3 * i + 2]); }
static inline V3 ldd(const double* p, int64_t i) { return V3(p[3 * i], p[3 * i + 1], p[3 * i + 2]); }
static inline bool finite3(V3 a) { return std::isfinite(a.x) && std::isfinite(a.y) && std::isfinite(a.z); }

// ---------------------------------------------------------------- inputs
// Tunables and per-call switches.  Mirrors collide_native.OcParams.
struct OcParams {
    double thickness, relax, push_cap, swept_margin;
    double clamp_thick, clamp_edge, geom_rel_eps, geom_tiny, eps;
    double edge_sep_min, friction_floor, friction_maxmove;
    double tri_damp, edge_damp, mu, static_thresh, friction_max;
    int32_t iters, contact_rounds, do_pt, do_tp, do_ee, do_friction, swept, pbd_exact;
};

// Everything the contacts read.  Mirrors collide_native.OcMesh.
struct OcMesh {
    // cloth
    float* co;                  // (nv,3) in/out
    const float* start_co;      // (nv,3)
    const int32_t* tridex;      // (nt,3)
    const int32_t* cl_eidx;     // (ne,2) cloth edge endpoints (C.tri_eidx)
    const double* char_len;     // (nv)   optional: per-vertex edge length for the move clamp
    const double* fric_group;   // (nv)   optional: cloth MC_friction weights
    int32_t nv, nt, ne;
    // collider (joined over every collider, in the cloth's local space)
    const double* trico1;       // (to,3,3) triangles now
    const double* trico0;       // (to,3,3) triangles at step start
    const double* tnorm1;       // (to,3)
    const double* tnorm0;       // (to,3)
    const double* vnorm;        // (vo,3) vertex normals now
    const double* joined;       // (vo,2,3) vertex positions [start, now]
    const int32_t* ob_eidx;     // (eo,2) collider edge endpoints
    const double* enorm1;       // (eo,3) edge normals now
    const double* enorm0;       // (eo,3) edge normals at step start
    const double* fric_tri;     // (to) optional
    const double* fric_vert;    // (vo) optional
    const double* fric_edge;    // (eo) optional
    int32_t to, vo, eo;
    // candidate pairs from the broad phase
    const int32_t* pt_v; const int32_t* pt_t; int32_t npt;   // cloth vert , collider tri
    const int32_t* tp_v; const int32_t* tp_t; int32_t ntp;   // collider vert, cloth tri
    const int32_t* ee_c; const int32_t* ee_o; int32_t nee;   // cloth edge , collider edge
};

API int32_t mc_collide_abi() { return MC_COLLIDE_ABI; }
API int32_t mc_collide_sizeof_params() { return (int32_t)sizeof(OcParams); }
API int32_t mc_collide_sizeof_mesh() { return (int32_t)sizeof(OcMesh); }

// ---------------------------------------------------------------- geometry
// inside_triangles: barycentric weights (w, u, v) of p in triangle abc and
// whether all are >= margin.  Degeneracy relative to the triangle's own size.
static bool inside_tri(V3 a, V3 b, V3 c, V3 p, double margin, double rel_eps, double w[3]) {
    V3 v0 = b - a, v1 = c - a, v2 = p - a;
    double d00 = dot(v0, v0), d11 = dot(v1, v1), d01 = dot(v0, v1);
    double d02 = dot(v0, v2), d12 = dot(v1, v2);
    double den = d00 * d11 - d01 * d01;
    bool degen = !(den > rel_eps * d00 * d11);
    if (degen) { w[0] = w[1] = w[2] = 1.0 / 3.0; return false; }
    double inv = 1.0 / den;
    double u = (d11 * d02 - d01 * d12) * inv;
    double v = (d00 * d12 - d01 * d02) * inv;
    double ww = 1.0 - (u + v);
    w[0] = ww; w[1] = u; w[2] = v;
    return (u >= margin) && (v >= margin) && (ww >= margin);
}

static inline V3 bary(V3 a, V3 b, V3 c, const double w[3]) {
    return a * w[0] + b * w[1] + c * w[2];
}

static void clip_weights(const double w[3], double o[3]) {
    double s = 0.0;
    for (int k = 0; k < 3; ++k) { o[k] = std::min(1.0, std::max(0.0, w[k])); s += o[k]; }
    if (s > 1e-12) { for (int k = 0; k < 3; ++k) o[k] /= s; }
    else { o[0] = o[1] = o[2] = 1.0 / 3.0; }
}

static inline double clip01(double x) { return std::min(1.0, std::max(0.0, x)); }

// closest_point_segments (Ericson), relative parallel test.
static void closest_segments(V3 p1, V3 q1, V3 p2, V3 q2, double tiny, double rel_eps,
                             V3& cpa, V3& cpb, double& s, double& t) {
    V3 d1 = q1 - p1, d2 = q2 - p2, r = p1 - p2;
    double a = dot(d1, d1), e = dot(d2, d2), f = dot(d2, r), c = dot(d1, r), b = dot(d1, d2);
    bool pt_a = a <= tiny, pt_e = e <= tiny;
    double a_safe = pt_a ? 1.0 : a, e_safe = pt_e ? 1.0 : e;
    double denom = a * e - b * b;
    bool nonpar = denom > rel_eps * a * e;
    s = nonpar ? clip01((b * f - c * e) / denom) : 0.0;
    t = (b * s + f) / e_safe;
    bool t_lt0 = t < 0.0, t_gt1 = t > 1.0;
    t = clip01(t);
    if (t_lt0) s = clip01(-c / a_safe);
    if (t_gt1) s = clip01((b - c) / a_safe);
    if (pt_a) { s = 0.0; t = clip01(f / e_safe); }
    if (pt_e && !pt_a) { s = clip01(-c / a_safe); t = 0.0; }
    if (pt_a && pt_e) { s = 0.0; t = 0.0; }
    cpa = p1 + d1 * s;
    cpb = p2 + d2 * t;
}

static V3 closest_on_boundary(V3 T[3], V3 p, const OcParams& P) {
    static const int ed[3][2] = {{0, 1}, {1, 2}, {2, 0}};
    V3 best; double best_d2 = 0.0;
    for (int k = 0; k < 3; ++k) {
        V3 A = T[ed[k][0]], B = T[ed[k][1]], AB = B - A;
        double ab2 = dot(AB, AB);
        double t = dot(p - A, AB) / std::max(ab2, P.geom_tiny);
        V3 q = A + AB * clip01(t);
        V3 dq = p - q;
        double d2 = dot(dq, dq);
        if (k == 0 || d2 < best_d2) { best = q; best_d2 = d2; }
    }
    return best;
}

// _face_contact: is p in this face's zone, its distance and push direction.
static bool face_contact(V3 T[3], V3 n, V3 p, const OcParams& P,
                         double& dist, V3& normal, double w[3]) {
    double d = dot(p - T[0], n);
    bool strict = inside_tri(T[0], T[1], T[2], p, 0.0, P.geom_rel_eps, w);
    double ws[3];
    bool slack = inside_tri(T[0], T[1], T[2], p, -P.swept_margin, P.geom_rel_eps, ws);
    bool edge = slack && !strict && (d > 0.0);
    dist = d; normal = n;
    if (edge) {
        V3 cp = closest_on_boundary(T, p, P);
        V3 v = p - cp;
        double ln = len(v);
        if (ln > P.eps) { dist = ln; normal = v * (1.0 / ln); }
    }
    return strict || edge;
}

// _tip_weights: cloth triangle over collider vertex P, looking along vn.
static bool tip_weights(V3 T[3], V3 Pv, V3 vn, double margin, const OcParams& P, double w[3]) {
    V3 f[3];
    for (int k = 0; k < 3; ++k) f[k] = T[k] - vn * dot(T[k], vn);
    V3 fp = Pv - vn * dot(Pv, vn);
    return inside_tri(f[0], f[1], f[2], fp, margin, P.geom_rel_eps, w);
}

// _edge_contact: signed distance / direction along the real separation.
static void edge_contact(V3 cpa, V3 cpb, V3 en, const OcParams& P,
                         double& dist, V3& normal, bool& ok) {
    double enl = len(en);
    ok = enl > P.eps;
    if (ok) en = en * (1.0 / enl);
    V3 sep = cpa - cpb;
    double sl = len(sep);
    double side = dot(sep, en) < 0.0 ? -1.0 : 1.0;
    bool real = sl > P.edge_sep_min * P.thickness;
    if (real) { normal = sep * (side / sl); dist = sl * side; }
    else { normal = en; dist = dot(sep, en); }
    ok = ok || real;
}

static V3 friction_correction(V3 c1, V3 c0, V3 a1, V3 a0, V3 n, double normal_push,
                              double mu, const OcParams& P) {
    V3 d_rel = (c1 - c0) - (a1 - a0);
    V3 d_t = d_rel - n * dot(d_rel, n);
    double mag = len(d_t);
    double budget;
    if (P.static_thresh > 0.0) budget = mu > 0.0 ? P.static_thresh : 0.0;
    else budget = mu * std::max(normal_push, P.friction_floor * P.thickness);
    double remove_mag = std::min(mag, budget);
    double remove = mag > P.eps ? remove_mag / std::max(mag, P.eps) : 0.0;
    V3 fric = d_t * (-remove);
    double fm = len(fric), cap = P.friction_maxmove * P.thickness;
    if (fm > cap) fric = fric * (cap / fm);
    return fric;
}

static inline double max_move(const OcMesh& M, const OcParams& P, int32_t v) {
    double base = P.clamp_thick * P.thickness;
    if (M.char_len) return std::max(base, P.clamp_edge * M.char_len[v]);
    return std::max(base, P.clamp_edge * 10.0 * P.thickness);
}

static inline V3 clamp_len(V3 m, double cap) {
    double l = len(m);
    return l > cap ? m * (cap / l) : m;
}

// ---------------------------------------------------------------- contacts
// One row per (vertex, contact).  cid groups the rows of one contact; a is
// that vertex's weight in it (1 for a single-vertex contact).
struct Rows {
    std::vector<int32_t> v;
    std::vector<V3> m;
    std::vector<int32_t> cid;
    std::vector<double> a;
    int32_t next_cid = 0;
    void clear() { v.clear(); m.clear(); cid.clear(); a.clear(); next_cid = 0; }
    void push(int32_t vv, V3 mm, int32_t c, double aa) { v.push_back(vv); m.push_back(mm); cid.push_back(c); a.push_back(aa); }
};

struct FricRows {       // friction: (vertex, move), averaged per vertex
    std::vector<int32_t> v;
    std::vector<V3> m;
    void clear() { v.clear(); m.clear(); }
};

static inline double cfg_at(const OcMesh& M, int32_t v) { return M.fric_group ? M.fric_group[v] : 1.0; }

// point_to_triangle: cloth vertex vs collider face
static void point_to_triangle(const OcMesh& M, const OcParams& P, const double* tstep,
                              Rows* out, FricRows* fout) {
    const double th = P.thickness;
    for (int32_t k = 0; k < M.npt; ++k) {
        int32_t vi = M.pt_v[k], ti = M.pt_t[k];
        V3 T1[3] = {ldd(M.trico1, 3 * (int64_t)ti), ldd(M.trico1, 3 * (int64_t)ti + 1), ldd(M.trico1, 3 * (int64_t)ti + 2)};
        V3 n1 = ldd(M.tnorm1, ti);
        V3 p1 = ldf(M.co, vi);
        double d1 = dot(p1 - T1[0], n1);
        double dist1, w_pt[3]; V3 nrm1;
        bool zone = face_contact(T1, n1, p1, P, dist1, nrm1, w_pt);
        bool prox = zone && (dist1 < th) && (dist1 > -P.push_cap * th);
        bool swept = false;
        if (P.swept) {
            V3 T0[3] = {ldd(M.trico0, 3 * (int64_t)ti), ldd(M.trico0, 3 * (int64_t)ti + 1), ldd(M.trico0, 3 * (int64_t)ti + 2)};
            V3 n0 = ldd(M.tnorm0, ti);
            V3 p0 = ldf(M.start_co, vi);
            double d0 = dot(p0 - T0[0], n0);
            bool deep = (d0 > th) && (d1 < -th);
            if (deep) {
                double tau = clip01(d0 / (d0 - d1));
                V3 pT = p0 + (p1 - p0) * tau;
                V3 TT[3];
                for (int j = 0; j < 3; ++j) TT[j] = T0[j] + (T1[j] - T0[j]) * tau;
                double wT[3];
                bool checkT = inside_tri(TT[0], TT[1], TT[2], pT, -P.swept_margin, P.geom_rel_eps, wT);
                swept = checkT;
                if (swept && !prox) { w_pt[0] = wT[0]; w_pt[1] = wT[1]; w_pt[2] = wT[2]; }
            }
        }
        if (!(prox || swept)) continue;
        double dd = prox ? dist1 : d1;
        V3 nn = prox ? nrm1 : n1;
        double push = std::min(P.push_cap * th, std::max(0.0, th - dd));
        if (fout) {
            double wh[3]; clip_weights(w_pt, wh);
            V3 a1 = bary(T1[0], T1[1], T1[2], wh);
            V3 T0[3] = {ldd(M.trico0, 3 * (int64_t)ti), ldd(M.trico0, 3 * (int64_t)ti + 1), ldd(M.trico0, 3 * (int64_t)ti + 2)};
            V3 a0 = bary(T0[0], T0[1], T0[2], wh);
            double push_ref = std::fabs(dot(ldd(tstep, vi), nn));
            double mu = P.mu * cfg_at(M, vi) * (M.fric_tri ? M.fric_tri[ti] : 1.0);
            fout->v.push_back(vi);
            fout->m.push_back(friction_correction(ldf(M.co, vi), ldf(M.start_co, vi), a1, a0, nn, push_ref, mu, P));
            continue;
        }
        V3 move = clamp_len(nn * push, max_move(M, P, vi));
        out->push(vi, move, out->next_cid++, 1.0);
    }
}

// triangle_to_point: collider vertex poking a cloth triangle
static void triangle_to_point(const OcMesh& M, const OcParams& P, const double* tstep,
                              Rows* out, FricRows* fout) {
    const double th = P.thickness;
    for (int32_t k = 0; k < M.ntp; ++k) {
        int32_t ov = M.tp_v[k], ct = M.tp_t[k];
        const int32_t* tv = M.tridex + 3 * (int64_t)ct;
        V3 tco1[3] = {ldf(M.co, tv[0]), ldf(M.co, tv[1]), ldf(M.co, tv[2])};
        V3 vn = ldd(M.vnorm, ov);
        double vl = len(vn);
        vn = vn * (1.0 / (vl > P.eps ? vl : 1.0));
        V3 P1 = ldd(M.joined, 2 * (int64_t)ov + 1);
        double w[3];
        bool check = tip_weights(tco1, P1, vn, -P.swept_margin, P, w);
        V3 plot1 = bary(tco1[0], tco1[1], tco1[2], w);
        double d1 = dot(P1 - plot1, vn);
        bool prox = check && (d1 > -th) && (d1 < th);
        bool swept = false;
        V3 P0 = ldd(M.joined, 2 * (int64_t)ov);
        if (P.swept) {
            V3 tco0[3] = {ldf(M.start_co, tv[0]), ldf(M.start_co, tv[1]), ldf(M.start_co, tv[2])};
            V3 plot0 = bary(tco0[0], tco0[1], tco0[2], w);
            double d0 = dot(P0 - plot0, vn);
            swept = check && (d0 < -th) && (d1 > th);
        }
        if (!(prox || swept)) continue;
        double wh[3]; clip_weights(w, wh);
        double push = std::min(P.push_cap * th, std::max(0.0, d1 + th));
        if (fout) {
            V3 c1 = bary(tco1[0], tco1[1], tco1[2], wh);
            V3 s0[3] = {ldf(M.start_co, tv[0]), ldf(M.start_co, tv[1]), ldf(M.start_co, tv[2])};
            V3 c0 = bary(s0[0], s0[1], s0[2], wh);
            V3 ts = bary(ldd(tstep, tv[0]), ldd(tstep, tv[1]), ldd(tstep, tv[2]), wh);
            double push_ref = std::fabs(dot(ts, vn));
            double mu = P.mu * (cfg_at(M, tv[0]) + cfg_at(M, tv[1]) + cfg_at(M, tv[2])) / 3.0
                        * (M.fric_vert ? M.fric_vert[ov] : 1.0);
            V3 fm = friction_correction(c1, c0, P1, P0, vn, push_ref, mu, P);
            for (int j = 0; j < 3; ++j) { fout->v.push_back(tv[j]); fout->m.push_back(fm * wh[j]); }
            continue;
        }
        V3 move = clamp_len(vn * push, max_move(M, P, tv[0]));
        double wsq = wh[0] * wh[0] + wh[1] * wh[1] + wh[2] * wh[2];
        int32_t c = out->next_cid++;
        for (int j = 0; j < 3; ++j) {
            double wt = wh[j] * P.tri_damp;
            if (P.pbd_exact) wt /= wsq;
            out->push(tv[j], move * wt, c, wh[j]);
        }
    }
}

// edge_to_edge: cloth edge vs collider edge.  Rows go [all first endpoints,
// all second endpoints], as the Python concatenates them.
static void edge_to_edge(const OcMesh& M, const OcParams& P, const double* tstep,
                         Rows* out, FricRows* fout) {
    const double th = P.thickness;
    struct Hit { int32_t a0, a1, b0, b1, oe; double s, t, push; V3 udir, cpa, cpb; };
    std::vector<Hit> hits;
    for (int32_t k = 0; k < M.nee; ++k) {
        int32_t ce = M.ee_c[k], oe = M.ee_o[k];
        int32_t a0 = M.cl_eidx[2 * (int64_t)ce], a1 = M.cl_eidx[2 * (int64_t)ce + 1];
        int32_t b0 = M.ob_eidx[2 * (int64_t)oe], b1 = M.ob_eidx[2 * (int64_t)oe + 1];
        V3 cpa, cpb; double s, t;
        closest_segments(ldf(M.co, a0), ldf(M.co, a1), ldd(M.joined, 2 * (int64_t)b0 + 1),
                         ldd(M.joined, 2 * (int64_t)b1 + 1), P.geom_tiny, P.geom_rel_eps, cpa, cpb, s, t);
        double dist1 = len(cpa - cpb);
        bool interior = (s > 1e-3) && (s < 1.0 - 1e-3) && (t > 1e-3) && (t < 1.0 - 1e-3);
        double d1; V3 udir; bool good;
        edge_contact(cpa, cpb, ldd(M.enorm1, oe), P, d1, udir, good);
        bool prox = interior && good && (dist1 < th);
        bool swept = false;
        if (P.swept) {
            V3 scpa, scpb; double ss, st;
            closest_segments(ldf(M.start_co, a0), ldf(M.start_co, a1), ldd(M.joined, 2 * (int64_t)b0),
                             ldd(M.joined, 2 * (int64_t)b1), P.geom_tiny, P.geom_rel_eps, scpa, scpb, ss, st);
            double d0; V3 un0; bool g0;
            edge_contact(scpa, scpb, ldd(M.enorm0, oe), P, d0, un0, g0);
            double sg0 = (d0 > 0) - (d0 < 0), sg1 = (d1 > 0) - (d1 < 0);   // np.sign
            swept = interior && (sg0 != sg1) && (dist1 < P.push_cap * th) &&
                    (std::min(std::fabs(d0), std::fabs(d1)) > 0.25 * th);
        }
        if (!(prox || swept)) continue;
        Hit h;
        h.a0 = a0; h.a1 = a1; h.b0 = b0; h.b1 = b1; h.oe = oe; h.s = s; h.t = t;
        h.push = std::min(P.push_cap * th, std::max(0.0, th - d1));
        h.udir = udir; h.cpa = cpa; h.cpb = cpb;
        hits.push_back(h);
    }
    const size_t n = hits.size();
    if (!n) return;
    std::vector<double> w1(n), w2(n);
    for (size_t i = 0; i < n; ++i) {
        double s = hits[i].s;
        if (P.pbd_exact) { double da = (1.0 - s) * (1.0 - s) + s * s + P.eps; w1[i] = (1.0 - s) / da; w2[i] = s / da; }
        else { w1[i] = 1.0 - s; w2[i] = s; }
    }
    if (fout) {
        std::vector<V3> fm(n);
        for (size_t i = 0; i < n; ++i) {
            const Hit& h = hits[i];
            V3 c0 = ldf(M.start_co, h.a0) * (1.0 - h.s) + ldf(M.start_co, h.a1) * h.s;
            V3 a0p = ldd(M.joined, 2 * (int64_t)h.b0) * (1.0 - h.t) + ldd(M.joined, 2 * (int64_t)h.b1) * h.t;
            V3 ts = ldd(tstep, h.a0) * (1.0 - h.s) + ldd(tstep, h.a1) * h.s;
            double push_ref = std::fabs(dot(ts, h.udir));
            double mu = P.mu * 0.5 * (cfg_at(M, h.a0) + cfg_at(M, h.a1)) * (M.fric_edge ? M.fric_edge[h.oe] : 1.0);
            fm[i] = friction_correction(h.cpa, c0, h.cpb, a0p, h.udir, push_ref, mu, P);
        }
        for (size_t i = 0; i < n; ++i) { fout->v.push_back(hits[i].a0); fout->m.push_back(fm[i] * w1[i]); }
        for (size_t i = 0; i < n; ++i) { fout->v.push_back(hits[i].a1); fout->m.push_back(fm[i] * w2[i]); }
        return;
    }
    const int32_t base = out->next_cid;
    for (size_t i = 0; i < n; ++i) {
        V3 f = hits[i].udir * (hits[i].push * w1[i] * P.edge_damp);
        out->push(hits[i].a0, clamp_len(f, max_move(M, P, hits[i].a0)), base + (int32_t)i, 1.0 - hits[i].s);
    }
    for (size_t i = 0; i < n; ++i) {
        V3 f = hits[i].udir * (hits[i].push * w2[i] * P.edge_damp);
        out->push(hits[i].a1, clamp_len(f, max_move(M, P, hits[i].a1)), base + (int32_t)i, hits[i].s);
    }
    out->next_cid = base + (int32_t)n;
}

// ---------------------------------------------------------------- combine
// _project_contacts: per vertex, the move that meets all its contacts.
static void project_contacts(int32_t nv, const Rows& R, const OcParams& P,
                             std::vector<V3>& x, std::vector<uint8_t>& hit) {
    std::fill(x.begin(), x.end(), V3());
    std::fill(hit.begin(), hit.end(), 0);
    const size_t nr = R.v.size();
    if (!nr) return;
    for (size_t r = 0; r < nr; ++r) hit[R.v[r]] = 1;

    std::vector<size_t> live;
    live.reserve(nr);
    std::vector<double> ml(nr);
    for (size_t r = 0; r < nr; ++r) {
        V3 m = R.m[r];
        if (!finite3(m)) m = V3();          // np.nan_to_num
        ml[r] = len(m);
        if (ml[r] > P.geom_tiny && R.a[r] > P.geom_tiny) live.push_back(r);
    }
    if (live.empty()) return;
    const int32_t nc = R.next_cid;
    std::vector<double> asq(nc, 0.0), push(nc, 0.0), moved(nc);
    std::vector<V3> n(nr);
    std::vector<double> share(nr);
    for (size_t r : live) { asq[R.cid[r]] += R.a[r] * R.a[r]; n[r] = R.m[r] * (1.0 / ml[r]); }
    for (size_t r : live) push[R.cid[r]] = std::max(push[R.cid[r]], ml[r] * asq[R.cid[r]] / R.a[r]);
    for (size_t r : live) share[r] = R.a[r] / asq[R.cid[r]];

    std::vector<double> best(nv);
    std::vector<int64_t> pick(nv);
    std::vector<double> corr(nr);
    for (int32_t round = 0; round < P.contact_rounds; ++round) {
        std::fill(moved.begin(), moved.end(), 0.0);
        for (size_t r : live) moved[R.cid[r]] += R.a[r] * dot(x[R.v[r]], n[r]);
        bool any = false;
        std::fill(best.begin(), best.end(), -std::numeric_limits<double>::infinity());
        std::fill(pick.begin(), pick.end(), -1);
        for (size_t r : live) {
            corr[r] = share[r] * (push[R.cid[r]] - moved[R.cid[r]]);
            if (corr[r] > P.eps && corr[r] > best[R.v[r]]) {   // first of equals wins
                best[R.v[r]] = corr[r];
                pick[R.v[r]] = (int64_t)r;
                any = true;
            }
        }
        if (!any) break;
        for (int32_t v = 0; v < nv; ++v)
            if (pick[v] >= 0) x[v] = x[v] + n[pick[v]] * corr[pick[v]];
    }
}

// ---------------------------------------------------------------- broad phase
// Every (a, b) whose boxes come within `margin`, sorted by (a, b).
//
// The test is the Python _aabb_pairs one, done in float32 exactly as numpy
// does it (the boxes are float32 and numpy keeps a Python-float margin in
// float32): separated on an axis when  amin - m > bmax  or  amax + m < bmin.
// Same arithmetic, same comparisons, so the pair lists are identical, not
// merely close.  The search is a sort-and-sweep on one axis; the pruning
// there is conservative and every candidate still gets the exact test.
static std::vector<int32_t> g_pa, g_pb;

API int64_t mc_box_pairs(const float* amin, const float* amax, int32_t na,
                         const float* bmin, const float* bmax, int32_t nb, float margin) {
    g_pa.clear();
    g_pb.clear();
    if (na <= 0 || nb <= 0) return 0;
    // sweep along the axis where the b boxes are most spread out
    int ax = 0;
    {
        double best = -1.0;
        for (int k = 0; k < 3; ++k) {
            float lo = std::numeric_limits<float>::max(), hi = -lo;
            for (int32_t j = 0; j < nb; ++j) { lo = std::min(lo, bmin[3 * (int64_t)j + k]); hi = std::max(hi, bmin[3 * (int64_t)j + k]); }
            if (hi - lo > best) { best = hi - lo; ax = k; }
        }
    }
    std::vector<int32_t> order(nb);
    for (int32_t j = 0; j < nb; ++j) order[j] = j;
    std::sort(order.begin(), order.end(), [&](int32_t p, int32_t q) {
        float x = bmin[3 * (int64_t)p + ax], y = bmin[3 * (int64_t)q + ax];
        return x < y || (x == y && p < q);
    });
    std::vector<float> smin(nb);
    double max_ext = 0.0;
    for (int32_t j = 0; j < nb; ++j) {
        int32_t b = order[j];
        smin[j] = bmin[3 * (int64_t)b + ax];
        max_ext = std::max(max_ext, (double)bmax[3 * (int64_t)b + ax] - (double)bmin[3 * (int64_t)b + ax]);
    }

    // single-threaded on purpose: OpenMP's idle workers spin after a
    // parallel region and slowed the numpy self collision that runs next by
    // 60%, far more than this search costs
    std::vector<std::vector<int32_t>> per_a(na);
    for (int32_t i = 0; i < na; ++i) {
        const float* a0 = amin + 3 * (int64_t)i;
        const float* a1 = amax + 3 * (int64_t)i;
        // any b that can reach this a has bmin >= a0 - margin - (its extent);
        // widened a little so float rounding can never drop a real pair
        double start_at = (double)a0[ax] - (double)margin - max_ext - 1e-6 * (1.0 + std::fabs(a0[ax]) + max_ext);
        auto it = std::lower_bound(smin.begin(), smin.end(), start_at,
                                   [](float v, double s) { return (double)v < s; });
        std::vector<int32_t>& hits = per_a[i];
        for (int64_t j = it - smin.begin(); j < nb; ++j) {
            int32_t b = order[j];
            const float* b0 = bmin + 3 * (int64_t)b;
            const float* b1 = bmax + 3 * (int64_t)b;
            if (a1[ax] + margin < b0[ax]) break;        // sorted by bmin: nothing further reaches
            bool sep = false;
            for (int k = 0; k < 3 && !sep; ++k)
                sep = (a0[k] - margin > b1[k]) || (a1[k] + margin < b0[k]);
            if (!sep) hits.push_back(b);
        }
        std::sort(hits.begin(), hits.end());
    }
    size_t total = 0;
    for (auto& h : per_a) total += h.size();
    g_pa.reserve(total);
    g_pb.reserve(total);
    for (int32_t i = 0; i < na; ++i)
        for (int32_t b : per_a[i]) { g_pa.push_back(i); g_pb.push_back(b); }
    return (int64_t)total;
}

// Copy out the pairs found by the last mc_box_pairs call.
API void mc_box_pairs_fetch(int32_t* a, int32_t* b) {
    if (!g_pa.empty()) {
        std::memcpy(a, g_pa.data(), sizeof(int32_t) * g_pa.size());
        std::memcpy(b, g_pb.data(), sizeof(int32_t) * g_pb.size());
    }
}

// ---------------------------------------------------------------- recollide
// near_pairs: which of the cached pairs are within `reach` of contact right
// now.  Writes one keep flag per pair of each kind.  Same tests as the Python.
API void oc_near_pairs(const OcParams* Pp, const OcMesh* Mp, double reach,
                       uint8_t* keep_pt, uint8_t* keep_tp, uint8_t* keep_ee) {
    const OcParams& P = *Pp;
    const OcMesh& M = *Mp;
    const double th = P.thickness;
    auto box_near = [&](V3 p, const V3* T, int n) {
        double lo[3] = {T[0].x, T[0].y, T[0].z}, hi[3] = {T[0].x, T[0].y, T[0].z};
        for (int j = 1; j < n; ++j) {
            lo[0] = std::min(lo[0], T[j].x); hi[0] = std::max(hi[0], T[j].x);
            lo[1] = std::min(lo[1], T[j].y); hi[1] = std::max(hi[1], T[j].y);
            lo[2] = std::min(lo[2], T[j].z); hi[2] = std::max(hi[2], T[j].z);
        }
        return p.x >= lo[0] - reach && p.x <= hi[0] + reach &&
               p.y >= lo[1] - reach && p.y <= hi[1] + reach &&
               p.z >= lo[2] - reach && p.z <= hi[2] + reach;
    };
    for (int32_t k = 0; k < M.npt; ++k) {
        int32_t v = M.pt_v[k], t = M.pt_t[k];
        V3 T[3] = {ldd(M.trico1, 3 * (int64_t)t), ldd(M.trico1, 3 * (int64_t)t + 1), ldd(M.trico1, 3 * (int64_t)t + 2)};
        V3 p = ldf(M.co, v);
        double d = dot(p - T[0], ldd(M.tnorm1, t));
        keep_pt[k] = (d < reach) && (d > -(P.push_cap * th + reach)) && box_near(p, T, 3);
    }
    for (int32_t k = 0; k < M.ntp; ++k) {
        int32_t ov = M.tp_v[k], ct = M.tp_t[k];
        const int32_t* tv = M.tridex + 3 * (int64_t)ct;
        V3 Pv = ldd(M.joined, 2 * (int64_t)ov + 1);
        V3 T[3] = {ldf(M.co, tv[0]), ldf(M.co, tv[1]), ldf(M.co, tv[2])};
        V3 c = V3((T[1] - T[0]).y * (T[2] - T[0]).z - (T[1] - T[0]).z * (T[2] - T[0]).y,
                  (T[1] - T[0]).z * (T[2] - T[0]).x - (T[1] - T[0]).x * (T[2] - T[0]).z,
                  (T[1] - T[0]).x * (T[2] - T[0]).y - (T[1] - T[0]).y * (T[2] - T[0]).x);
        double cl = len(c);
        V3 n = c * (1.0 / (cl > P.eps ? cl : 1.0));
        double d = dot(Pv - T[0], n);
        keep_tp[k] = (std::fabs(d) < reach) && box_near(Pv, T, 3);
    }
    for (int32_t k = 0; k < M.nee; ++k) {
        int32_t ce = M.ee_c[k], oe = M.ee_o[k];
        int32_t a0 = M.cl_eidx[2 * (int64_t)ce], a1 = M.cl_eidx[2 * (int64_t)ce + 1];
        int32_t b0 = M.ob_eidx[2 * (int64_t)oe], b1 = M.ob_eidx[2 * (int64_t)oe + 1];
        V3 cpa, cpb; double s, t;
        closest_segments(ldf(M.co, a0), ldf(M.co, a1), ldd(M.joined, 2 * (int64_t)b0 + 1),
                         ldd(M.joined, 2 * (int64_t)b1 + 1), P.geom_tiny, P.geom_rel_eps, cpa, cpb, s, t);
        keep_ee[k] = len(cpa - cpb) < reach;
    }
}

// ================================================================ self
// A port of self_collide_2.py: candidates (collision_force's broad phase and
// filters) and the relaxation loop.  Tunables arrive in ScParams.

struct ScParams {
    double thickness, detect, relax, blend, swept_margin;
    double eps, geom_rel_eps, geom_tiny, clamp_thick, clamp_edge;
    int32_t iters, do_edges, swept, pbd_exact, sheet_side;
};

struct ScMesh {
    float* co;                  // (nv,3) in/out
    const float* start_co;      // (nv,3)
    const int32_t* tridex;      // (nt,3)
    const int32_t* edges;       // (ne,2) endpoints of the edges that belong to a triangle
    const int64_t* reach_keys;  // (nk) optional: sorted v*nv+u for u within K rings of v
    const int32_t* labels;      // (nv) optional: sew labels (verts with one label are one point)
    const double* char_len;     // (nv) optional
    const int32_t* adj_pad;     // (nv,max_deg) optional: 1-ring, padded
    const uint8_t* adj_mask;    // (nv,max_deg)
    int64_t nk;
    int32_t nv, nt, ne, max_deg;
};

API int32_t mc_collide_sizeof_sc_params() { return (int32_t)sizeof(ScParams); }
API int32_t mc_collide_sizeof_sc_mesh() { return (int32_t)sizeof(ScMesh); }

static inline bool within_rings(const ScMesh& M, int32_t v, int32_t u) {
    if (!M.reach_keys) return false;
    int64_t k = (int64_t)v * M.nv + u;
    return std::binary_search(M.reach_keys, M.reach_keys + M.nk, k);
}

static inline int32_t lab(const ScMesh& M, int32_t v) { return M.labels ? M.labels[v] : v; }

static inline double sc_max_move(const ScMesh& M, const ScParams& P, int32_t v) {
    double base = P.clamp_thick * P.thickness;
    if (M.char_len) return std::max(base, P.clamp_edge * M.char_len[v]);
    return std::max(base, P.clamp_edge * 10.0 * P.thickness);
}

static std::vector<int32_t> g_vt_v, g_vt_t, g_ee_a, g_ee_b;

// Candidate vertex-triangle and edge-edge pairs, filtered, in the Python's
// order: vt sorted by (vertex, triangle); ee sorted by the two edges' sorted
// endpoint keys, each pair oriented lower edge first.  Returns counts through
// the out arguments; fetch with sc_candidates_fetch.
API void sc_candidates(const ScParams* Pp, const ScMesh* Mp, int64_t* n_vt, int64_t* n_ee) {
    const ScParams& P = *Pp;
    const ScMesh& M = *Mp;
    const float margin = (float)P.detect;
    g_vt_v.clear(); g_vt_t.clear(); g_ee_a.clear(); g_ee_b.clear();

    // ---- vertex vs triangle: swept boxes over start and current
    std::vector<float> vmin(3 * (size_t)M.nv), vmax(3 * (size_t)M.nv);
    for (int32_t v = 0; v < M.nv; ++v)
        for (int k = 0; k < 3; ++k) {
            float a = M.start_co[3 * (int64_t)v + k], b = M.co[3 * (int64_t)v + k];
            vmin[3 * (size_t)v + k] = std::min(a, b);
            vmax[3 * (size_t)v + k] = std::max(a, b);
        }
    std::vector<float> tmin(3 * (size_t)M.nt), tmax(3 * (size_t)M.nt);
    for (int32_t t = 0; t < M.nt; ++t)
        for (int k = 0; k < 3; ++k) {
            float lo = std::numeric_limits<float>::max(), hi = -lo;
            for (int j = 0; j < 3; ++j) {
                int32_t v = M.tridex[3 * (int64_t)t + j];
                lo = std::min(lo, vmin[3 * (size_t)v + k]);
                hi = std::max(hi, vmax[3 * (size_t)v + k]);
            }
            tmin[3 * (size_t)t + k] = lo;
            tmax[3 * (size_t)t + k] = hi;
        }
    mc_box_pairs(vmin.data(), vmax.data(), M.nv, tmin.data(), tmax.data(), M.nt, margin);
    for (size_t i = 0; i < g_pa.size(); ++i) {
        int32_t v = g_pa[i], t = g_pb[i];
        const int32_t* tv = M.tridex + 3 * (int64_t)t;
        int32_t lv = lab(M, v);
        if (lv == lab(M, tv[0]) || lv == lab(M, tv[1]) || lv == lab(M, tv[2])) continue;
        if (within_rings(M, v, tv[0]) || within_rings(M, v, tv[1]) || within_rings(M, v, tv[2])) continue;
        g_vt_v.push_back(v);
        g_vt_t.push_back(t);
    }

    // ---- edge vs edge: every pair of triangle edges within detect
    std::vector<float> emin(3 * (size_t)M.ne), emax(3 * (size_t)M.ne);
    for (int32_t e = 0; e < M.ne; ++e) {
        int32_t a = M.edges[2 * (int64_t)e], b = M.edges[2 * (int64_t)e + 1];
        for (int k = 0; k < 3; ++k) {
            emin[3 * (size_t)e + k] = std::min(vmin[3 * (size_t)a + k], vmin[3 * (size_t)b + k]);
            emax[3 * (size_t)e + k] = std::max(vmax[3 * (size_t)a + k], vmax[3 * (size_t)b + k]);
        }
    }
    mc_box_pairs(emin.data(), emax.data(), M.ne, emin.data(), emax.data(), M.ne, margin);
    struct EE { int64_t lo, hi; int32_t i, j; };
    std::vector<EE> ee;
    for (size_t k = 0; k < g_pa.size(); ++k) {
        int32_t i = g_pa[k], j = g_pb[k];
        if (!(i < j)) continue;
        int32_t a0 = M.edges[2 * (int64_t)i], a1 = M.edges[2 * (int64_t)i + 1];
        int32_t b0 = M.edges[2 * (int64_t)j], b1 = M.edges[2 * (int64_t)j + 1];
        int32_t la0 = lab(M, a0), la1 = lab(M, a1), lb0 = lab(M, b0), lb1 = lab(M, b1);
        if (la0 == lb0 || la0 == lb1 || la1 == lb0 || la1 == lb1) continue;
        if (within_rings(M, a0, b0) || within_rings(M, a0, b1) ||
            within_rings(M, a1, b0) || within_rings(M, a1, b1)) continue;
        int64_t ka = (int64_t)std::min(a0, a1) * M.nv + std::max(a0, a1);
        int64_t kb = (int64_t)std::min(b0, b1) * M.nv + std::max(b0, b1);
        ee.push_back({std::min(ka, kb), std::max(ka, kb), i, j});
    }
    std::sort(ee.begin(), ee.end(), [](const EE& x, const EE& y) {
        return x.lo < y.lo || (x.lo == y.lo && (x.hi < y.hi || (x.hi == y.hi && (x.i < y.i || (x.i == y.i && x.j < y.j)))));
    });
    int64_t last_lo = -1, last_hi = -1;
    for (const EE& x : ee) {
        if (x.lo == last_lo && x.hi == last_hi) continue;           // np.unique: first kept
        last_lo = x.lo; last_hi = x.hi;
        g_ee_a.push_back(x.i);
        g_ee_b.push_back(x.j);
    }
    g_pa.clear(); g_pb.clear();
    *n_vt = (int64_t)g_vt_v.size();
    *n_ee = (int64_t)g_ee_a.size();
}

// vt_v/vt_t (n_vt); ee_a/ee_b are edge-list positions (n_ee).
API void sc_candidates_fetch(int32_t* vt_v, int32_t* vt_t, int32_t* ee_a, int32_t* ee_b) {
    if (!g_vt_v.empty()) {
        std::memcpy(vt_v, g_vt_v.data(), sizeof(int32_t) * g_vt_v.size());
        std::memcpy(vt_t, g_vt_t.data(), sizeof(int32_t) * g_vt_t.size());
    }
    if (!g_ee_a.empty()) {
        std::memcpy(ee_a, g_ee_a.data(), sizeof(int32_t) * g_ee_a.size());
        std::memcpy(ee_b, g_ee_b.data(), sizeof(int32_t) * g_ee_b.size());
    }
}

static inline V3 cross(V3 a, V3 b) {
    return V3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}
static inline double sgn(double x) { return (double)((x > 0) - (x < 0)); }

struct ScRows {
    std::vector<int32_t> v;
    std::vector<V3> m;
    std::vector<uint8_t> sw;
    void clear() { v.clear(); m.clear(); sw.clear(); }
};

// tri_point_response: vertex rows for every hit (candidate order), then the
// triangle corner rows.
static void sc_tri_point(const ScMesh& M, const ScParams& P, const int32_t* vt_v, const int32_t* vt_t,
                         int64_t n, ScRows& out) {
    const double th = P.thickness;
    struct Hit { int32_t v; int32_t t; V3 move; double w[3]; bool only_swept; };
    std::vector<Hit> hits;
    for (int64_t k = 0; k < n; ++k) {
        int32_t vi = vt_v[k], ti = vt_t[k];
        const int32_t* T = M.tridex + 3 * (int64_t)ti;
        V3 A1 = ldf(M.co, T[0]), B1 = ldf(M.co, T[1]), D1 = ldf(M.co, T[2]);
        V3 p1 = ldf(M.co, vi);
        V3 c1 = cross(B1 - A1, D1 - A1);
        double l1 = len(c1);
        V3 n1 = c1 * (1.0 / (l1 > P.eps ? l1 : 1.0));
        double d1 = dot(p1 - A1, n1);
        double w1[3];
        bool check1 = inside_tri(A1, B1, D1, p1, 0.0, P.geom_rel_eps, w1);
        V3 plot1 = bary(A1, B1, D1, w1);
        bool prox = check1 && (len(p1 - plot1) < th);
        bool swept = false;
        double d0 = 0.0, wT[3] = {0, 0, 0};
        if (P.swept) {
            V3 A0 = ldf(M.start_co, T[0]), B0 = ldf(M.start_co, T[1]), D0 = ldf(M.start_co, T[2]);
            V3 p0 = ldf(M.start_co, vi);
            V3 c0 = cross(B0 - A0, D0 - A0);
            double l0 = len(c0);
            V3 n0 = c0 * (1.0 / (l0 > P.eps ? l0 : 1.0));
            d0 = dot(p0 - A0, n0);
            double denom = d0 - d1;
            bool crossed = (sgn(d0) != sgn(d1)) && (std::fabs(denom) > P.eps);
            if (crossed) {
                double tau = clip01(d0 / denom);
                V3 pT = p0 + (p1 - p0) * tau;
                V3 AT = A0 + (A1 - A0) * tau, BT = B0 + (B1 - B0) * tau, DT = D0 + (D1 - D0) * tau;
                bool checkT = inside_tri(AT, BT, DT, pT, -P.swept_margin, P.geom_rel_eps, wT);
                swept = checkT;
            }
        }
        if (!(prox || swept)) continue;
        bool only_swept = swept && !prox;
        double w[3] = {w1[0], w1[1], w1[2]};
        if (only_swept) { w[0] = wT[0]; w[1] = wT[1]; w[2] = wT[2]; }
        V3 plot_end = bary(A1, B1, D1, w);
        // which side to push toward: where it came from, or where its own
        // 1-ring sits (SC_SHEET_SIDE) -- never sign(d1), see the Python
        double hist = std::fabs(d0) > P.eps ? sgn(d0) : sgn(d1);
        if (hist == 0.0) hist = 1.0;
        double prox_side = hist;
        if (P.sheet_side && M.adj_pad) {
            double s = 0.0;
            for (int32_t j = 0; j < M.max_deg; ++j) {
                if (!M.adj_mask[(int64_t)vi * M.max_deg + j]) continue;
                int32_t u = M.adj_pad[(int64_t)vi * M.max_deg + j];
                s += dot(ldf(M.co, u) - plot_end, n1);
            }
            if (std::fabs(s) > P.eps) prox_side = sgn(s);
        }
        double side = only_swept ? hist : prox_side;
        if (side == 0.0) side = 1.0;
        V3 target = plot_end + n1 * (side * th);
        V3 move = clamp_len(target - p1, sc_max_move(M, P, vi));
        Hit h;
        h.v = vi; h.t = ti; h.move = move; h.only_swept = only_swept;
        h.w[0] = w[0]; h.w[1] = w[1]; h.w[2] = w[2];
        hits.push_back(h);
    }
    for (const Hit& h : hits) {
        double bl = h.only_swept ? 1.0 : P.blend;
        out.v.push_back(h.v);
        out.m.push_back(h.move * bl);
        out.sw.push_back(h.only_swept);
    }
    for (const Hit& h : hits) {
        double bl = h.only_swept ? 1.0 : P.blend;
        double dd = h.w[0] * h.w[0] + h.w[1] * h.w[1] + h.w[2] * h.w[2] + P.eps;
        const int32_t* T = M.tridex + 3 * (int64_t)h.t;
        for (int j = 0; j < 3; ++j) {
            double coef = P.pbd_exact ? h.w[j] / dd : h.w[j];
            out.v.push_back(T[j]);
            out.m.push_back(h.move * (-(1.0 - bl) * coef));
            out.sw.push_back(h.only_swept);
        }
    }
}

// edge_edge_response: rows [all a0, all a1, all b0, all b1].
static void sc_edge_edge(const ScMesh& M, const ScParams& P, const int32_t* ee_a, const int32_t* ee_b,
                         int64_t n, ScRows& out) {
    const double th = P.thickness;
    struct Hit { int32_t a0, a1, b0, b1; double s, t; V3 half; bool crossed; };
    std::vector<Hit> hits;
    for (int64_t k = 0; k < n; ++k) {
        int32_t a0 = M.edges[2 * (int64_t)ee_a[k]], a1 = M.edges[2 * (int64_t)ee_a[k] + 1];
        int32_t b0 = M.edges[2 * (int64_t)ee_b[k]], b1 = M.edges[2 * (int64_t)ee_b[k] + 1];
        V3 cpa, cpb; double s, t;
        closest_segments(ldf(M.co, a0), ldf(M.co, a1), ldf(M.co, b0), ldf(M.co, b1),
                         P.geom_tiny, P.geom_rel_eps, cpa, cpb, s, t);
        V3 sep1 = cpa - cpb;
        double dist1 = len(sep1);
        bool interior = (s > 0.0) && (s < 1.0) && (t > 0.0) && (t < 1.0);
        bool prox = interior && (dist1 < th);
        bool hit = prox, crossed = false;
        V3 dir = dist1 > P.eps ? sep1 * (1.0 / dist1) : V3();
        if (P.swept) {
            V3 scpa, scpb; double ss, st;
            closest_segments(ldf(M.start_co, a0), ldf(M.start_co, a1), ldf(M.start_co, b0),
                             ldf(M.start_co, b1), P.geom_tiny, P.geom_rel_eps, scpa, scpb, ss, st);
            V3 sep0 = scpa - scpb;
            double d0n = len(sep0);
            crossed = (dot(sep0, sep1) < 0.0) && interior;
            hit = hit || crossed;
            if (d0n > P.eps) dir = sep0 * (1.0 / d0n);
        }
        if (!hit) continue;
        V3 corr = dir * th - sep1;
        hits.push_back({a0, a1, b0, b1, s, t, corr * 0.5, crossed});
    }
    const size_t nh = hits.size();
    std::vector<double> wa1(nh), wa2(nh), wb1(nh), wb2(nh);
    for (size_t i = 0; i < nh; ++i) {
        double s = hits[i].s, t = hits[i].t;
        if (P.pbd_exact) {
            double da = (1.0 - s) * (1.0 - s) + s * s + P.eps;
            double db = (1.0 - t) * (1.0 - t) + t * t + P.eps;
            wa1[i] = (1.0 - s) / da; wa2[i] = s / da; wb1[i] = (1.0 - t) / db; wb2[i] = t / db;
        } else {
            wa1[i] = 1.0 - s; wa2[i] = s; wb1[i] = 1.0 - t; wb2[i] = t;
        }
    }
    for (int part = 0; part < 4; ++part)
        for (size_t i = 0; i < nh; ++i) {
            const Hit& h = hits[i];
            int32_t v = part == 0 ? h.a0 : part == 1 ? h.a1 : part == 2 ? h.b0 : h.b1;
            V3 m = part == 0 ? h.half * wa1[i] : part == 1 ? h.half * wa2[i]
                 : part == 2 ? h.half * (-wb1[i]) : h.half * (-wb2[i]);
            out.v.push_back(v);
            out.m.push_back(clamp_len(m, sc_max_move(M, P, v)));
            out.sw.push_back(h.crossed);
        }
}

// The relaxation loop of self_collide_2.collision_force.  Moves M.co;
// any_hit / vert_swept (nv) and total_step (nv,3) are written fresh.
API void sc_resolve(const ScParams* Pp, const ScMesh* Mp,
                    const int32_t* vt_v, const int32_t* vt_t, int64_t n_vt,
                    const int32_t* ee_a, const int32_t* ee_b, int64_t n_ee,
                    uint8_t* any_hit, uint8_t* vert_swept, double* total_step) {
    const ScParams& P = *Pp;
    const ScMesh& M = *Mp;
    const int32_t nv = M.nv;
    std::memset(any_hit, 0, (size_t)nv);
    std::memset(vert_swept, 0, (size_t)nv);
    std::memset(total_step, 0, sizeof(double) * 3 * (size_t)nv);
    std::vector<V3> acc(nv);
    std::vector<double> cnt(nv);
    ScRows R;
    for (int32_t it = 0; it < std::max(1, P.iters); ++it) {
        std::fill(acc.begin(), acc.end(), V3());
        std::fill(cnt.begin(), cnt.end(), 0.0);
        R.clear();
        if (n_vt) sc_tri_point(M, P, vt_v, vt_t, n_vt, R);
        if (P.do_edges && n_ee) sc_edge_edge(M, P, ee_a, ee_b, n_ee, R);
        for (size_t r = 0; r < R.v.size(); ++r) {
            V3 m = R.m[r];
            if (!finite3(m)) m = V3();
            acc[R.v[r]] = acc[R.v[r]] + m;
            cnt[R.v[r]] += 1.0;
            if (R.sw[r]) vert_swept[R.v[r]] = 1;
        }
        bool any = false;
        for (int32_t v = 0; v < nv; ++v) if (cnt[v] > 0.0) { any = true; break; }
        if (!any) break;
        for (int32_t v = 0; v < nv; ++v) {
            if (cnt[v] <= 0.0) continue;
            V3 st = acc[v] * ((1.0 / cnt[v]) * P.relax);
            st = clamp_len(st, sc_max_move(M, P, v));
            if (!finite3(st)) st = V3();
            M.co[3 * (int64_t)v] = M.co[3 * (int64_t)v] + (float)st.x;
            M.co[3 * (int64_t)v + 1] = M.co[3 * (int64_t)v + 1] + (float)st.y;
            M.co[3 * (int64_t)v + 2] = M.co[3 * (int64_t)v + 2] + (float)st.z;
            total_step[3 * (int64_t)v] += st.x;
            total_step[3 * (int64_t)v + 1] += st.y;
            total_step[3 * (int64_t)v + 2] += st.z;
            any_hit[v] = 1;
        }
    }
}
// ---------------------------------------------------------------- resolve
// _narrow_resolve.  Moves M.co in place; any_hit (nv) is OR-ed, total_step
// (nv,3) accumulated.  Returns the number of friction corrections applied.
API int32_t oc_narrow_resolve(const OcParams* Pp, const OcMesh* Mp,
                              uint8_t* any_hit, double* total_step) {
    const OcParams& P = *Pp;
    const OcMesh& M = *Mp;
    const int32_t nv = M.nv;
    std::memset(any_hit, 0, (size_t)nv);
    std::memset(total_step, 0, sizeof(double) * 3 * (size_t)nv);

    Rows R;
    std::vector<V3> x(nv);
    std::vector<uint8_t> hit(nv);
    bool any_any = false;
    const int32_t iters = std::max(1, P.iters);
    for (int32_t it = 0; it < iters; ++it) {
        R.clear();
        if (P.do_pt && M.npt) point_to_triangle(M, P, total_step, &R, nullptr);
        if (P.do_tp && M.ntp) triangle_to_point(M, P, total_step, &R, nullptr);
        if (P.do_ee && M.nee) edge_to_edge(M, P, total_step, &R, nullptr);
        project_contacts(nv, R, P, x, hit);
        bool any = false;
        for (int32_t v = 0; v < nv; ++v) if (hit[v]) { any = true; break; }
        if (!any) break;
        for (int32_t v = 0; v < nv; ++v) {
            if (!hit[v]) continue;
            V3 st = x[v] * P.relax;
            if (!finite3(st)) st = V3();
            st = clamp_len(st, max_move(M, P, v));
            M.co[3 * (int64_t)v] = M.co[3 * (int64_t)v] + (float)st.x;
            M.co[3 * (int64_t)v + 1] = M.co[3 * (int64_t)v + 1] + (float)st.y;
            M.co[3 * (int64_t)v + 2] = M.co[3 * (int64_t)v + 2] + (float)st.z;
            total_step[3 * (int64_t)v] += st.x;
            total_step[3 * (int64_t)v + 1] += st.y;
            total_step[3 * (int64_t)v + 2] += st.z;
            any_hit[v] = 1;
        }
        any_any = true;
    }

    // ---- friction: one pass on the now-settled positions ----
    int32_t n_fric = 0;
    if (P.do_friction && P.mu > 0.0 && P.friction_max > 0.0 && any_any) {
        bool anyh = false;
        for (int32_t v = 0; v < nv; ++v) if (any_hit[v]) { anyh = true; break; }
        if (anyh) {
            FricRows F;
            if (P.do_pt && M.npt) point_to_triangle(M, P, total_step, nullptr, &F);
            if (P.do_tp && M.ntp) triangle_to_point(M, P, total_step, nullptr, &F);
            if (P.do_ee && M.nee) edge_to_edge(M, P, total_step, nullptr, &F);
            n_fric = (int32_t)F.v.size();
            std::vector<V3> acc(nv);
            std::vector<double> cnt(nv, 0.0);
            for (size_t r = 0; r < F.v.size(); ++r) {
                V3 m = F.m[r];
                if (!finite3(m)) m = V3();
                acc[F.v[r]] = acc[F.v[r]] + m;
                cnt[F.v[r]] += 1.0;
            }
            const double cap = P.friction_maxmove * P.thickness;
            for (int32_t v = 0; v < nv; ++v) {
                if (cnt[v] <= 0.0) continue;
                V3 fs = acc[v] * (1.0 / cnt[v]);
                if (!finite3(fs)) fs = V3();
                fs = clamp_len(fs, cap);
                M.co[3 * (int64_t)v] = M.co[3 * (int64_t)v] + (float)fs.x;
                M.co[3 * (int64_t)v + 1] = M.co[3 * (int64_t)v + 1] + (float)fs.y;
                M.co[3 * (int64_t)v + 2] = M.co[3 * (int64_t)v + 2] + (float)fs.z;
            }
        }
    }
    return n_fric;
}
