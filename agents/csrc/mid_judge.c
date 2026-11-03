/* Middle-game judge (part 3 c): exact Dots and Boxes search on the real board.
 *
 * The same algorithm as agents/strong.py Engine.search (and the Python
 * reference sim/mid_judge.py PyJudge), in C, with:
 *   - a transposition table of value *bounds* (lower, upper) keyed by the
 *     canonical position: the smallest Zobrist hash over the board's
 *     symmetries (4 on a rectangle, 8 on a square);
 *   - the loony endgame without junctions solved by Berlekamp's recursion
 *     (sim/chain_solver.py at s = 0, exact) instead of search;
 *   - null-window searches from Python decide the sign of the result
 *     (win / draw / loss) rather than the score.
 * Value = net boxes still to be won by the side to move.
 *
 * Captures: a capturable box is taken at once unless it is a decision point
 * (the first of the last two boxes of a chain ending at the ground or a
 * junction, or the first of four boxes of an opened loop): there also try
 * the double-dealing line. Moves without captures: every safe line plus one
 * representative per chain / loop (Engine.moves).
 */
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define MAXL 400
#define MAXB 200
#define MAXSYM 8
#define LO0 (-127)
#define HI0 127

typedef struct {
    uint64_t key;
    int16_t lo, hi;
    int16_t move;   /* in the canonical frame, -1 none */
    int16_t pad;
} tt_t;

typedef struct {
    uint64_t key;
    int16_t val;
    int16_t used;
} dp_t;

typedef struct {
    int nl, nb, nsym;
    int16_t box_lines[MAXB][4];
    int16_t ends[MAXL][2];          /* boxes on both sides, -1 = ground */
    int16_t perm[MAXSYM][MAXL];     /* line -> image under symmetry s */
    int16_t inv[MAXSYM][MAXL];
    uint64_t zob[MAXL];
    uint64_t zc[2][MAXB + 1][MAXB + 1];   /* simple_value keys: kind, length, count */
    uint8_t lines[MAXL];
    uint8_t sides[MAXB];
    uint64_t h[MAXSYM];
    tt_t *tt;
    uint64_t tt_mask;
    dp_t *dp;
    uint64_t dp_mask;
    int64_t nodes, node_limit;
    double deadline;
    int timeout;
    int ndrawn;
} judge_t;

static uint64_t rng_state = 0x9E3779B97F4A7C15ULL;
static uint64_t rnd(void) {
    uint64_t z = (rng_state += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

static double now(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + 1e-9 * t.tv_nsec;
}

judge_t *judge_new(int nl, int nb, const int16_t *box_lines, const int16_t *ends,
                   int nsym, const int16_t *perm, int tt_bits) {
    if (nl > MAXL || nb > MAXB || nsym > MAXSYM) return NULL;
    judge_t *j = calloc(1, sizeof(judge_t));
    j->nl = nl; j->nb = nb; j->nsym = nsym;
    memcpy(j->box_lines, box_lines, sizeof(int16_t) * 4 * nb);
    memcpy(j->ends, ends, sizeof(int16_t) * 2 * nl);
    for (int s = 0; s < nsym; s++)
        for (int l = 0; l < nl; l++) {
            j->perm[s][l] = perm[s * nl + l];
            j->inv[s][perm[s * nl + l]] = l;
        }
    rng_state = 0x1234567ULL;
    for (int l = 0; l < nl; l++) j->zob[l] = rnd();
    for (int k = 0; k < 2; k++)
        for (int a = 0; a <= nb; a++)
            for (int c = 0; c <= nb; c++) j->zc[k][a][c] = rnd();
    j->tt = calloc((size_t)1 << tt_bits, sizeof(tt_t));
    j->tt_mask = ((uint64_t)1 << tt_bits) - 1;
    j->dp = calloc((size_t)1 << 20, sizeof(dp_t));
    j->dp_mask = ((uint64_t)1 << 20) - 1;
    return j;
}

void judge_free(judge_t *j) {
    free(j->tt);
    free(j->dp);
    free(j);
}

void judge_clear(judge_t *j) {
    memset(j->tt, 0, sizeof(tt_t) * (j->tt_mask + 1));
}

/* --- primitives -------------------------------------------------------------- */

static inline int draw(judge_t *j, int ln) {
    int got = 0;
    j->lines[ln] = 1;
    j->ndrawn++;
    for (int s = 0; s < j->nsym; s++) j->h[s] ^= j->zob[j->perm[s][ln]];
    for (int k = 0; k < 2; k++) {
        int b = j->ends[ln][k];
        if (b >= 0 && ++j->sides[b] == 4) got++;
    }
    return got;
}

static inline void undraw(judge_t *j, int ln) {
    j->lines[ln] = 0;
    j->ndrawn--;
    for (int s = 0; s < j->nsym; s++) j->h[s] ^= j->zob[j->perm[s][ln]];
    for (int k = 0; k < 2; k++) {
        int b = j->ends[ln][k];
        if (b >= 0) j->sides[b]--;
    }
}

static inline int undrawn(judge_t *j, int b, int skip) {
    for (int k = 0; k < 4; k++) {
        int ln = j->box_lines[b][k];
        if (!j->lines[ln] && ln != skip) return ln;
    }
    return -1;
}

static inline int other(judge_t *j, int ln, int b) {
    return j->ends[ln][0] == b ? j->ends[ln][1] : j->ends[ln][0];
}

/* Engine._walk: double-dealing line of a decision point, or -1. */
static int walk_dd(judge_t *j, int c, int ln) {
    int n = 1, prev = ln, cur = other(j, ln, c), open = 0;
    int l1 = -1;
    while (1) {
        if (cur < 0) break;
        int d = 4 - j->sides[cur];
        if (d == 1) { n++; open = 1; break; }
        if (d >= 3) break;
        n++;
        int nxt = undrawn(j, cur, prev);
        if (l1 < 0) l1 = nxt;
        prev = nxt;
        cur = other(j, nxt, cur);
    }
    if (!open && n == 2) return l1;
    if (open && n == 4) return l1;
    return -1;
}

/* Apply captures that never lose; push the lines on stk. Returns boxes won;
 * fills dec (pairs take, dd) with the decision points left. */
static int forced(judge_t *j, int16_t *stk, int *ns, int16_t *dec, int *nd) {
    int gained = 0;
    while (1) {
        int any = 0;
        *nd = 0;
        for (int c = 0; c < j->nb; c++) {
            if (j->sides[c] != 3) continue;
            int ln = undrawn(j, c, -1);
            int dd = walk_dd(j, c, ln);
            if (dd < 0) {
                gained += draw(j, ln);
                stk[(*ns)++] = ln;
                any = 1;
                break;
            }
            dec[2 * *nd] = ln;
            dec[2 * *nd + 1] = dd;
            (*nd)++;
        }
        if (!any) return gained;
    }
}

/* Engine.moves: safe lines and chain representatives. */
static int gen_moves(judge_t *j, int16_t *mv, int *nsafe) {
    int n = 0;
    static __thread uint8_t seen[MAXB];
    for (int ln = 0; ln < j->nl; ln++) {
        if (j->lines[ln]) continue;
        int e0 = j->ends[ln][0], e1 = j->ends[ln][1];
        if ((e0 < 0 || j->sides[e0] < 2) && (e1 < 0 || j->sides[e1] < 2)) mv[n++] = ln;
    }
    *nsafe = n;
    memset(seen, 0, j->nb);
    for (int b = 0; b < j->nb; b++) {
        if (j->sides[b] != 2 || seen[b]) continue;
        int l1 = undrawn(j, b, -1), l2 = undrawn(j, b, l1);
        int len = 1, loop = 0, endl[2] = {-1, -1}, arm0len = 0, arm1first = l2;
        seen[b] = 1;
        for (int k = 0; k < 2 && !loop; k++) {
            int start = k ? l2 : l1, prev = start, cur = other(j, start, b), alen = 1;
            while (cur >= 0 && j->sides[cur] == 2) {
                if (cur == b) { loop = 1; break; }
                seen[cur] = 1;
                len++;
                int nxt = undrawn(j, cur, prev);
                alen++;
                prev = nxt;
                cur = other(j, nxt, cur);
            }
            if (k == 0) arm0len = alen;
            if (!loop) endl[k] = prev;
        }
        int rep;
        if (loop || len == 1) rep = l1;
        else if (len == 2) rep = arm0len == 2 ? l1 : arm1first;
        else rep = endl[0];
        mv[n++] = rep;
    }
    return n;
}

/* --- loony endgame without junctions ----------------------------------------- */

static int cc[2][MAXB + 1];   /* counts of chains / loops by length (one thread per process) */

static int simple_value(judge_t *j, uint64_t key, int total) {
    if (!total) return 0;
    uint64_t i = key & j->dp_mask;
    while (j->dp[i].used) {
        if (j->dp[i].key == key) return j->dp[i].val;
        i = (i + 1) & j->dp_mask;
    }
    int best = -1000;
    for (int k = 0; k < 2; k++)
        for (int n = 1; n <= j->nb; n++) {
            if (!cc[k][n]) continue;
            uint64_t k2 = key ^ j->zc[k][n][cc[k][n]] ^ j->zc[k][n][cc[k][n] - 1];
            cc[k][n]--;
            int r = simple_value(j, k2, total - 1);
            cc[k][n]++;
            int v;
            if (k == 0) {
                if (n <= 2) v = -(n + r);
                else { int a = n + r, b2 = n - 4 - r; v = -(a > b2 ? a : b2); }
            } else { int a = n + r, b2 = n - 8 - r; v = -(a > b2 ? a : b2); }
            if (v > best) best = v;
        }
    i = key & j->dp_mask;
    while (j->dp[i].used) i = (i + 1) & j->dp_mask;
    j->dp[i].key = key;
    j->dp[i].val = best;
    j->dp[i].used = 1;
    return best;
}

/* Value of a junction-free loony position, or -1000 if it has a junction. */
static int components(judge_t *j) {
    static __thread uint8_t seen[MAXB];
    memset(seen, 0, j->nb);
    memset(cc, 0, sizeof(cc));
    uint64_t key = 0;
    int total = 0;
    for (int b = 0; b < j->nb; b++) {
        if (j->sides[b] == 4 || seen[b]) continue;
        if (j->sides[b] != 2) return -1000;
        int l1 = undrawn(j, b, -1), l2 = undrawn(j, b, l1);
        int n = 1, loop = 0;
        seen[b] = 1;
        for (int k = 0; k < 2 && !loop; k++) {
            int start = k ? l2 : l1, prev = start, cur = other(j, start, b);
            while (cur >= 0) {
                if (cur == b) { loop = 1; break; }
                if (j->sides[cur] != 2) return -1000;
                seen[cur] = 1;
                n++;
                int nxt = undrawn(j, cur, prev);
                prev = nxt;
                cur = other(j, nxt, cur);
            }
        }
        int k = loop;
        key ^= j->zc[k][n][cc[k][n]] ^ j->zc[k][n][cc[k][n] + 1];
        cc[k][n]++;
        total++;
    }
    return simple_value(j, key, total);
}

/* --- search ------------------------------------------------------------------- */

static int search(judge_t *j, int alpha, int beta);

static inline void canon(judge_t *j, uint64_t *key, int *sym) {
    uint64_t k = j->h[0];
    int s0 = 0;
    for (int s = 1; s < j->nsym; s++)
        if (j->h[s] < k) { k = j->h[s]; s0 = s; }
    *key = k;
    *sym = s0;
}

static void tt_store(judge_t *j, uint64_t key, int lo, int hi, int move) {
    tt_t *e = &j->tt[key & j->tt_mask];
    if (e->key == key) {
        if (lo > e->lo) e->lo = lo;
        if (hi < e->hi) e->hi = hi;
        if (move >= 0) e->move = move;
        return;
    }
    e->key = key;
    e->lo = lo;
    e->hi = hi;
    e->move = move;
}

static int node(judge_t *j, int alpha, int beta, const int16_t *dec, int nd) {
    uint64_t key;
    int sym;
    canon(j, &key, &sym);
    tt_t *e = &j->tt[key & j->tt_mask];
    int tt_move = -1;
    if (e->key == key) {
        if (e->lo >= beta) return e->lo;
        if (e->hi <= alpha) return e->hi;
        if (e->lo == e->hi) return e->lo;
        if (e->lo > alpha) alpha = e->lo;
        if (e->hi < beta) beta = e->hi;
        if (e->move >= 0) tt_move = j->inv[sym][e->move];
    }
    int16_t mv[MAXL + 2 * MAXB];
    int n = 0, nsafe = 0;
    if (nd) {
        for (int i = 0; i < nd; i++) {
            mv[n++] = dec[2 * i];
            mv[n++] = dec[2 * i + 1];
        }
    } else {
        int16_t tmp[MAXL + MAXB];
        int m = gen_moves(j, tmp, &nsafe);
        if (!nsafe) {
            int v = components(j);
            if (v > -1000) {
                tt_store(j, key, v, v, -1);
                return v;
            }
        }
        /* order: chain representatives (sacrifices) first, then safe lines */
        for (int i = nsafe; i < m; i++) mv[n++] = tmp[i];
        for (int i = 0; i < nsafe; i++) mv[n++] = tmp[i];
    }
    if (tt_move >= 0)
        for (int i = 0; i < n; i++)
            if (mv[i] == tt_move) {
                int16_t t = mv[i];
                memmove(mv + 1, mv, sizeof(int16_t) * i);
                mv[0] = t;
                break;
            }
    int a0 = alpha, best = -1000, best_move = -1;
    for (int i = 0; i < n && !j->timeout; i++) {
        int ln = mv[i], v;
        int got = draw(j, ln);
        if (got) v = got + search(j, alpha - got, beta - got);
        else v = -search(j, -beta, -alpha);
        undraw(j, ln);
        if (v > best) {
            best = v;
            best_move = ln;
            if (v > alpha) {
                alpha = v;
                if (alpha >= beta) break;
            }
        }
    }
    if (j->timeout) return 0;
    int lo = LO0, hi = HI0;
    if (best <= a0) hi = best;
    else if (best >= beta) lo = best;
    else lo = hi = best;
    tt_store(j, key, lo, hi, best_move >= 0 ? j->perm[sym][best_move] : -1);
    return best;
}

static int search(judge_t *j, int alpha, int beta) {
    j->nodes++;
    if (j->node_limit > 0 && j->nodes > j->node_limit) j->timeout = 1;
    if (!(j->nodes & 4095) && j->deadline > 0 && now() > j->deadline) j->timeout = 1;
    if (j->timeout) return 0;
    int16_t stk[MAXL], dec[4 * MAXB];
    int ns = 0, nd = 0;
    int gained = forced(j, stk, &ns, dec, &nd);
    int v;
    if (j->ndrawn == j->nl) v = gained;
    else v = gained + node(j, alpha - gained, beta - gained, dec, nd);
    while (ns) undraw(j, stk[--ns]);
    return v;
}

/* Value for the side to move at ``lines`` within (alpha, beta), fail-soft.
 * *status: 0 solved, 1 out of time / nodes (then the value means nothing). */
int judge_search(judge_t *j, const uint8_t *lines, int alpha, int beta, double seconds,
                 int64_t node_limit, int *status, int64_t *nodes) {
    memset(j->lines, 0, j->nl);
    memset(j->sides, 0, j->nb);
    memset(j->h, 0, sizeof(j->h));
    j->ndrawn = 0;
    for (int l = 0; l < j->nl; l++)
        if (lines[l]) draw(j, l);
    j->nodes = 0;
    j->node_limit = node_limit;
    j->deadline = seconds > 0 ? now() + seconds : 0;
    j->timeout = 0;
    int v = search(j, alpha, beta);
    *status = j->timeout;
    *nodes = j->nodes;
    return v;
}
