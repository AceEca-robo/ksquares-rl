/* Tree walk of agents/az_mcts.py in C: select, descend with virtual loss,
 * undo, backup. Boards, expansion and the network stay in Python; the C side
 * sees every node through a node_t that points into the node's numpy arrays.
 * The arithmetic follows the numpy expressions of the Python version
 * operation by operation (built with -ffp-contract=off), so both pick the
 * same moves. */
#include <math.h>
#include <stdint.h>

typedef struct {
    double *P, *N, *W;   /* per legal move, owned by the Python Node */
    int32_t *kid;        /* child node id per legal move, -1 = none */
    int32_t nm;          /* legal moves; 0 = not expanded */
    int32_t n;           /* visits through this node (incl. virtual) */
    int32_t player;
    int32_t terminal;
    double value;
} node_t;

static int select_child(const node_t *x, double c_puct, double fpu_red)
{
    double sq = x->n > 0 ? sqrt((double)x->n) : 1.0;
    double fpu = x->value - fpu_red;
    double cs = c_puct * sq;
    int best = 0;
    double bu = 0;
    for (int i = 0; i < x->nm; i++) {
        double Ni = x->N[i];
        double q = Ni > 0 ? x->W[i] / (Ni > 1 ? Ni : 1) : fpu;
        double u = q + cs * x->P[i] / (1.0 + Ni);
        if (i == 0 || u > bu) {
            bu = u;
            best = i;
        }
    }
    return best;
}

void mcts_undo(node_t *nodes, const int32_t *pn, const int32_t *pa, int len, double vl)
{
    for (int i = 0; i < len; i++) {
        node_t *x = &nodes[pn[i]];
        x->n -= 1;
        x->N[pa[i]] -= vl;
        x->W[pa[i]] += vl;
    }
}

/* Walk from root to a leaf adding virtual loss; the path goes to pn/pa,
 * its length to *len. Returns the leaf id (expanded-terminal or unexpanded
 * fresh), -1 on a collision (the path is already undone), or -2 when the
 * last edge (pn[len-1], pa[len-1]) has no child yet: the caller creates it. */
int mcts_descend(node_t *nodes, int root, double c_puct, double fpu_red, double vl,
                 int32_t *pn, int32_t *pa, int *len)
{
    int k = 0, id = root;
    node_t *x = &nodes[id];
    while (x->nm > 0 && !x->terminal) {
        int a = select_child(x, c_puct, fpu_red);
        pn[k] = id;
        pa[k] = a;
        k++;
        x->n += 1;
        x->N[a] += vl;
        x->W[a] -= vl;
        int c = x->kid[a];
        if (c < 0) {
            *len = k;
            return -2;
        }
        id = c;
        x = &nodes[id];
    }
    *len = k;
    if (id != root && !x->terminal) {
        mcts_undo(nodes, pn, pa, k, vl);
        return -1;
    }
    return id;
}

/* leaf.n += 1, then back up the leaf's value (for its mover) along the path. */
void mcts_backup(node_t *nodes, const int32_t *pn, const int32_t *pa, int len, int leaf, double vl)
{
    double v = nodes[leaf].value;
    nodes[leaf].n += 1;
    for (int i = len - 1; i >= 0; i--) {
        node_t *x = &nodes[pn[i]];
        int a = pa[i];
        if (nodes[x->kid[a]].player != x->player)
            v = -v;
        x->N[a] += 1 - vl;
        x->W[a] += v + vl;
    }
}
