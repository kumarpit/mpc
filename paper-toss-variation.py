import numpy as np
import cvxpy as cp
from visualization.viz import visualize

################################################################
# Problem Parameters
################################################################

dt = 0.15
N  = 21           # total steps (3.0 s)
H  = 10            # fixed MPC prediction horizon
g  = np.array([0.0, 0.0, -9.8])

r0     = np.array([0.0,  0.0,  2.0])
v0     = np.array([3.0,  0.0,  4.5])
r_goal = np.array([7.0, 10.0,  1.5])

ALPHA_MAX = 80.0

################################################################
# Level Design
# TODO: make a simple DSL for this
################################################################
class Fan:
    def __init__(self, bounds, direction, color, name):
        self.bounds    = np.array(bounds, dtype=float)
        d              = np.array(direction, dtype=float)
        self.direction = d / np.linalg.norm(d)
        self.color     = color
        self.name      = name

    def contains(self, r):
        return all(self.bounds[i, 0] <= r[i] <= self.bounds[i, 1] for i in range(3))

ZL, ZH = -5.0, 15.0

fans = [
    Fan([[-0.5, 2.5], [-0.5, 3.5], [ZL, ZH]], [ 0.0,  1.0,  1.0], '#2E86AB', 'Fan 0 (+y+z)'),
    Fan([[ 2.5, 5.0], [-0.5, 3.5], [ZL, ZH]], [ 0.0,  1.0,  0.8], '#E8834C', 'Fan 1 (+y+z)'),
    Fan([[ 5.0, 8.5], [-0.5, 3.5], [ZL, ZH]], [ 0.0,  1.0,  0.4], '#F5C542', 'Fan 2 (+y+z)'),
    Fan([[-0.5, 2.5], [ 3.5, 7.0], [ZL, ZH]], [ 0.0,  1.0,  0.4], '#3EC47A', 'Fan 3 (+y+z)'),
    Fan([[ 2.5, 5.0], [ 3.5, 7.0], [ZL, ZH]], [ 0.0,  1.0,  0.0], '#9B59B6', 'Fan 4 (+y)'),
    Fan([[ 5.0, 8.5], [ 3.5, 7.0], [ZL, ZH]], [ 0.0,  1.0, -0.5], '#E84C7B', 'Fan 5 (+y−z)'),
    Fan([[-0.5, 2.5], [ 7.0,11.5], [ZL, ZH]], [ 1.0,  0.5, -0.3], '#1ABC9C', 'Fan 6 (+x+y−z)'),
    Fan([[ 2.5, 5.0], [ 7.0,11.5], [ZL, ZH]], [ 1.0,  0.2, -1.0], '#E67E22', 'Fan 7 (+x−z)'),
    Fan([[ 5.0, 8.5], [ 7.0,11.5], [ZL, ZH]], [ 0.0, -0.3, -1.0], '#C0392B', 'Fan 8 (−y−z)'),
]
N_FANS = len(fans)

################################################################
# Utils 
################################################################

# returns the list of positions for `steps`time steps, representing the path taken via free 
# flight given the initial position r0 and initial velocity v0
def simulate_free_flight(r0, v0, steps=N):
    r = np.zeros((steps, 3))
    v = np.zeros((steps, 3))
    r[0], v[0] = r0, v0
    for k in range(steps - 1):
        r[k+1] = r[k] + v[k] * dt
        v[k+1] = v[k] + g * dt
    return r

r_free = simulate_free_flight(r0, v0)

################################################################
# Main MPC Loop 
################################################################

# to kick things, we start by guessing our trajectory to be the straight line between the start 
# position and the end goal position
r_pred = np.linspace(r0, r_goal, N)

r_actual   = [r0]
v_actual   = [v0]
alpha_log  = []
active_log = []
pred_trajs = []

print(f"MPC  fixed horizon H={H}  (N={N}, dt={dt}s, T={(N-1)*dt:.1f}s)")
print(f"  {'k':>3}  {'horizon':>7}  {'planned err':>13}  {'actual err':>12}  {'status'}")
print("  " + "─" * 56)

for k in range(N - 1):
    r_k = r_actual[k]
    v_k = v_actual[k]

    horizon = min(H, N - k)

    # determine `active` fans (i.e. those that have an influence over the next `horizon` timesteps)
    active = np.zeros((horizon - 1, N_FANS))
    for j in range(horizon - 1):
        r_check = r_k if j == 0 else r_pred[k + j]
        for i, fan in enumerate(fans):
            if fan.contains(r_check):
                active[j, i] = 1.0

    # now that we know which fans are active for each time step in this horizon, we can setup our problem 
    # for the current iteration
    r_var     = cp.Variable((horizon, 3))
    v_var     = cp.Variable((horizon, 3))
    alpha_var = cp.Variable((horizon - 1, N_FANS), nonneg=True)

    # constraints!
    cons = [r_var[0] == r_k, v_var[0] == v_k]
    for j in range(horizon - 1):
        wind_j = sum(active[j, i] * alpha_var[j, i] * fans[i].direction
                     for i in range(N_FANS))
        acc_j = g + wind_j
        cons += [
            r_var[j + 1] == r_var[j] + v_var[j] * dt + 0.5 * acc_j * dt ** 2,
            v_var[j + 1] == v_var[j] + acc_j * dt,
            alpha_var[j] <= ALPHA_MAX,
        ]

    # objective; minimize the distance between the ball and the end goal at the end of the current trajectory
    obj = cp.Minimize(cp.sum_squares(r_var[horizon - 1] - r_goal))
    prob = cp.Problem(obj, cons)
    prob.solve(solver=cp.CLARABEL, verbose=False)

    status = prob.status
    if status not in ("optimal", "optimal_inaccurate"):
        print(f"  {k:>3}  {horizon:>7}  {'—':>13}  {'—':>12}  {status}  ← FAILED")
        break

    # Update r_pred for the next H steps
    r_pred[k:k + horizon] = r_var.value

    # Extract first action
    alpha_k  = alpha_var.value[0]
    active_k = active[0]

    alpha_log.append(alpha_k)
    active_log.append(active_k)
    pred_trajs.append(r_var.value)

    # Simulate one true step
    wind_true = sum(active_k[i] * alpha_k[i] * fans[i].direction
                    for i in range(N_FANS))
    acc_true = g + wind_true
    r_next = r_k + v_k * dt + 0.5 * acc_true * dt ** 2
    v_next = v_k + acc_true * dt

    r_actual.append(r_next)
    v_actual.append(v_next)

    planned_err = float(np.linalg.norm(r_var.value[-1] - r_goal))
    actual_err  = float(np.linalg.norm(r_next - r_goal))
    print(f"  {k:>3}  {horizon:>7}  {planned_err:>13.4f}  {actual_err:>12.4f}  {status}")

r_actual   = np.array(r_actual)
alpha_log  = np.array(alpha_log)
active_log = np.array(active_log)
final_err = float(np.linalg.norm(r_actual[-1] - r_goal))
print(f"\nActual terminal error: {final_err:.4f} m")

################################################################
# Visualization!
################################################################
visualize(fans, r_free, r_actual, alpha_log, active_log, pred_trajs,
          final_err, H, N, dt, ALPHA_MAX, r0, r_goal)
