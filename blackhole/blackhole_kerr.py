import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve, brentq
import matplotlib.pyplot as plt
M = 1.0
a = 0.9
r_horizon = M + np.sqrt(M**2 - a**2)
r_ergo_equatorial = 2 * M
r_stop_horizon = r_horizon * 1.05
def R_photon(r, b):
    return r**4 + (a**2 - b**2) * r**2 + 2 * M * (b - a)**2 * r
def dR_photon_dr(r, b):
    return 4 * r**3 + 2 * (a**2 - b**2) * r + 2 * M * (b - a)**2
def phi_rate(r, b):
    Delta = r**2 - 2 * M * r + a**2
    return (-(a - b) + (a / Delta) * (r**2 + a**2 - a * b)) / r**2
def photon_sphere_eqs(vars):
    r, b = vars
    return [R_photon(r, b), dR_photon_dr(r, b)]
r_ph_pro, b_ph_pro = fsolve(photon_sphere_eqs, [2.0, 3.0])
r_ph_retro, b_ph_retro = fsolve(photon_sphere_eqs, [4.0, -6.0])
def Delta(r):
    return r**2 - 2 * M * r + a**2
def R_timelike(r, E, L):
    A = E * (r**2 + a**2) - a * L
    K = (L - a * E)**2
    return A**2 - Delta(r) * (r**2 + K)
def dR_timelike_dr_fd(r, E, L, h=1e-5):
    return (R_timelike(r + h, E, L) - R_timelike(r - h, E, L)) / (2 * h)
def d2R_timelike_dr2_fd(r, E, L, h=1e-4):
    return (R_timelike(r + h, E, L) - 2 * R_timelike(r, E, L) + R_timelike(r - h, E, L)) / h**2
def isco_eqs(vars):
    r, E, L = vars
    return [R_timelike(r, E, L), dR_timelike_dr_fd(r, E, L), d2R_timelike_dr2_fd(r, E, L)]
r_isco_pro, E_isco_pro, L_isco_pro = fsolve(isco_eqs, [6.0, 0.94, 3.6])
r_isco_retro, E_isco_retro, L_isco_retro = fsolve(isco_eqs, [6.0, 0.94, -3.6])
def integrate_photon(b, r0=15.0, lam_max=200.0):
    rs = np.linspace(r_stop_horizon, r0, 4000)
    Rs = R_photon(rs, b)
    sign_changes = np.where(np.diff(np.sign(Rs)) != 0)[0]
    r_min_true = None
    if len(sign_changes) > 0:
        idx = sign_changes[-1]
        r_min_true = brentq(lambda r: R_photon(r, b), rs[idx], rs[idx + 1])
    def rhs(lam, y, sign):
        r, phi = y
        Rval = max(R_photon(r, b), 0.0)
        dr = sign * np.sqrt(Rval) / r**2
        dphi = phi_rate(r, b)
        return [dr, dphi]
    r_pts, phi_pts = [], []
    y0 = [r0, 0.0]
    if r_min_true is None:
        def horizon_event(lam, y, sign):
            return y[0] - r_stop_horizon
        horizon_event.terminal = True
        horizon_event.direction = -1
        sol = solve_ivp(rhs, (0, lam_max), y0, args=(-1,), events=[horizon_event], max_step=0.01, rtol=1e-10, atol=1e-13)
        r_pts.append(sol.y[0])
        phi_pts.append(sol.y[1])
        captured = sol.status == 1
    else:
        r_turn = r_min_true * 1.02
        def turn_event(lam, y, sign):
            return y[0] - r_turn
        turn_event.terminal = True
        turn_event.direction = -1
        sol_in = solve_ivp(rhs, (0, lam_max), y0, args=(-1,), events=[turn_event], max_step=0.01, rtol=1e-10, atol=1e-13)
        r_pts.append(sol_in.y[0])
        phi_pts.append(sol_in.y[1])
        captured = False
        if sol_in.status == 1:
            y1 = sol_in.y[:, -1]
            def escape_event(lam, y, sign):
                return y[0] - r0
            escape_event.terminal = True
            escape_event.direction = 1
            sol_out = solve_ivp(rhs, (0, lam_max), y1, args=(1,), events=[escape_event], max_step=0.01, rtol=1e-10, atol=1e-13)
            r_pts.append(sol_out.y[0])
            phi_pts.append(sol_out.y[1])
    r_all = np.concatenate(r_pts)
    phi_all = np.concatenate(phi_pts)
    return r_all, phi_all, captured
prograde_b = [1.0, 2.0, 2.7, 4.0, 6.0, 9.0]
retrograde_b = [-3.0, -5.0, -6.5, -8.0, -10.0, -13.0]
prograde_trajectories = [(b, *integrate_photon(b)) for b in prograde_b]
retrograde_trajectories = [(b, *integrate_photon(b)) for b in retrograde_b]
print(f'Horizon: r+ = {r_horizon:.4f} M')
print(f'Ergosphere (equatorial): r = {r_ergo_equatorial:.4f} M')
print(f'Photon sphere prograde: r = {r_ph_pro:.4f} M, b_crit = {b_ph_pro:.4f} M')
print(f'Photon sphere retrograde: r = {r_ph_retro:.4f} M, b_crit = {b_ph_retro:.4f} M')
print(f'ISCO prograde: r = {r_isco_pro:.4f} M')
print(f'ISCO retrograde: r = {r_isco_retro:.4f} M')
fig, ax = plt.subplots(figsize=(9, 9))
horizon_circle = plt.Circle((0, 0), r_horizon, color='black', zorder=5, label='Event horizon')
ergo_circle = plt.Circle((0, 0), r_ergo_equatorial, color='gray', fill=False, linestyle='-.', linewidth=1.2, label='Ergosphere (equatorial)')
photon_sphere_pro_circle = plt.Circle((0, 0), r_ph_pro, color='darkorange', fill=False, linestyle='--', linewidth=1.3, label='Photon sphere (prograde)')
photon_sphere_retro_circle = plt.Circle((0, 0), r_ph_retro, color='purple', fill=False, linestyle='--', linewidth=1.3, label='Photon sphere (retrograde)')
isco_pro_circle = plt.Circle((0, 0), r_isco_pro, color='seagreen', fill=False, linestyle=':', linewidth=1.2, label='ISCO (prograde)')
isco_retro_circle = plt.Circle((0, 0), r_isco_retro, color='teal', fill=False, linestyle=':', linewidth=1.2, label='ISCO (retrograde)')
for circle in [horizon_circle, ergo_circle, photon_sphere_pro_circle, photon_sphere_retro_circle, isco_pro_circle, isco_retro_circle]:
    ax.add_patch(circle)
for i, (b, r_vals, phi_vals, captured) in enumerate(prograde_trajectories):
    x_vals = r_vals * np.cos(phi_vals)
    y_vals = r_vals * np.sin(phi_vals)
    color = plt.cm.Reds(0.5 + 0.45 * i / max(len(prograde_trajectories) - 1, 1)) if captured else plt.cm.Oranges(0.4 + 0.5 * i / max(len(prograde_trajectories) - 1, 1))
    status = 'captured' if captured else 'scattered'
    ax.plot(x_vals, y_vals, color=color, linewidth=1.2, linestyle='-', label=f'prograde b={b:.1f}M {status}')
for i, (b, r_vals, phi_vals, captured) in enumerate(retrograde_trajectories):
    x_vals = r_vals * np.cos(phi_vals)
    y_vals = r_vals * np.sin(phi_vals)
    color = plt.cm.Blues(0.5 + 0.45 * i / max(len(retrograde_trajectories) - 1, 1)) if captured else plt.cm.Purples(0.4 + 0.5 * i / max(len(retrograde_trajectories) - 1, 1))
    status = 'captured' if captured else 'scattered'
    ax.plot(x_vals, y_vals, color=color, linewidth=1.2, linestyle='--', label=f'retrograde b={b:.1f}M {status}')
ax.set_xlim(-15, 15)
ax.set_ylim(-15, 15)
ax.set_aspect('equal')
ax.set_xlabel('x / M')
ax.set_ylabel('y / M')
ax.set_title(f'Kerr (a = {a}M) photon orbits, spin-induced lensing asymmetry')
ax.legend(loc='upper right', fontsize=6, ncol=1)
fig.tight_layout()
fig.savefig('blackhole_kerr.png', dpi=150)
