import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize_scalar, fsolve
import matplotlib.pyplot as plt
M = 1.0
r_horizon = 2 * M
photon_sphere_objective = lambda u: -(u**2 * (1 - 2 * M * u))
photon_sphere_result = minimize_scalar(photon_sphere_objective, bounds=(1e-6, 1 / (2 * M) - 1e-6), method='bounded')
u_photon_sphere = photon_sphere_result.x
r_photon_sphere = 1 / u_photon_sphere
b_crit = 1 / np.sqrt(-photon_sphere_result.fun)
def isco_equations(vars):
    r, L = vars
    Vp = -2 * L**2 / r**3 + 2 * M / r**2 + 6 * M * L**2 / r**4
    Vpp = 6 * L**2 / r**4 - 4 * M / r**3 - 24 * M * L**2 / r**5
    return [Vp, Vpp]
r_isco, L_isco = fsolve(isco_equations, [6.0, 4.0])
def photon_orbit_rhs(phi, y):
    u, dudphi = y
    return [dudphi, -u + 3 * M * u**2]
def horizon_event(phi, y):
    return y[0] - 1 / r_horizon
horizon_event.terminal = True
horizon_event.direction = 1
r0 = 20 * M
u0 = 1 / r0
phi_span = (0, 30)
phi_eval = np.linspace(0, 30, 20000)
impact_parameters = [2.0, 4.0, b_crit - 0.3, b_crit - 0.05, b_crit + 0.05, b_crit + 0.3, 7.0, 10.0]
trajectories = []
for b in impact_parameters:
    dudphi0 = np.sqrt(max(1 / b**2 - u0**2 * (1 - 2 * M * u0), 0))
    sol = solve_ivp(photon_orbit_rhs, phi_span, [u0, dudphi0], events=horizon_event, max_step=0.01, t_eval=phi_eval)
    captured = sol.status == 1
    r_vals = 1 / sol.y[0]
    x_vals = r_vals * np.cos(sol.t)
    y_vals = r_vals * np.sin(sol.t)
    trajectories.append((b, captured, x_vals, y_vals))
print(f'Photon sphere: r = {r_photon_sphere:.4f} M (analytic 3M)')
print(f'Critical impact parameter: b_crit = {b_crit:.4f} M (analytic 3*sqrt(3) M = {3 * np.sqrt(3):.4f} M)')
print(f'ISCO: r = {r_isco:.4f} M, L = {L_isco:.4f} M (analytic 6M, L = sqrt(12) M = {np.sqrt(12):.4f} M)')
fig, ax = plt.subplots(figsize=(8, 8))
horizon_circle = plt.Circle((0, 0), r_horizon, color='black', zorder=5, label='Event horizon (2M)')
photon_sphere_circle = plt.Circle((0, 0), r_photon_sphere, color='darkorange', fill=False, linestyle='--', linewidth=1.5, label='Photon sphere (3M)')
isco_circle = plt.Circle((0, 0), r_isco, color='seagreen', fill=False, linestyle=':', linewidth=1.5, label='ISCO (6M)')
ax.add_patch(horizon_circle)
ax.add_patch(photon_sphere_circle)
ax.add_patch(isco_circle)
captured_trajectories = [t for t in trajectories if t[1]]
scattered_trajectories = [t for t in trajectories if not t[1]]
for i, (b, captured, x_vals, y_vals) in enumerate(captured_trajectories):
    color = plt.cm.Reds(0.5 + 0.45 * i / max(len(captured_trajectories) - 1, 1))
    label = f'b={b:.2f}M captured'
    ax.plot(x_vals, y_vals, color=color, linewidth=1.2, label=label)
for i, (b, captured, x_vals, y_vals) in enumerate(scattered_trajectories):
    color = plt.cm.Blues(0.9 - 0.55 * i / max(len(scattered_trajectories) - 1, 1))
    label = f'b={b:.2f}M scattered'
    ax.plot(x_vals, y_vals, color=color, linewidth=1.0, label=label)
ax.set_xlim(-20, 20)
ax.set_ylim(-20, 20)
ax.set_aspect('equal')
ax.set_xlabel('x / M')
ax.set_ylabel('y / M')
ax.set_title('Schwarzschild photon orbits, photon sphere, and ISCO')
ax.legend(loc='upper right', fontsize=7)
fig.tight_layout()
fig.savefig('blackhole_schwarzschild.png', dpi=150)
