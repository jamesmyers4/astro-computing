import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize_scalar, fsolve
def test_schwarzschild_photon_sphere_isco_and_capture():
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
    def Vpp(r, L):
        return 6 * L**2 / r**4 - 4 * M / r**3 - 24 * M * L**2 / r**5
    def photon_orbit_rhs(phi, y):
        u, dudphi = y
        return [dudphi, -u + 3 * M * u**2]
    def horizon_event(phi, y):
        return y[0] - 1 / r_horizon
    horizon_event.terminal = True
    horizon_event.direction = 1
    r0 = 40 * M
    u0 = 1 / r0
    phi_span = (0, 40)
    def is_captured(b):
        dudphi0 = np.sqrt(max(1 / b**2 - u0**2 * (1 - 2 * M * u0), 0))
        sol = solve_ivp(photon_orbit_rhs, phi_span, [u0, dudphi0], events=horizon_event, max_step=0.01)
        return sol.status == 1
    b_below_critical = b_crit - 0.05
    b_above_critical = b_crit + 0.05
    assert abs(r_photon_sphere - 3 * M) < 0.01
    assert abs(b_crit - 3 * np.sqrt(3) * M) < 0.01
    assert abs(r_isco - 6 * M) < 0.01
    assert abs(L_isco**2 - 12 * M**2) < 0.1
    assert Vpp(r_isco + 0.5, L_isco) > 0
    assert Vpp(r_isco - 0.5, L_isco) < 0
    assert is_captured(b_below_critical) == True
    assert is_captured(b_above_critical) == False
