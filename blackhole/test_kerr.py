import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve
def test_kerr_photon_sphere_isco_and_capture_asymmetry():
    M = 1.0
    a = 0.9
    r_horizon = M + np.sqrt(M**2 - a**2)
    def R_photon(r, b):
        return r**4 + (a**2 - b**2) * r**2 + 2 * M * (b - a)**2 * r
    def dR_photon_dr(r, b):
        return 4 * r**3 + 2 * (a**2 - b**2) * r + 2 * M * (b - a)**2
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
    def bpt_isco(sign):
        Z1 = 1 + (1 - a**2 / M**2)**(1 / 3) * ((1 + a / M)**(1 / 3) + (1 - a / M)**(1 / 3))
        Z2 = np.sqrt(3 * a**2 / M**2 + Z1**2)
        return M * (3 + Z2 + sign * np.sqrt((3 - Z1) * (3 + Z1 + 2 * Z2)))
    def phi_rate(r, b):
        D = r**2 - 2 * M * r + a**2
        return (-(a - b) + (a / D) * (r**2 + a**2 - a * b)) / r**2
    def is_captured(b, r0=15.0, lam_max=200.0):
        def rhs(lam, y):
            r, phi = y
            Rval = max(R_photon(r, b), 0.0)
            dr = -np.sqrt(Rval) / r**2
            dphi = phi_rate(r, b)
            return [dr, dphi]
        def horizon_event(lam, y):
            return y[0] - r_horizon * 1.05
        horizon_event.terminal = True
        horizon_event.direction = -1
        sol = solve_ivp(rhs, (0, lam_max), [r0, 0.0], events=[horizon_event], max_step=0.01, rtol=1e-10, atol=1e-13)
        return sol.status == 1
    def photon_sphere_a0(vars):
        r, b = vars
        return [r**4 - b**2 * r**2 + 2 * M * b**2 * r, 4 * r**3 - 2 * b**2 * r + 2 * M * b**2]
    r_ph_a0, b_ph_a0 = fsolve(photon_sphere_a0, [3.0, 5.0])
    assert r_ph_pro < r_ph_retro
    assert abs(b_ph_pro) < abs(b_ph_retro)
    assert r_horizon < r_ph_pro < r_ph_retro
    assert r_isco_pro < r_isco_retro
    assert r_ph_pro < r_isco_pro
    assert r_ph_retro < r_isco_retro
    assert abs(r_isco_pro - bpt_isco(-1)) < 1e-3
    assert abs(r_isco_retro - bpt_isco(1)) < 1e-3
    assert is_captured(b_ph_pro - 0.05) == True
    assert is_captured(b_ph_pro + 0.05) == False
    assert is_captured(b_ph_retro + 0.05) == True
    assert is_captured(b_ph_retro - 0.05) == False
    assert abs(r_ph_a0 - 3 * M) < 0.01
    assert abs(b_ph_a0 - 3 * np.sqrt(3) * M) < 0.01
