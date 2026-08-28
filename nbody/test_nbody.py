import numpy as np
import rebound
def test_nbody_semi_major_axes_stay_within_known_bounds():
    planet_names = ['Mercury', 'Venus', 'Earth', 'Mars']
    known_a = [0.38709893, 0.72333199, 1.00000011, 1.52366231]
    sim = rebound.Simulation()
    sim.units = ('yr', 'AU', 'Msun')
    sim.add('Sun')
    for name in planet_names:
        sim.add(name)
    sim.move_to_com()
    a0 = [sim.particles[i].a for i in range(1, 5)]
    sim.integrator = 'whfast'
    sim.dt = sim.particles[1].P / 30
    sample_times = np.linspace(0, 100, 300)
    semi_major_axes = [[] for _ in range(4)]
    positions = [[] for _ in range(4)]
    for t in sample_times:
        sim.integrate(t)
        for i in range(4):
            semi_major_axes[i].append(sim.particles[i + 1].a)
            positions[i].append((sim.particles[i + 1].x, sim.particles[i + 1].y))
    tolerance = 0.01
    band = 0.005
    for i in range(4):
        assert abs(a0[i] - known_a[i]) < tolerance
        a_arr = np.array(semi_major_axes[i])
        lower_bound = known_a[i] * (1 - band)
        upper_bound = known_a[i] * (1 + band)
        assert a_arr.min() > lower_bound
        assert a_arr.max() < upper_bound
    for i in range(4):
        path = np.array(positions[i])
        segment_lengths = np.hypot(np.diff(path[:, 0]), np.diff(path[:, 1]))
        total_path_length = segment_lengths.sum()
        assert total_path_length > 10 * a0[i]
