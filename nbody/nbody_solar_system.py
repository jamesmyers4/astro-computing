import numpy as np
import matplotlib.pyplot as plt
import rebound
planet_names = ['Mercury', 'Venus', 'Earth', 'Mars']
planet_colors = ['gray', 'goldenrod', 'royalblue', 'firebrick']
sim = rebound.Simulation()
sim.units = ('yr', 'AU', 'Msun')
sim.add('Sun')
for name in planet_names:
    sim.add(name)
sim.move_to_com()
a0 = [sim.particles[i].a for i in range(1, 5)]
sim.integrator = 'whfast'
sim.dt = sim.particles[1].P / 30
years = 100
n_samples = 8000
sample_times = np.linspace(0, years, n_samples)
positions = [[] for _ in range(4)]
semi_major_axes = [[] for _ in range(4)]
for t in sample_times:
    sim.integrate(t)
    for i in range(4):
        p = sim.particles[i + 1]
        positions[i].append((p.x, p.y))
        semi_major_axes[i].append(p.a)
for i, name in enumerate(planet_names):
    a_arr = np.array(semi_major_axes[i])
    print(f'{name}: a0 = {a0[i]:.6f} AU, min a = {a_arr.min():.6f} AU, max a = {a_arr.max():.6f} AU, max rel dev = {np.max(np.abs(a_arr - a0[i])) / a0[i]:.2e}')
fig, ax = plt.subplots(figsize=(8, 8))
ax.scatter([0], [0], color='gold', s=200, zorder=5, label='Sun')
for i, name in enumerate(planet_names):
    x_vals, y_vals = zip(*positions[i])
    ax.plot(x_vals, y_vals, color=planet_colors[i], linewidth=0.8, label=f'{name} (a={a0[i]:.3f} AU)')
ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)
ax.set_aspect('equal')
ax.set_xlabel('x (AU)')
ax.set_ylabel('y (AU)')
ax.set_title('N-body integration: Sun + rocky planets, 100 years (WHFast)')
ax.legend(loc='upper right', fontsize=8)
fig.tight_layout()
fig.savefig('nbody_solar_system.png', dpi=150)
