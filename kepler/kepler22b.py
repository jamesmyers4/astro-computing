import lightkurve as lk
import numpy as np
import matplotlib.pyplot as plt
from astropy.time import Time
search = lk.search_lightcurve('Kepler-22', mission='Kepler', author='Kepler', exptime=1800)
collection = search.download_all(download_dir='data')
lc = collection.stitch().flatten(window_length=901).remove_outliers()
period_days = 289.863876
epoch = Time(2454966.7001, format='jd', scale='tdb')
transit_duration_hours = 7.415
folded = lc.fold(period=period_days, epoch_time=epoch, normalize_phase=True)
binned = folded.bin(time_bin_size=0.0005)
half_width_phase = (transit_duration_hours / 24 / 2) / period_days
in_transit = np.abs(folded.phase.value) < half_width_phase * 2
out_of_transit = np.abs(folded.phase.value) > half_width_phase * 10
min_in_transit_flux = np.nanmin(folded.flux.value[in_transit])
baseline_flux = np.nanmean(folded.flux.value[out_of_transit])
print('minimum in-transit normalized flux:', min_in_transit_flux)
print('out-of-transit baseline flux:', baseline_flux)
print('transit depth (baseline - min):', baseline_flux - min_in_transit_flux)
fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(folded.phase.value, folded.flux.value, s=2, color='lightgray', alpha=0.5, label='unbinned')
ax.plot(binned.phase.value, binned.flux.value, color='C0', label='binned (0.0005 phase)')
ax.axvspan(-half_width_phase, half_width_phase, color='orange', alpha=0.2, label='transit window')
ax.set_xlim(-0.02, 0.02)
ax.set_ylim(0.9985, 1.0015)
ax.set_xlabel('Orbital phase')
ax.set_ylabel('Normalized flux')
ax.set_title('Kepler-22b phase-folded transit')
ax.legend()
fig.savefig('kepler22b_transit.png', dpi=150)
