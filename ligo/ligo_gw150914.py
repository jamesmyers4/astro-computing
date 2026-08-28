from gwpy.timeseries import TimeSeries
from gwosc.datasets import event_gps
import matplotlib.pyplot as plt
gps = event_gps('GW150914')
data = TimeSeries.fetch_open_data('H1', gps - 16, gps + 16, sample_rate=4096, cache=True, verbose=True)
white = data.whiten()
bandpassed = white.bandpass(35, 350)
zoomed = bandpassed.crop(gps - 0.2, gps + 0.1)
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(zoomed.times.value - gps, zoomed.value, color='C0')
ax.set_xlabel('Time since GW150914 (s)')
ax.set_ylabel('Whitened strain')
ax.set_title('GW150914 chirp, H1, whitened + bandpassed 35-350 Hz')
fig.savefig('ligo_gw150914_chirp.png', dpi=150)
