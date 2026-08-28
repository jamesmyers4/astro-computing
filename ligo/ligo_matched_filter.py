from pycbc.catalog import Merger
from pycbc.filter import highpass, resample_to_delta_t, matched_filter
from pycbc.psd import interpolate, inverse_spectrum_truncation
from pycbc.waveform import get_td_waveform
import matplotlib.pyplot as plt
merger = Merger('GW150914')
strain = merger.strain('H1')
strain = highpass(strain, 15.0)
strain = resample_to_delta_t(strain, 1.0 / 2048)
conditioned = strain.crop(2, 2)
psd = conditioned.psd(4)
psd = interpolate(psd, conditioned.delta_f)
psd = inverse_spectrum_truncation(psd, int(4 * conditioned.sample_rate), low_frequency_cutoff=15)
hp, hc = get_td_waveform(approximant='SEOBNRv4_opt', mass1=36, mass2=29, delta_t=conditioned.delta_t, f_lower=20)
hp.resize(len(conditioned))
template = hp.cyclic_time_shift(hp.start_time)
snr = matched_filter(template, conditioned, psd=psd, low_frequency_cutoff=20)
snr = snr.crop(4, 4)
peak = abs(snr).numpy().argmax()
snr_peak = abs(snr[peak])
peak_time = snr.sample_times[peak]
print(f'Merger GPS time (GWOSC catalog): {merger.time}')
print(f'SNR peak: {snr_peak:.2f} at GPS time {peak_time:.3f} (offset {peak_time - merger.time:.4f}s)')
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(snr.sample_times - merger.time, abs(snr), color='C0')
ax.axvline(0, color='red', linestyle='--', label='Merger GPS time')
ax.set_xlabel('Time since GW150914 merger (s)')
ax.set_ylabel('Matched-filter SNR')
ax.set_title('GW150914 matched-filter SNR, H1, template m1=36 m2=29 Msun')
ax.legend()
fig.tight_layout()
fig.savefig('ligo_gw150914_snr.png', dpi=150)
