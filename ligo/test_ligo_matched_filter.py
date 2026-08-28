from pycbc.catalog import Merger
from pycbc.filter import highpass, resample_to_delta_t, matched_filter
from pycbc.psd import interpolate, inverse_spectrum_truncation
from pycbc.waveform import get_td_waveform
import numpy as np
def test_gw150914_matched_filter_peak_significance():
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
    abs_snr = abs(snr).numpy()
    times = snr.sample_times.numpy()
    in_window = np.abs(times - merger.time) < 0.1
    peak_in_window = abs_snr[in_window].max()
    peak_outside_window = abs_snr[~in_window].max()
    assert peak_in_window > 8.0
    assert peak_in_window > peak_outside_window
