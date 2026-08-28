import lightkurve as lk
import numpy as np
from astropy.time import Time
def test_flux_drops_below_threshold_during_kepler22b_transit():
    search = lk.search_lightcurve('Kepler-22', mission='Kepler', author='Kepler', exptime=1800)
    collection = search.download_all(download_dir='data')
    lc = collection.stitch().flatten(window_length=901).remove_outliers()
    period_days = 289.863876
    epoch = Time(2454966.7001, format='jd', scale='tdb')
    transit_duration_hours = 7.415
    folded = lc.fold(period=period_days, epoch_time=epoch, normalize_phase=True)
    half_width_phase = (transit_duration_hours / 24 / 2) / period_days
    in_transit = np.abs(folded.phase.value) < half_width_phase * 2
    out_of_transit = np.abs(folded.phase.value) > half_width_phase * 10
    min_in_transit_flux = np.nanmin(folded.flux.value[in_transit])
    baseline_flux = np.nanmean(folded.flux.value[out_of_transit])
    assert min_in_transit_flux < 0.9995
    assert min_in_transit_flux < baseline_flux
