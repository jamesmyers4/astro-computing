from astroquery.vizier import Vizier
from astroquery.sdss import SDSS
from astropy.coordinates import SkyCoord
import astropy.units as u
import numpy as np
def test_galaxy_zoo_morphology_tracks_color_and_concentration():
    SDSS.TIMEOUT = 150
    vizier = Vizier(columns=['*'], row_limit=1000, column_filters={'S': 'M', 'fU': '!=1'})
    zoo = vizier.get_catalogs('J/MNRAS/410/166/galaxies')[0]
    zoo = zoo[(np.array(zoo['fE']) == 1) | (np.array(zoo['fS']) == 1)]
    coords = SkyCoord(ra=zoo['RAJ2000'], dec=zoo['DEJ2000'], unit=(u.hourangle, u.deg))
    photo = SDSS.query_crossid(coords, photoobj_fields=['objid', 'ra', 'dec', 'u', 'g', 'r', 'i', 'z', 'petroR50_r', 'petroR90_r'], data_release=17)
    matched_idx = np.array([int(name.split('_')[1]) for name in photo['name']])
    zoo_matched = zoo[matched_idx]
    color_ur = np.array(photo['u']) - np.array(photo['r'])
    concentration = np.array(photo['petroR90_r']) / np.array(photo['petroR50_r'])
    is_elliptical = np.array(zoo_matched['fE']) == 1
    is_spiral = np.array(zoo_matched['fS']) == 1
    valid = np.isfinite(color_ur) & np.isfinite(concentration) & (concentration > 0)
    color_ur = color_ur[valid]
    concentration = concentration[valid]
    is_elliptical = is_elliptical[valid]
    is_spiral = is_spiral[valid]
    assert is_elliptical.sum() > 50
    assert is_spiral.sum() > 50
    assert np.median(color_ur[is_elliptical]) > np.median(color_ur[is_spiral])
    assert np.median(concentration[is_elliptical]) > np.median(concentration[is_spiral])
