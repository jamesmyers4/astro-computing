from astroquery.vizier import Vizier
from astroquery.sdss import SDSS
from astropy.coordinates import SkyCoord
import astropy.units as u
import numpy as np
import matplotlib.pyplot as plt
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
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
ax1.hist(color_ur[is_spiral], bins=25, alpha=0.6, density=True, color='C0', label=f'Spiral (n={is_spiral.sum()})')
ax1.hist(color_ur[is_elliptical], bins=25, alpha=0.6, density=True, color='C3', label=f'Elliptical (n={is_elliptical.sum()})')
ax1.set_xlabel('u - r color')
ax1.set_ylabel('normalized count')
ax1.set_title('Color by Galaxy Zoo morphology')
ax1.legend()
ax2.hist(concentration[is_spiral], bins=25, alpha=0.6, density=True, color='C0', label='Spiral')
ax2.hist(concentration[is_elliptical], bins=25, alpha=0.6, density=True, color='C3', label='Elliptical')
ax2.set_xlabel('concentration index (petroR90_r / petroR50_r)')
ax2.set_ylabel('normalized count')
ax2.set_title('Concentration by Galaxy Zoo morphology')
ax2.legend()
fig.suptitle('SDSS DR7 photometry + Galaxy Zoo 1 morphology (Lintott+ 2011)')
fig.tight_layout()
fig.savefig('sdss_galaxy_zoo.png', dpi=150)
