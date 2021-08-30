# verticalvelocities
Quasi geostrophic omega equation integration on 4D arrays (t, x, y, z)

This code has been developped by Thomas Hermilly in the frame of an internship at the IMEDEA
research center in 2021, under the supervision of Simon Ruiz. It can be mofied and used for
non-lucrative purposes but must be properly cited. The associated report can be found in the
root folder as pdf.

This project provides functions to integrate temperature and salinity 4D arrays into vertical
velocities, using the quasi geostrophic omega equation. The finite differences are computed
at 1st order. Bathymetry sections are handled properly, with nonetheless edge effects. The
functions.py can be used as it is with the following packages: numpy, time, matplotlib, 
mpl_toolkits, gsw, xarray and scipy. It is adapted to python 3.6. The xarray DataArrays
used have always the dimensions 'time', 'depth', 'longitude' and 'latitude' in that order
and are therefore 4D DataArrays.

Example of functions you might need:

- bicubic_filter: filters scales below radius.
- get_zeta_from_uv: gives vorticity from zonal and meridional fields.
- get_DH_from_TS: gives dynamic height from temperature and salinity.
- get_W_from_QGOE: gives vertical velocities from the QGOE integration (iterative finite
differences) using a forcing term F = 2*div(Q). You can refer to Hoskin et al. 1977 for 
more info about this equation.
- draw_alboran_sea: draws a Basemap object on a matplotlib map of the Alboran sea.
- draw_map_scalar_field: draws the scalar field as xarray on the given axes.

Full python example in velocities.py.

Last updated: 2018-08-27
