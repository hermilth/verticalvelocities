# This scripts is dedicated to vertical currents.

from functions import *

if __name__ == '__main__':

    folder = 'data/'
    data = xr.open_dataset(folder + 'cal2018_sigma_kriged.nc')

    Re = 6.4e6
    g = 9.81
    rho0 = 1026.
    f = sw.geostrophy.f(36.)

    ref_depth = 150
    dz = 10
    days = [0]
    smoothing_radius = 5000

    save = False
    savefolder = 'data/output/datasets/'
    fn = '20180601_insitu.nc'
    date = 'June 1st, 2018'

    # Cruise data

    sigma, var = data.sigma, data.variance
    sigma_sm = smooth_xarr(sigma, method='bicubic', radius=smoothing_radius)
    DH_sm = get_DH_from_sigma(sigma_sm, no_motion=ref_depth, rho0=rho0, g=g)
    Ugeo_sm, Vgeo_sm = get_Ugeo_from_DH(DH_sm, f=f)

    # Vertical velocities from QG quasi omega

    epsilon = 0.5
    forcing_sm, N_sm = get_forcing_N_from_Ugeo_sigma(Ugeo_sm, Vgeo_sm, sigma_sm, rho0=rho0, g=g)
    Wqg_sm = get_W_from_QGOE(forcing_sm, N_sm, epsilon=epsilon, f=f)

    # Same without smoothing

    DH = get_DH_from_sigma(sigma, no_motion=ref_depth, rho0=rho0, g=g)
    Ugeo, Vgeo = get_Ugeo_from_DH(DH, f=f)
    epsilon = 0.5
    forcing, N = get_forcing_N_from_Ugeo_sigma(Ugeo, Vgeo, sigma, rho0=rho0, g=g)
    Wqg = get_W_from_QGOE(forcing, N, epsilon=epsilon, f=f)

    ### Interpolation on the same nodes as Wqg

    sigma_interp = sigma.interp(depth=Wqg.depth, longitude=Wqg.longitude, latitude=Wqg.latitude,
                                kwargs={'fill_value': 'extrapolate'})
    DH_interp = DH.interp(depth=Wqg.depth, longitude=Wqg.longitude, latitude=Wqg.latitude,
                          kwargs={'fill_value': 'extrapolate'})
    Ugeo_interp, Vgeo_interp = Ugeo.interp(depth=Wqg.depth, longitude=Wqg.longitude, latitude=Wqg.latitude,
                                           kwargs={'fill_value': 'extrapolate'}), \
                               Vgeo.interp(depth=Wqg.depth, longitude=Wqg.longitude, latitude=Wqg.latitude,
                                           kwargs={'fill_value': 'extrapolate'})
    forcing_interp = forcing.interp(depth=Wqg.depth, longitude=Wqg.longitude, latitude=Wqg.latitude,
                                    kwargs={'fill_value': 'extrapolate'})
    N_interp = N.interp(depth=Wqg.depth, kwargs={'fill_value': 'extrapolate'})

    sigma_sm_interp = sigma_sm.interp(depth=Wqg.depth, longitude=Wqg.longitude, latitude=Wqg.latitude,
                                      kwargs={'fill_value': 'extrapolate'})
    DH_sm_interp = DH_sm.interp(depth=Wqg.depth, longitude=Wqg.longitude, latitude=Wqg.latitude,
                                kwargs={'fill_value': 'extrapolate'})
    Ugeo_sm_interp, Vgeo_sm_interp = Ugeo_sm.interp(depth=Wqg.depth, longitude=Wqg.longitude, latitude=Wqg.latitude,
                                                    kwargs={'fill_value': 'extrapolate'}), \
                                     Vgeo_sm.interp(depth=Wqg.depth, longitude=Wqg.longitude, latitude=Wqg.latitude,
                                                    kwargs={'fill_value': 'extrapolate'})
    forcing_sm_interp = forcing_sm.interp(depth=Wqg.depth, longitude=Wqg.longitude, latitude=Wqg.latitude,
                                          kwargs={'fill_value': 'extrapolate'})
    N_sm_interp = N_sm.interp(depth=Wqg.depth, kwargs={'fill_value': 'extrapolate'})

    ### Save

    if save:
        data_vars = {'sigma': sigma_interp,
                     'DH': DH_interp,
                     'Ugeo': Ugeo_interp,
                     'Vgeo': Vgeo_interp,
                     'Wqg': Wqg,
                     'sigma_sm': sigma_sm_interp,
                     'DH_sm': DH_sm_interp,
                     'Ugeo_sm': Ugeo_sm_interp,
                     'Vgeo_sm': Vgeo_sm_interp,
                     'Wqg_sm': Wqg_sm}

        dataset = xr.Dataset(data_vars, attrs={'Description': 'Output dataset of T. Hermilly\'s code to compute '
                                                              'vertical velocities from quasi-geostrophic omega '
                                                              'equation.',
                                               'Date': date,
                                               'lonmin, lonmax, latmin, latmax': (min(Wqg.longitude.values),
                                                                                  max(Wqg.longitude.values),
                                                                                  min(Wqg.latitude.values),
                                                                                  max(Wqg.latitude.values)),
                                               'Delta z [m]': dz,
                                               'No motion level [m]': ref_depth,
                                               'HF filter scale [m]': smoothing_radius,
                                               'Coriolis coefficient [rad.s⁻¹]': np.round(f, 10),
                                               'Gravity value [N.m⁻²]': g,
                                               'Mean density [kg.m⁻³]': rho0,
                                               'Earth radius [m]': Re})
        dataset.to_netcdf(savefolder + fn)
