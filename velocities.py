# This scripts is dedicated to vertical currents.

from functions import *

if __name__ == '__main__':

    Re = 6.4e6
    g = 9.81
    rho0 = 1026.
    f = sw.geostrophy.f(36.)
    dz = 10

    folder = 'data/'
    fn = '20180531to20180602_CMEMS_daily_masked.nc'
    data = xr.open_dataset(folder+fn)

    ref_depth = 300
    days = [2]
    smoothing_radius = 5000

    save = False
    folder = 'data/'
    fn = '20180602_z1_ref'+str(ref_depth)+'.nc'
    date = 'June 2nd, 2018'

    # Model data

    SSH = data['zos'].isel(time=days)
    SA = data['so'].isel(time=days)
    nt, nz, ny, nx = np.shape(SA.values)
    P = p_array_build(SA['depth'].values, nt, ny, nx, rho0=rho0, g=g)
    PT = data['thetao'].isel(time=days)
    U, V = data['uo'].isel(time=days), data['vo'].isel(time=days)

    # Filtering high spatial frequencies with median filter

    SA_sm = smooth_xarr(SA, method='bicubic', radius=smoothing_radius)
    PT_sm = smooth_xarr(PT, method='bicubic', radius=smoothing_radius)

    # Additional variables from smoothed variables

    CT = sw.conversions.CT_from_t(SA_sm, PT_sm, P)
    sigma = sw.density.sigma0(SA_sm, CT)
    DH = get_DH_from_TS(SA_sm, CT, no_motion=ref_depth, rho0=rho0, g=g)
    Ugeo, Vgeo = get_Ugeo_from_DH(DH, f=f)
    U_sm, V_sm = smooth_xarr(U, method='bicubic', radius=smoothing_radius),\
                 smooth_xarr(V, method='bicubic', radius=smoothing_radius)

    # Vertical velocities from continuity equation and ageostrophic velocities

    Wc = get_W_from_continuity(U_sm, V_sm)

    # Vertical velocities from QG quasi omega

    epsilon = 0.1
    forcing, N = get_forcing_N_from_Ugeo_sigma(Ugeo, Vgeo, sigma, rho0=rho0, g=g)
    Wqg = get_W_from_QGOE(forcing, N, epsilon=epsilon, f=f)

    # Vertical velocities without smoothing

    CT_ns = sw.conversions.CT_from_t(SA, PT, P)
    DH_ns = get_DH_from_TS(SA, CT_ns, no_motion=ref_depth, rho0=rho0, g=g)
    Ugeo_ns, Vgeo_ns = get_Ugeo_from_DH(DH_ns, f=f)
    sigma_ns = sw.density.sigma0(SA, CT_ns)
    forcing_ns, N_ns = get_forcing_N_from_Ugeo_sigma(Ugeo_ns, Vgeo_ns, sigma_ns, rho0=rho0, g=g)
    Wqg_ns = get_W_from_QGOE(forcing_ns, N_ns, epsilon=epsilon, f=f)

    ### Interpolation on the same nodes as Wqg

    SSH_interp = SSH.interp(longitude=Wqg.longitude, latitude=Wqg.latitude, kwargs={'fill_value': 'extrapolate'})
    SSH_interp = xr.DataArray(SSH_interp.values[:, np.newaxis, :, :], dims=('time', 'depth', 'latitude', 'longitude'),
                              coords={'time': Wqg.time, 'depth': [0], 'longitude': Wqg.longitude,
                                      'latitude': Wqg.latitude})
    SA_interp = SA.interp(depth=Wqg.depth, longitude=Wqg.longitude, latitude=Wqg.latitude,
                          kwargs={'fill_value': 'extrapolate'})
    PT_interp = PT.interp(depth=Wqg.depth, longitude=Wqg.longitude, latitude=Wqg.latitude,
                          kwargs={'fill_value': 'extrapolate'})
    SA_sm_interp = SA_sm.interp(depth=Wqg.depth, longitude=Wqg.longitude, latitude=Wqg.latitude,
                                kwargs={'fill_value': 'extrapolate'})
    PT_sm_interp = PT_sm.interp(depth=Wqg.depth, longitude=Wqg.longitude, latitude=Wqg.latitude,
                                kwargs={'fill_value': 'extrapolate'})
    U_interp, V_interp = U_sm.interp(depth=Wqg.depth, longitude=Wqg.longitude, latitude=Wqg.latitude,
                                     kwargs={'fill_value': 'extrapolate'}),\
                         V_sm.interp(depth=Wqg.depth, longitude=Wqg.longitude, latitude=Wqg.latitude,
                                     kwargs={'fill_value': 'extrapolate'})
    sigma_interp = sigma.interp(depth=Wqg.depth, longitude=Wqg.longitude, latitude=Wqg.latitude,
                                kwargs={'fill_value': 'extrapolate'})
    CT_interp = CT.interp(depth=Wqg.depth, longitude=Wqg.longitude, latitude=Wqg.latitude,
                          kwargs={'fill_value': 'extrapolate'})
    DH_interp = DH.interp(depth=Wqg.depth, longitude=Wqg.longitude, latitude=Wqg.latitude,
                          kwargs={'fill_value': 'extrapolate'})
    Ugeo_interp, Vgeo_interp = Ugeo.interp(depth=Wqg.depth, longitude=Wqg.longitude, latitude=Wqg.latitude,
                                           kwargs={'fill_value': 'extrapolate'}),\
                               Vgeo.interp(depth=Wqg.depth, longitude=Wqg.longitude, latitude=Wqg.latitude,
                                           kwargs={'fill_value': 'extrapolate'})
    Ua_interp, Va_interp = U_interp - Ugeo_interp, V_interp - Vgeo_interp
    Wc_interp = Wc.interp(depth=Wqg.depth, longitude=Wqg.longitude, latitude=Wqg.latitude,
                          kwargs={'fill_value': 'extrapolate'})
    forcing_interp = forcing.interp(depth=Wqg.depth, longitude=Wqg.longitude, latitude=Wqg.latitude,
                                    kwargs={'fill_value': 'extrapolate'})
    N_interp = N.interp(depth=Wqg.depth, kwargs={'fill_value': 'extrapolate'})

    ### Save

    if save:

        data_vars = {'SSH': SSH_interp,
                     'SA': SA_interp,
                     'SA_sm': SA_sm_interp,
                     'PT': PT_interp,
                     'PT_sm': PT_sm_interp,
                     'CT': CT_interp,
                     'sigma': sigma_interp,
                     'DH': DH_interp,
                     'Ugeo': Ugeo_interp,
                     'Vgeo': Vgeo_interp,
                     'Ua': Ua_interp,
                     'Va': Va_interp,
                     'Wc': Wc_interp,
                     'Wqg': Wqg,
                     'Wqg_ns': Wqg_ns}

        dataset = xr.Dataset(data_vars, attrs={'Description': 'Output dataset of T. Hermilly\'s code to compute '
                                                              'vertical velocities from quasi-geostrophic omega '
                                                              'equation.',
                                               'Date': date,
                                               'lonmin, lonmax, latmin, latmax': (min(Wqg.longitude),
                                                                                  max(Wqg.longitude),
                                                                                  min(Wqg.latitude),
                                                                                  max(Wqg.latitude)),
                                               'Delta z [m]': dz,
                                               'No motion level [m]': ref_depth,
                                               'HF filter scale [m]': smoothing_radius,
                                               'Coriolis coefficient [rad.s⁻¹]': np.round(f, 10),
                                               'Gravity value [N.m⁻²]': g,
                                               'Mean density [kg.m⁻³]': rho0,
                                               'Earth radius [m]': Re})
        dataset.to_netcdf(folder+fn)