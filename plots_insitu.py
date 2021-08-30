# This script is for plotting the in situ figures

from functions import *

if __name__ == '__main__':

    folder = 'data/'
    fn = '20180601_insitu.nc'
    ds = xr.open_dataset(folder+fn)

    den = False
    sm = False
    dh = False
    qg = True
    vvsect = False
    vvcomp = False

    date = ds.attrs['Date']
    dz = ds.attrs['Delta z [m]']
    bounds = ds.attrs['lonmin, lonmax, latmin, latmax']
    bounds += np.array([-(bounds[1]-bounds[0])/2., (bounds[1]-bounds[0])/2.,
                        -(bounds[3]-bounds[2])/5., (bounds[3]-bounds[2])/5.])

    vv_range = 20.
    ugeo_max = 1.
    deltadh = 0.01
    day_ind = 0
    depth = 10
    layer = int(round(depth / dz))

    sigma = ds.sigma
    DH = ds.DH
    Ugeo, Vgeo = ds.Ugeo, ds.Vgeo
    Wqg = ds.Wqg    
    sigma_sm = ds.sigma_sm
    DH_sm = ds.DH_sm
    Ugeo_sm, Vgeo_sm = ds.Ugeo_sm, ds.Vgeo_sm
    Wqg_sm = ds.Wqg_sm

    lon, lat = np.meshgrid(Wqg['longitude'].values, Wqg['latitude'].values)

    if den:

        fig, ax = plt.subplots(1, 1)
        fig.suptitle('Potential density\n'
                     'Depth = {}m / {}'.format(depth, date), y=0.93)
        chart = draw_alboran_sea(ax=ax, bounds=bounds, scalesize=30, skipgrid=0.25)
        draw_map_scalar_field(lon, lat, sigma_sm[day_ind, layer].values, ax=ax, map=chart,
                              cmap='magma_r', cbar_label='$\sigma_0$ ($kg.m^{-3}$)', bins=10)

    if dh:

        ### Dynamic height and geostrophic velocities

        fig, ax = plt.subplots(1, 2)
        fig.suptitle('Dynamic height and geostrophic velocities\n'
                     'Depth = {}m / {}'.format(depth, date), y=0.93)

        ax[0].set_title('Dynamic height', pad=15)
        chart = draw_alboran_sea(ax=ax[0], bounds=bounds, scalesize=30, skipgrid=0.25)
        draw_map_scalar_field(lon, lat, DH_sm.values[day_ind, layer] -
                              np.nanmean(DH_sm.values[day_ind, layer]),
                              ax=ax[0], map=chart, cmap='plasma', cbar_label='Dynamic height ($m$)',
                              levels=np.arange(-0.05, 0.05, deltadh), cbarticks=np.arange(-0.05, 0.05, 0.01))

        ax[1].set_title('Geostrophic velocities', pad=15)
        draw_alboran_sea(ax=ax[1], bounds=bounds, scalesize=30, skipgrid=0.25)
        draw_map_vector_field(lon, lat, Ugeo_sm.values[day_ind, layer],
                              Vgeo_sm.values[day_ind, layer], ax=ax[1], map=chart, cmap='Blues',
                              zoom=1.3, skip=2, cbar_label='Velocity ($m.s^{-1}$)', intensity_lim=ugeo_max)

    if qg:

        # W from QGOE

        fig, ax = plt.subplots()
        fig.suptitle('Vertical velocities from QG omega equation\n'
                     'Depth = {}m / {}'.format(depth, date), y=0.93)

        chart = draw_alboran_sea(ax=ax, bounds=bounds, scalesize=10, skipgrid=0.25)
        isolines = np.squeeze(DH.isel(depth=layer).interp(longitude=Wqg_sm['longitude'].values,
                              latitude=Wqg_sm['latitude'].values, kwargs={'fill_value': 'extrapolate'}).values)
        isolines = isolines - np.nanmean(isolines)
        draw_map_scalar_field(lon, lat, Wqg_sm.values[day_ind, layer], ax=ax, map=chart, cmap='seismic',
                              cbar_label='Velocity ($m.day^{-1}$)', values_bounds=(-vv_range, vv_range),
                              isolines=isolines, levels=np.arange(-0.1, 0.1, deltadh), shading='gouraud',
                              contour_legend='$\Delta DH = {:.2f}cm$'.format(deltadh*100))

    if sm:

        # With smoothing and without smoothing

        # Density

        fig, ax = plt.subplots(1, 3)
        fig.suptitle('Potential density before and after smoothing\n'
                     'Depth = {}m / {}'.format(depth, date), y=0.93)

        ax[0].set_title('Without smoothing', pad=15)
        chart = draw_alboran_sea(ax=ax[0], bounds=bounds, scalesize=30, skipgrid=0.25)
        draw_map_scalar_field(lon, lat, sigma.values[day_ind, layer], ax=ax[0], map=chart,
                              cmap='magma_r', cbar_label='$\sigma_{0} (kg.m^{-3})$')

        ax[1].set_title('With smoothing', pad=15)
        draw_alboran_sea(ax=ax[1], bounds=bounds, scalesize=30, skipgrid=0.25)
        draw_map_scalar_field(lon, lat, sigma_sm.values[day_ind, layer],
                              ax=ax[1], map=chart, cmap='magma_r', cbar_label='$\sigma_{0} (kg.m^{-3})$')

        diff = sigma_sm.values[day_ind, layer] - sigma.values[day_ind, layer]
        ax[2].set_title('Difference (rms: {:.2f})'.format(np.sqrt(np.nanmean(diff**2))), pad=15)
        chart = draw_alboran_sea(ax=ax[2], bounds=bounds, scalesize=30, skipgrid=0.25)
        draw_map_scalar_field(lon, lat, diff, ax=ax[2], map=chart, cmap='seismic', centered=True,
                              cbar_label='$\Delta \sigma_{0} (kg.m^{-3})$', cbarticks=np.arange(-0.009, 0.01, 0.003))

        # Vertical velocities from QGOE

        fig, ax = plt.subplots(1, 3)
        fig.suptitle('Vertical velocities from QGOE with and without smoothing\n'
                     'Depth = {}m / {}'.format(depth, date), y=0.93)

        ax[0].set_title('Without smoothing', pad=15)
        chart = draw_alboran_sea(ax=ax[0], bounds=bounds, scalesize=30, skipgrid=0.25)
        isolines = np.squeeze(DH.isel(depth=layer).interp(longitude=Wqg['longitude'].values,
                              latitude=Wqg['latitude'].values, kwargs={'fill_value': 'extrapolate'}).values)
        isolines = isolines - np.nanmean(isolines)
        draw_map_scalar_field(lon, lat, Wqg.values[day_ind, layer], ax=ax[0], map=chart,
                              values_bounds=(-vv_range, vv_range), cmap='seismic', cbar_label='Velocity ($m.day^{-1}$)',
                              isolines=isolines, levels=np.arange(-0.1, 0.1, deltadh),
                              contour_legend='$\Delta DH = {:.2f}cm$'.format(deltadh*100))

        ax[1].set_title('With smoothing', pad=15)
        draw_alboran_sea(ax=ax[1], bounds=bounds, scalesize=30, skipgrid=0.25)
        draw_map_scalar_field(lon, lat, Wqg_sm.values[day_ind, layer], values_bounds=(-vv_range, vv_range),
                              ax=ax[1], map=chart, cmap='seismic', cbar_label='Velocity ($m.day^{-1}$)',
                              isolines=isolines, levels=np.arange(-0.1, 0.1, deltadh),
                              contour_legend='$\Delta DH = {:.2f}cm$'.format(deltadh*100))

        diff = Wqg_sm.values[day_ind, layer] - Wqg.values[day_ind, layer]
        ax[2].set_title('Difference (rms: {:.2f})'.format(np.sqrt(np.nanmean(diff**2))), pad=15)
        chart = draw_alboran_sea(ax=ax[2], bounds=bounds, scalesize=30, skipgrid=0.25)
        draw_map_scalar_field(lon, lat, diff, ax=ax[2], map=chart, cmap='seismic', centered=True,
                              cbar_label='Velocity difference ($m.day^{-1}$)', values_bounds=(-vv_range, vv_range))

    if vvsect:

        A_B_depth = ((-2.22, 36.41), (-2.22, 35.95), (10., 150.))
        vsect = vsect_interp(Wqg, dim='lat', day_ind=day_ind, A_B_depth=A_B_depth)
        draw_vsect(vsect, cmap='seismic', values_bounds=(-100., 100.), cbar_label='Velocity ($m.day^{-1}$)')

    if vvcomp:

        ds = xr.open_dataset('data/output/datasets/20180602_z5_ref300.nc')
        bounds = ds.attrs['lonmin, lonmax, latmin, latmax']

        vvmod = ds.Wqg.values
        lonmod, latmod = ds.longitude.values, ds.latitude.values
        lonmod, latmod = np.meshgrid(lonmod, latmod)

        _, ax = plt.subplots()
        chart = draw_alboran_sea(ax=ax, bounds=bounds, skipgrid=0.25, scalesize=10)
        draw_map_scalar_field(lonmod, latmod, vvmod[day_ind, layer], map=chart, cmap='seismic',
                              values_bounds=(-vv_range, vv_range), shading='gouraud')
        draw_map_scalar_field(lon, lat, Wqg_sm.values[day_ind, layer], map=chart, cmap='seismic',
                              values_bounds=(-vv_range, vv_range), print_cbar=False)