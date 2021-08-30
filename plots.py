# This script is for plotting the model figures

from functions import *

if __name__ == '__main__':

    folder = 'data/'
    fn = '20180601_z1_ref300.nc'
    ds = xr.open_dataset(folder+fn)

    rawvars = False
    sm = False
    dh = False
    ce = False
    qg = True
    vvsect = False
    ts = False

    date = ds.attrs['Date']
    bounds = ds.attrs['lonmin, lonmax, latmin, latmax']
    dz = ds.attrs['Delta z [m]']
    scalesize = 100.
    skipgrid = 1.

    vv_range = 45.
    ugeo_max = 1.1
    deltadh = 0.01
    ageo_max = 0.4
    day_ind = 0
    depth = 50
    layer = int(round(depth / dz))

    SSH = ds.SSH
    SA = ds.SA
    SA_sm = ds.SA_sm
    PT = ds.PT
    PT_sm = ds.PT_sm
    CT = ds.CT
    sigma = ds.sigma
    DH = ds.DH
    Ugeo, Vgeo = ds.Ugeo, ds.Vgeo
    Ua, Va = ds.Ua, ds.Va
    U, V = Ugeo + Ua, Vgeo + Va
    Wc = ds.Wc
    Wqg = ds.Wqg
    Wqg_ns = ds.Wqg_ns

    lon, lat = np.meshgrid(Wqg['longitude'].values, Wqg['latitude'].values)

    if rawvars:

        fig, ax = plt.subplots(1, 3)
        fig.suptitle('Salinity, conservative temperature and potential density\n'
                     'Depth = {}m / {}'.format(depth, date), y=0.93)
        ax[0].set_title('Salinity', pad=15)
        chart = draw_alboran_sea(skipgrid=skipgrid, ax=ax[0], bounds=bounds, scalesize=scalesize)
        draw_map_scalar_field(lon, lat, SA_sm[day_ind, layer].values, ax=ax[0], map=chart,
                              cmap='viridis', cbar_label='Salinity ($ppt$)', bins=10)
        ax[1].set_title('Temperature', pad=15)
        draw_alboran_sea(skipgrid=skipgrid, ax=ax[1], bounds=bounds, scalesize=scalesize)
        draw_map_scalar_field(lon, lat, CT[day_ind, layer].values, ax=ax[1], map=chart,
                              cmap='plasma', cbar_label='Temperature ($^{\circ}C$)', bins=10)
        ax[2].set_title('Density', pad=15)
        draw_alboran_sea(skipgrid=skipgrid, ax=ax[2], bounds=bounds, scalesize=scalesize)
        draw_map_scalar_field(lon, lat, sigma[day_ind, layer].values, ax=ax[2], map=chart,
                              cmap='magma_r', cbar_label='$\sigma_0$ ($kg.m^{-3}$)', bins=10)

    if dh:

        ### SSH and dynamic height and difference

        fig, ax = plt.subplots(1, 3)
        fig.suptitle('Sea surface height and dynamic height from specific volume\n'
                     '{}'.format(date), y=0.93)

        ax[0].set_title('Sea surface height', pad=15)
        chart = draw_alboran_sea(skipgrid=skipgrid, ax=ax[0], bounds=bounds, scalesize=scalesize)
        draw_map_scalar_field(lon, lat, SSH.values[day_ind, 0] -
                              np.nanmean(SSH.values[day_ind, 0]), ax=ax[0], map=chart, shading='flat',
                              cmap='plasma', cbar_label='SSH ($m$)', levels=np.arange(-0.2, 0.21, deltadh))

        ax[1].set_title('Dynamic height at surface', pad=15)
        draw_alboran_sea(skipgrid=skipgrid, ax=ax[1], bounds=bounds, scalesize=scalesize)
        draw_map_scalar_field(lon, lat, DH.values[day_ind, 0] -
                              np.nanmean(DH.values[day_ind, 0]), shading='flat',
                              ax=ax[1], map=chart, cmap='plasma', cbar_label='Dynamic height ($m$)',
                              levels=np.arange(-0.2, 0.21, deltadh))

        diff = SSH.values[day_ind, 0] - DH.values[day_ind, 0]
        diff = diff - np.nanmean(diff)
        ax[2].set_title('Difference (rms: {:.2f})'.format(np.sqrt(np.nanmean(diff**2))), pad=15)
        chart = draw_alboran_sea(skipgrid=skipgrid, ax=ax[2], bounds=bounds, scalesize=scalesize)
        draw_map_scalar_field(lon, lat, diff * 100, ax=ax[2], map=chart, cmap='seismic', centered=True,
                              shading='flat', cbar_label='Height difference ($cm$)',
                              levels=np.arange(-4., 4., deltadh*100))

        ### Dynamic height and geostrophic velocities

        fig, ax = plt.subplots(1, 2)
        fig.suptitle('Dynamic height and geostrophic velocities\n'
                     'Depth = {}m / {}'.format(depth, date), y=0.93)

        ax[0].set_title('Dynamic height', pad=15)
        chart = draw_alboran_sea(skipgrid=skipgrid, ax=ax[0], bounds=bounds, scalesize=scalesize)
        draw_map_scalar_field(lon, lat, DH.values[day_ind, layer] -
                              np.nanmean(DH.values[day_ind, layer]),
                              ax=ax[0], map=chart, cmap='plasma', cbar_label='Dynamic height ($m$)', shading='flat',
                              levels=np.arange(-0.2, 0.21, deltadh), cbarticks=[-0.1, -0.05, 0., 0.05, 0.1])

        ax[1].set_title('Geostrophic velocities', pad=15)
        draw_alboran_sea(skipgrid=skipgrid, ax=ax[1], bounds=bounds, scalesize=scalesize)
        draw_map_vector_field(lon, lat, Ugeo.values[day_ind, layer],
                              Vgeo.values[day_ind, layer], ax=ax[1], map=chart, cmap='Blues', shading='flat',
                              zoom=3., skip=1, cbar_label='Velocity ($m.s^{-1}$)', intensity_lim=ugeo_max)

    if ce:

        ### Ageostrophic flow and W from CE

        fig, ax = plt.subplots(1, 2)
        fig.suptitle('Ageostrophic flow and vertical velocities from CE\n'
                     'Depth = {}m / {}'.format(depth, date), y=0.93)

        ax[0].set_title('Ageostrophic flow', pad=15)
        chart = draw_alboran_sea(skipgrid=skipgrid, ax=ax[0], bounds=bounds, scalesize=scalesize)
        draw_map_vector_field(lon, lat, Ua.values[day_ind, layer],
                              Va.values[day_ind, layer], ax=ax[0], map=chart, cmap='Blues', shading='flat',
                              zoom=3., skip=1, intensity_lim=ageo_max, cbar_label='Velocity ($m.s^{-1}$)')

        ax[1].set_title('Vertical velocities from CE', pad=15)
        draw_alboran_sea(skipgrid=skipgrid, ax=ax[1], bounds=bounds, scalesize=scalesize)
        draw_map_scalar_field(lon, lat, Wc.values[day_ind, layer], ax=ax[1], cmap='seismic', map=chart,
                              shading='fxlat', centered=True, values_bounds=(-vv_range, vv_range),
                              cbar_label='Velocity ($m.day^{-1}$)')

    if ce or qg:

        ### W from CE and W from QGOE

        fig, ax = plt.subplots(1, 3)
        plt.subplots_adjust(wspace=0.45)
        fig.suptitle('Vertical velocities from CE and from QGOE\n'
                     'Depth = {}m / {}'.format(depth, date), y=0.93)

        ax[0].set_title('Vertical velocities from CE', pad=15)
        chart = draw_alboran_sea(skipgrid=skipgrid, ax=ax[0], bounds=bounds, scalesize=scalesize)
        isolines = np.squeeze(DH.isel(depth=layer).interp(longitude=Wc['longitude'].values,
                              latitude=Wc['latitude'].values, kwargs={'fill_value': 'extrapolate'}).values)
        isolines = isolines - np.nanmean(isolines)
        draw_map_scalar_field(lon, lat, Wc.values[day_ind, layer], ax=ax[0], map=chart,
                              cmap='seismic', values_bounds=(-vv_range, vv_range), cbar_label='Velocity ($m.day^{-1}$)',
                              isolines=isolines, levels=np.arange(-0.2, 0.2, deltadh), shading='flat',
                              contour_legend='$\Delta DH = {:.2f}cm$'.format(deltadh*100))

        ax[1].set_title('Vertical velocities from QGOE', pad=15)
        draw_alboran_sea(skipgrid=skipgrid, ax=ax[1], bounds=bounds, scalesize=scalesize)
        isolines = np.squeeze(DH.isel(depth=layer).interp(longitude=Wqg['longitude'].values,
                              latitude=Wqg['latitude'].values, kwargs={'fill_value': 'extrapolate'}).values)
        isolines = isolines - np.nanmean(isolines)
        draw_map_scalar_field(lon, lat, Wqg.values[day_ind, layer], ax=ax[1], map=chart, cmap='seismic',
                              cbar_label='Velocity ($m.day^{-1}$)', values_bounds=(-vv_range, vv_range),
                              isolines=isolines, levels=np.arange(-0.2, 0.2, deltadh), shading='flat',
                              contour_legend='$\Delta DH = {:.2f}cm$'.format(deltadh*100))

        ax[2].set_xlabel('W from CE')
        ax[2].set_ylabel('W from QGOE')
        ax[2].set_xlim(-vv_range, vv_range)
        ax[2].set_ylim(-vv_range, vv_range)
        corr_plot(Wc.values[day_ind, layer], Wqg.values[day_ind, layer], c='orange', ax=ax[2])

    if qg:

        # W from QGOE

        fig, ax = plt.subplots()
        fig.suptitle('Vertical velocities from QG omega equation\n'
                     'Depth = {}m / {}'.format(depth, date), y=0.93)

        chart = draw_alboran_sea(skipgrid=skipgrid, ax=ax, bounds=bounds, scalesize=scalesize)
        isolines = np.squeeze(DH.isel(depth=layer).interp(longitude=Wqg['longitude'].values,
                              latitude=Wqg['latitude'].values, kwargs={'fill_value': 'extrapolate'}).values)
        isolines = isolines - np.nanmean(isolines)
        draw_map_scalar_field(lon, lat, Wqg.values[day_ind, layer], ax=ax, map=chart, cmap='seismic',
                              cbar_label='Velocity ($m.day^{-1}$)', values_bounds=(-vv_range, vv_range),
                              isolines=isolines, levels=np.arange(-0.2, 0.2, deltadh), shading='gouraud',
                              contour_legend='$\Delta DH = {:.2f}cm$'.format(deltadh*100))

    if sm:

        # With smoothing and without smoothing

        # Temperature

        fig, ax = plt.subplots(1, 3)
        fig.suptitle('Potential temperature before and after smoothing\n'
                     'Depth = {}m / {}'.format(depth, date), y=0.93)

        ax[0].set_title('Without smoothing', pad=15)
        chart = draw_alboran_sea(skipgrid=skipgrid, ax=ax[0], bounds=bounds, scalesize=scalesize)
        draw_map_scalar_field(lon, lat, PT.values[day_ind, layer], ax=ax[0], map=chart, shading='flat',
                              cmap='plasma', cbar_label='Temperature (°$C$)')

        ax[1].set_title('With smoothing', pad=15)
        draw_alboran_sea(skipgrid=skipgrid, ax=ax[1], bounds=bounds, scalesize=scalesize)
        draw_map_scalar_field(lon, lat, PT_sm.values[day_ind, layer], shading='flat',
                              ax=ax[1], map=chart, cmap='plasma', cbar_label='Temperature (°$C$)')

        diff = PT_sm.values[day_ind, layer] - PT.values[day_ind, layer]
        ax[2].set_title('Difference (rms: {:.2f})'.format(np.sqrt(np.nanmean(diff**2))), pad=15)
        chart = draw_alboran_sea(skipgrid=skipgrid, ax=ax[2], bounds=bounds, scalesize=scalesize)
        draw_map_scalar_field(lon, lat, diff, ax=ax[2], map=chart, cmap='seismic', centered=True, shading='flat',
                              cbar_label='$\Delta T$ (°$C$)')

        # Salinity

        fig, ax = plt.subplots(1, 3)
        fig.suptitle('Salinity before and after smoothing\n'
                     'Depth = {}m / {}'.format(depth, date), y=0.93)

        ax[0].set_title('Without smoothing', pad=15)
        chart = draw_alboran_sea(skipgrid=skipgrid, ax=ax[0], bounds=bounds, scalesize=scalesize)
        draw_map_scalar_field(lon, lat, SA.values[day_ind, layer], ax=ax[0], map=chart, shading='flat',
                              cmap='viridis', cbar_label='Salinity ($ppt$)')

        ax[1].set_title('With smoothing', pad=15)
        draw_alboran_sea(skipgrid=skipgrid, ax=ax[1], bounds=bounds, scalesize=scalesize)
        draw_map_scalar_field(lon, lat, SA_sm.values[day_ind, layer], shading='flat',
                              ax=ax[1], map=chart, cmap='viridis', cbar_label='Salinity ($ppt$)')

        diff = SA_sm.values[day_ind, layer] - SA.values[day_ind, layer]
        ax[2].set_title('Difference (rms: {:.2f})'.format(np.sqrt(np.nanmean(diff**2))), pad=15)
        chart = draw_alboran_sea(skipgrid=skipgrid, ax=ax[2], bounds=bounds, scalesize=scalesize)
        draw_map_scalar_field(lon, lat, diff, ax=ax[2], map=chart, cmap='seismic', centered=True, shading='flat',
                              cbar_label='$\Delta S$ ($ppt$)')

        # Vertical velocities from QGOE

        fig, ax = plt.subplots(1, 3)
        fig.suptitle('Vertical velocities from QGOE with and without smoothing\n'
                     'Depth = {}m / {}'.format(depth, date), y=0.93)

        ax[0].set_title('Without smoothing', pad=15)
        chart = draw_alboran_sea(skipgrid=skipgrid, ax=ax[0], bounds=bounds, scalesize=scalesize)
        isolines = np.squeeze(DH.isel(depth=layer).interp(longitude=Wqg['longitude'].values,
                              latitude=Wqg['latitude'].values, kwargs={'fill_value': 'extrapolate'}).values)
        isolines = isolines - np.nanmean(isolines)
        draw_map_scalar_field(lon, lat, Wqg_ns.values[day_ind, layer], ax=ax[0], map=chart,
                              values_bounds=(-vv_range, vv_range), cmap='seismic', cbar_label='Velocity ($m.day^{-1}$)',
                              isolines=isolines, levels=np.arange(-0.2, 0.2, deltadh), shading='flat',
                              contour_legend='$\Delta DH = {:.2f}cm$'.format(deltadh*100))

        ax[1].set_title('With smoothing', pad=15)
        draw_alboran_sea(skipgrid=skipgrid, ax=ax[1], bounds=bounds, scalesize=scalesize)
        draw_map_scalar_field(lon, lat, Wqg.values[day_ind, layer], values_bounds=(-vv_range, vv_range),
                              ax=ax[1], map=chart, cmap='seismic', cbar_label='Velocity ($m.day^{-1}$)',
                              isolines=isolines, levels=np.arange(-0.2, 0.2, deltadh), shading='flat',
                              contour_legend='$\Delta DH = {:.2f}cm$'.format(deltadh*100))

        diff = Wqg_ns.values[day_ind, layer] - Wqg.values[day_ind, layer]
        ax[2].set_title('Difference (rms: {:.2f})'.format(np.sqrt(np.nanmean(diff**2))), pad=15)
        chart = draw_alboran_sea(skipgrid=skipgrid, ax=ax[2], bounds=bounds, scalesize=scalesize)
        draw_map_scalar_field(lon, lat, diff, ax=ax[2], map=chart, cmap='seismic', centered=True, shading='flat',
                              cbar_label='Velocity difference ($m.day^{-1}$)', values_bounds=(-vv_range, vv_range),
                              isolines=isolines, levels=np.arange(-0.2, 0.2, deltadh),
                              contour_legend='$\Delta DH = {:.2f}cm$'.format(deltadh * 100)
                              )

    if vvsect:

        A_B_depth = ((-2.75, 35.7), (-2.25, 36.15), (10., 200.))
        vsect = vsect_interp(Wqg, day_ind=day_ind, A_B_depth=A_B_depth)
        draw_map_vsect(vsect, bounds=bounds, Xarr_2D=Wqg.isel(time=day_ind, depth=2), cmap='seismic', shading='gouraud',
                       values_bounds=(-vv_range, vv_range), levels=6, scalesize=scalesize, skipgrid=skipgrid)

    if ts:

        draw_map_ts_diagram(SA_sm, CT, coords=np.array([[-1.8, 35.6], [-2.7, 36.5]]),
                            Xarr_2D=sigma.isel(time=day_ind, depth=0),
                            bounds=bounds, map_cmap='magma_r', dsig=0.5, bins=10,
                            map_cbar_label='Density ($\mathit{kg} \hspace{0.05cm} m^{-3}$)')