import skgstat as skg
from pykrige.ok import OrdinaryKriging
from scipy.optimize import curve_fit
from skgstat import models
from functions import *

if __name__ == '__main__':

    folder = 'data/'
    fn = 'sigma_uctd.nc'
    NC = xr.open_dataset(folder + fn)
    save = False

    lon, lat = NC.lon.values, NC.lat.values
    limits = (np.nanmin(lon), np.nanmax(lon), np.nanmin(lat), np.nanmax(lat))
    gridx = np.arange(limits[0], limits[1], 0.015)
    gridy = np.arange(limits[2], limits[3], 0.015)
    mesh = np.meshgrid(gridx, gridy)
    stop = 151
    sigmarray = np.empty((stop, np.size(gridy), np.size(gridx)))
    vararray = np.empty((stop, np.size(gridy), np.size(gridx)))

    for d in range(stop):

        data = np.vstack([lon, lat, NC.sigma.values[d]])
        data = data[:, ~np.isnan(data).any(axis=0)]

        V = skg.Variogram(data[:2].T, data[2].T)
        V.estimator = 'matheron'
        V.model = 'spherical'
        xdata = V.bins
        ydata = V.experimental
        p0 = [np.mean(xdata), np.mean(ydata), 0]
        cof, cov = curve_fit(models.spherical, xdata, ydata, p0=p0)

        ran, sill, nug = cof[0], cof[1], cof[2]
        if nug < 0.:
            nug = 0.

        # Ordinary krigging object

        OK = OrdinaryKriging(
            data[0],
            data[1],
            data[2],
            variogram_model='spherical',
            variogram_parameters=[sill, ran, nug]
        )

        # Kriged grid and the variance grid.

        sigmarray[d], vararray[d] = OK.execute("grid", gridx, gridy)


    # Plots

    # d = 0
    #
    # data = np.vstack([lon, lat, NC.sigma.values[d]])
    # data = data[:, ~np.isnan(data).any(axis=0)]
    #
    # V = skg.Variogram(data[:2].T, data[2].T)
    # V.estimator = 'matheron'
    # V.model = 'spherical'
    # xdata = V.bins
    # ydata = V.experimental
    # p0 = [np.mean(xdata), np.mean(ydata), 0]
    # cof, cov = curve_fit(models.spherical, xdata, ydata, p0=p0)

    # fig, ax = plt.subplots()
    # V.plot(axes=ax)
    # fig.suptitle('Spherical variogram estimation at {}m depth'.format(d), y=0.9)
    # plt.show()
    #
    # fig = plt.figure()
    # fig.suptitle('Estimated potential density from kriging at {}m depth'.format(d), y=0.9)
    # chart = draw_alboran_sea(bounds=limits+np.array([-(limits[1]-limits[0])/2.3, (limits[1]-limits[0])/2.3,
    #                                                  -(limits[3]-limits[2])/4., (limits[3]-limits[2])/4.]),
    #                          skipgrid=0.25, scalesize=10)
    # draw_map_scalar_field(mesh[0], mesh[1], sigmarray[d], map=chart, cmap='magma_r', shading='flat',
    #                       cbar_label='$\sigma_{0} (kg.m^{-3})$', cbar_dir='vertical')
    # x, y = chart(data[0], data[1])
    # plt.scatter(x, y, c=data[2], s=12, cmap='magma_r')
    # plt.show()
    #
    # fig = plt.figure()
    # fig.suptitle('Variance from kriging at {}m depth'.format(d), y=0.9)
    # chart = draw_alboran_sea(bounds=limits+np.array([-(limits[1]-limits[0])/2.3, (limits[1]-limits[0])/2.3,
    #                                                  -(limits[3]-limits[2])/4., (limits[3]-limits[2])/4.]),
    #                          skipgrid=0.25, scalesize=10)
    # draw_map_scalar_field(mesh[0], mesh[1], vararray[d], map=chart, cmap='jet', shading='gouraud',
    #                       cbar_label='Variance', extend='both', cbar_dir='vertical')
    # x, y = chart(data[0], data[1])
    # plt.scatter(x, y, s=3, c='k')
    # plt.show()

    # Save

    if save:

        xrsig = xr.DataArray(sigmarray[np.newaxis, :151:10], dims=('time', 'depth', 'latitude', 'longitude'),
                             coords={'time': np.array(['2018-06-01']),
                                     'depth': np.arange(0., 151., 10.),
                                     'latitude': gridy,
                                     'longitude': gridx})
        xrvar = xr.DataArray(vararray[np.newaxis, :151:10], dims=('time', 'depth', 'latitude', 'longitude'),
                             coords={'time': np.array(['2018-06-01']),
                                     'depth': np.arange(0., 151., 10.),
                                     'latitude': gridy,
                                     'longitude': gridx})
        ds = xr.Dataset(data_vars={'sigma': xrsig, 'variance': xrvar},
                        coords={'time': np.array(['2018-06-01']),
                                'depth': np.arange(0., 151., 10.),
                                'latitude': gridy,
                                'longitude': gridx},
                        attrs={'Description': 'Kriging interpolation of potential density associated to the'
                                              ' 2018 CALYPSO cruise uCTD sections. Variance is given as variable'
                                              ' on top of the density array.',
                               'longitude grid': gridx,
                               'latitude grid': gridy,
                               'depth grid': np.arange(0., 151., 10.),
                               'Day': '2018-06-01',
                               'Author': 'T. Hermilly'})

        ds.to_netcdf('cal2018_sigma_kriged.nc')