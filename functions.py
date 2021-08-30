# This script is for writing and testing the user's functions. Those are very
# talkative, I am sincerely sorry in advance.

### Imports

import time as t
import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
from mpl_toolkits.basemap import Basemap
import warnings as war
import gsw as sw
import xarray as xr
import scipy.interpolate as sci

### General functions

def secs_to_string(s):
    """
        Converts any number of seconds to an appropriate decomposition string.

        Parameters
        ----------
        s: float.
            Number of seconds

        Returns
        -------
        string: str
            The formatted string.

        Notes
        -----
        This code has been developped by T. Hermilly in the frame of an internship at the IMEDEA
        research center in 2021. It can be mofied and used for non-lucrative purposes but must be
        properly cited. It is adapted to python 3.6 and requires the following packages: numpy,
        matplotlib, time, mpl_toolkits, warnings, gsw, xarray and scipy. The xarray DataArrays
        used have always the dimensions 'time', 'depth', 'longitude' and 'latitude' in that order
        and are therefore 4D DataArrays.
        
        Last updated: 2018-08-27
    """

    if s < 1:

        return '{:.0f}ms'.format(s * 1000)

    elif s < 60:

        return '{:.2f}s'.format(s)

    elif s < 3600:

        m = int(s // 60)
        s = int(s % 60)
        return '{:02}m{:02}s'.format(m, s)

    else:

        h = int(s // 3600)
        m = int((s % 3600) // 60)
        s = int(s % 60)
        return '{:02}h{:02}m{:02}s'.format(h, m, s)


def barytraj(traj_array):
    """
        Returns the trajectory of the barycenter of N trajectories. Trajectories must
        have the same time sampling. If the time sampling has T steps, the traj_array variable
        must be an array of dimension N*T*2. All trajectories are weighted the same.

        Parameters
        ----------
        traj_array: array-like.
            The N*T*2 array containing the N-trajectories to average, with T time-steps of (lon, lat).

        Returns
        -------
        barytraj: array-like.
            The averaged trajectory of size T*2.

        Notes
        -----
        This code has been developped by T. Hermilly in the frame of an internship at the IMEDEA
        research center in 2021. It can be mofied and used for non-lucrative purposes but must be
        properly cited. It is adapted to python 3.6 and requires the following packages: numpy,
        matplotlib, time, mpl_toolkits, warnings, gsw, xarray and scipy. The xarray DataArrays
        used have always the dimensions 'time', 'depth', 'longitude' and 'latitude' in that order
        and are therefore 4D DataArrays.
        
        Last updated: 2018-08-27
    """

    return np.mean(traj_array, axis=0)

def bicubic_filter(A, dx, rad):
    """
        A 2D smoothing filter that uses a weigthed mean of the variable using bicubic
        weigths (x -> (1-x**3)**3) up to a limit radius. The grid has to have the same
        discretization in x and y.

        Parameters
        ----------
        A: 2D array-like.
            The array to be smoothed.
        dx: float.
            The regular grid spacing.
        rad: float.
            The search radius.

        Returns
        -------
        smoothed_array: 2D array-like.
            The smoothed array.

        Notes
        -----
        This code has been developped by T. Hermilly in the frame of an internship at the IMEDEA
        research center in 2021. It can be mofied and used for non-lucrative purposes but must be
        properly cited. It is adapted to python 3.6 and requires the following packages: numpy,
        matplotlib, time, mpl_toolkits, warnings, gsw, xarray and scipy. The xarray DataArrays
        used have always the dimensions 'time', 'depth', 'longitude' and 'latitude' in that order
        and are therefore 4D DataArrays.
        
        Last updated: 2018-08-27
    """

    def radius_search(arr, n, i, j):

        def bicubic(x, scale):
            if np.abs(x / scale) > 1:
                return 0.
            else:
                return (1 - np.abs(x / scale) ** 3) ** 3

        N, M = np.shape(arr)
        imin, imax, jmin, jmax = max(0, i - n), min(N - 1, i + n), max(0, j - n), min(M - 1, j + n + 1)

        list = []

        for ii in range(imin, imax + 1):
            for jj in range(jmin, jmax + 1):

                d = np.sqrt((ii - i) ** 2 + (jj - j) ** 2)

                if d <= np.float(n):
                    list.append([ii, jj, arr[ii, jj], bicubic(d, n + 1)])

        out = np.array(list)

        return out[~np.isnan(out).any(axis=1), :]

    n = int(rad / dx)
    N, M = np.shape(A)
    A_new = np.empty((N, M))
    mask = np.isnan(A)

    for i in range(N):
        for j in range(M):

            w = radius_search(A, n, i, j)
            s = np.sum(w[:, 3].squeeze())

            if s != 0:
                A_new[i, j] = np.sum(w[:, 2] * w[:, 3]) / s
            else:
                A_new[i, j] = np.nan

    A_new = np.where(mask, np.nan, A_new)

    return A_new


def smooth_xarr(xarr, method='median', nrows=1, smooth=2., radius=7000.):
    """
        A smoothing function for 4D DataArrays.

        Parameters
        ----------
        xarr: 4D DataArray.
            The 4D DataArray of dimensions 'time', 'depth', 'longitude', 'latitude' to be smoothed.
        method: {'median', 'bicubic', 'rbf'}, optional.
            The method to use. Default is median.
        nrows: int, optional.
            The number of rows each side of the center to use for median filtering. The
            footprint is therefore a 2*nrows+1 sized square, centered on the filtered point.
        smooth: float, optional.
            Smoothing factor for rbf filtering. Be careful for quantitative studies because
            this parameter has no physical meaning and should be adjusted visually. Default
            is 2.
        radius: float, optional.
            The search radius in meters for the bicubic filtering. Default is 7000.

        Returns
        -------
        smxarr: 4D DataArray.
            The smoothed DataArray.

        Notes
        -----
        This code has been developped by T. Hermilly in the frame of an internship at the IMEDEA
        research center in 2021. It can be mofied and used for non-lucrative purposes but must be
        properly cited. It is adapted to python 3.6 and requires the following packages: numpy,
        matplotlib, time, mpl_toolkits, warnings, gsw, xarray and scipy. The xarray DataArrays
        used have always the dimensions 'time', 'depth', 'longitude' and 'latitude' in that order
        and are therefore 4D DataArrays.
        
        Last updated: 2018-08-27
    """

    time = t.time()

    print('Beginning smoothing of the array...')

    if method == 'rbf':

        from scipy.interpolate import Rbf

        smarr = np.zeros(np.shape(xarr.values))

        for i in range(np.shape(smarr)[0]):
            for j in range(np.shape(smarr)[1]):
                arr = xarr.values[i, j]
                mask = np.isnan(arr)
                lon, lat = xarr['longitude'].values, xarr['latitude'].values
                lon, lat = np.meshgrid(lon, lat)
                darr = np.vstack([lon.flatten(), lat.flatten(), arr.flatten()])
                darr = darr[:, ~np.isnan(darr).any(axis=0)]
                rbf = Rbf(darr[0], darr[1], darr[2], function='multiquadric', smooth=smooth)
                eval = rbf(lon, lat)
                eval = np.where(mask, np.nan, eval)
                corr = np.nanmean(eval - arr)
                smarr[i, j] = eval - corr

        smxarr = xr.DataArray(smarr, dims=xarr.dims, coords=xarr.coords)
        print('Data array smoothing: {}'.format(secs_to_string(t.time() - time)))

    elif method == 'bicubic':

        dlat = xarr['latitude'].values[1] - xarr['latitude'].values[0]
        smarr = np.zeros(np.shape(xarr.values))

        for i in range(np.shape(smarr)[0]):
            for j in range(np.shape(smarr)[1]):
                print('Array smoothing: {:.0f}% done{}'.format(((np.shape(smarr)[1]) * i + j) /
                                                               (np.shape(smarr)[0]) / (np.shape(smarr)[1]) * 100,
                                                               '.' * (j % 3 + 1)), end='\r')
                arr = xarr.values[i, j]
                dh = dlat * (2 * np.pi * 6.4e6 / 360)
                smarr[i, j] = bicubic_filter(arr, dh, radius)

        smxarr = xr.DataArray(smarr, dims=xarr.dims, coords=xarr.coords)
        print('Data array smoothing: {}'.format(secs_to_string(t.time() - time)))

    else:

        from scipy import ndimage

        smarr = np.zeros(np.shape(xarr.values))

        for i in range(np.shape(smarr)[0]):
            for j in range(np.shape(smarr)[1]):
                smarr[i, j] = ndimage.median_filter(xarr.values[i, j],
                                                    footprint=np.ones((2 * nrows + 1, 2 * nrows + 1)),
                                                    mode='nearest')

        smarr = np.where(np.isnan(xarr.values), np.nan, smarr)
        smxarr = xr.DataArray(smarr, dims=xarr.dims, coords=xarr.coords)

        print('Data array smoothing: {}'.format(secs_to_string(t.time() - time)))
    print('Array smoothed!\n')

    return smxarr


def f_array_build(latlist, nt, nz, nx):
    """
        Builds and return a Coriolis factor 4D array of dimensions time, depth, lat, lon of resp. sizes nt,
        nz, latlist length, ny.

        Parameters
        ----------
        latlist: array-like.
            The latitudes where to compute the cCoriolis coefficient.
        nt: int.
            The size of the time dimension.
        nz: int.
            The size of the depth dimension.
        nx: int.
            The size of the longitude dimension.

        Returns
        -------
        farray: array-like.
            The 4D array containing the right Coriolis parameter at the right node.

        Notes
        -----
        This code has been developped by T. Hermilly in the frame of an internship at the IMEDEA
        research center in 2021. It can be mofied and used for non-lucrative purposes but must be
        properly cited. It is adapted to python 3.6 and requires the following packages: numpy,
        matplotlib, time, mpl_toolkits, warnings, gsw, xarray and scipy. The xarray DataArrays
        used have always the dimensions 'time', 'depth', 'longitude' and 'latitude' in that order
        and are therefore 4D DataArrays.
        
        Last updated: 2018-08-27
    """

    f_array = np.zeros((nt, nz, len(latlist), nx))

    for k, lat in enumerate(latlist):
        f_array[:, :, k, :] = np.ones((nt, nz, nx)) * sw.geostrophy.f(lat)

    return f_array

def vsect_interp(XA, day_ind=0, dim='lon', A_B_depth=None, nh=100, nz=50, start_ang_l_depth=None):
    """
        Returns a vertical section as a 2D DataArray of the given scalar field with dimensions 'lon' or
        'lat' and 'depth'.

        Parameters
        ----------
        XA: 4D DataArray.
            The DataArray of the data to intersect of dimensions 'time', 'depth', 'longitude', 'latitude'.
        day_ind: int, optional.
            The index of the day of the wanted vertical section.
        A_B_depth: [[lon_start, lon_stop], [lat_start, lat_stop], [depth_start, depth_stop]].
            The coordinates of the wanted section. Either this or start_ang_l_depth has to be not None.
        start_ang_l_depth: [[lon_start, lat_start], angle, length, [depth_start, depth_stop]].
            Another way of determining the location of the section, given a starting point, a direction
            angle (west is 0., north is pi/2), the section length and the depth limits. Either this or
            A_B_depth has to be not None.
        dim: {'lon', 'lat'}, optional.
            The dimension to keep in the output (there can only be one).
        nh: int, optional.
            The horizontal discretization. Default is 100.
        nz: int, optional.
            The vertical discretization. Default is 50

        Returns
        -------
        vsect: 2D DataArray.
            The DataArray of dimensions 'depth' and 'lon' or 'lat' containing the vertical section data.

        Notes
        -----
        This code has been developped by T. Hermilly in the frame of an internship at the IMEDEA
        research center in 2021. It can be mofied and used for non-lucrative purposes but must be
        properly cited. It is adapted to python 3.6 and requires the following packages: numpy,
        matplotlib, time, mpl_toolkits, warnings, gsw, xarray and scipy. The xarray DataArrays
        used have always the dimensions 'time', 'depth', 'longitude' and 'latitude' in that order
        and are therefore 4D DataArrays.
        
        Last updated: 2018-08-27
    """

    def grid(A_B_depth, nh, nz):

        lon = np.linspace(A_B_depth[0][0], A_B_depth[1][0], nh)
        lat = np.linspace(A_B_depth[0][1], A_B_depth[1][1], nh)
        depth = np.linspace(A_B_depth[2][0], A_B_depth[2][1], nz)

        grid = np.empty((nz, nh, 3))

        for i in range(nz):
            for j in range(nh):
                grid[i, j] = np.array([lon[j], lat[j], depth[i]])

        return grid


    if A_B_depth is not None:

        coords_dict = {'depth': XA['depth'].values,
                       'latitude': XA['latitude'].values,
                       'longitude': XA['longitude'].values}
        XA = xr.DataArray(XA.isel(time=day_ind).values.squeeze(), dims=('depth', 'latitude', 'longitude'),
                          coords=coords_dict)
        grid = grid(A_B_depth, nh, nz)

        lon_xa = xr.DataArray(grid[:, :, 0], dims=('depth', 'x'))
        lat_xa = xr.DataArray(grid[:, :, 1], dims=('depth', 'x'))
        depth_xa = xr.DataArray(grid[:, :, 2], dims=('depth', 'x'))

        vsect = XA.interp(depth=depth_xa, longitude=lon_xa, latitude=lat_xa, method='linear')
        attrs = {'start': A_B_depth[0], 'stop': A_B_depth[1], 'depth_range': A_B_depth[2]}

        if dim == 'lat':
            coords = {'depth': grid[:, 0, 2], 'lat': grid[0, :, 1]}
            vsect = xr.DataArray(vsect.values.squeeze(), dims=('depth', 'lat'), coords=coords, attrs=attrs)
        else:
            coords = {'depth': grid[:, 0, 2], 'lon': grid[0, :, 0]}
            vsect = xr.DataArray(vsect.values.squeeze(), dims=('depth', 'lon'), coords=coords, attrs=attrs)

    elif start_ang_l_depth is not None:

        A = start_ang_l_depth[0]
        B = (A[0] + start_ang_l_depth[2] * 1e3 / (2 * np.pi * 6.4e6 * np.cos(36 * np.pi / 180) / 360)
             * np.cos(start_ang_l_depth[1]),
             A[1] + start_ang_l_depth[2] * 1e3 / (2 * np.pi * 6.4e6 / 360) * np.sin(start_ang_l_depth[1]))
        depth = (start_ang_l_depth[3][0], start_ang_l_depth[3][1])

        vsect = vsect_interp(XA, dim=dim, A_B_depth=(A, B, depth), nh=nh, nz=nz)

    else:

        vsect = 'Either start_dist_angle_depth or A_B_depth sould be not None.' \
                ' Cannot proceed to transect interpolation.'

    return vsect


def get_zeta_from_uv(U, V):
    """
        Returns the vorticity scalar field associated to given velocities.

        Parameters
        ----------
        U: 4D DataArray.
            The zonal component of the velocities of dimensions 'time', 'depth', 'longitude', 'latitude'.
        V: 4D DataArray.
            The meridional component of the velocities of dimensions 'time', 'depth', 'longitude', 'latitude'.

        Returns
        -------
        zeta: 4D DataArray.
            The vorticity DataArray of dimensions 'time', 'depth', 'longitude' and 'latitude'.

        Notes
        -----
        This code has been developped by T. Hermilly in the frame of an internship at the IMEDEA
        research center in 2021. It can be mofied and used for non-lucrative purposes but must be
        properly cited. It is adapted to python 3.6 and requires the following packages: numpy,
        matplotlib, time, mpl_toolkits, warnings, gsw, xarray and scipy. The xarray DataArrays
        used have always the dimensions 'time', 'depth', 'longitude' and 'latitude' in that order
        and are therefore 4D DataArrays.
        
        Last updated: 2018-08-27
    """

    x, y = (U['longitude'].values - U['longitude'].values[0]) * \
           (2 * np.pi * 6.4e6 * np.cos(36 * np.pi / 180) / 360), \
           (U['latitude'].values - U['latitude'].values[0]) * (2 * np.pi * 6.4e6 / 360)

    u, v = U.values, V.values
    Udy = (u[:, :, 1:, :] - u[:, :, :-1, :]) / (y[1:] - y[:-1])[np.newaxis, np.newaxis, :, np.newaxis]
    Vdx = (v[:, :, :, 1:] - v[:, :, :, :-1]) / (x[1:] - x[:-1])[np.newaxis, np.newaxis, np.newaxis, :]
    Udy = (Udy[:, :, :, 1:] + Udy[:, :, :, :-1]) / 2.
    Vdx = (Vdx[:, :, 1:, :] + Vdx[:, :, :-1, :]) / 2.

    lon_mid, lat_mid = np.meshgrid((U['longitude'].values[1:] + U['longitude'].values[:-1]) / 2.,
                                   (U['latitude'].values[1:] + U['latitude'].values[:-1]) / 2.)
    zeta = Vdx - Udy

    dict = {'depth': U['depth'].values,
            'longitude': lon_mid,
            'latitude': lat_mid}
    zeta = xr.DataArray(zeta, dims=('depth', 'longitude', 'latitude'), coords=dict)

    return zeta


def get_W_from_continuity(U, V):
    """
        Computes the vertical velocities from the total or ageostrophic flow using the continuity
        equation du/dx + dv/dy + dw/dz = 0.

        Parameters
        ----------
        U: 4D DataArray.
            The zonal component of the velocities of dimensions 'time', 'depth', 'longitude', 'latitude'.
        V: 4D DataArray.
            The meridional component of the velocities of dimensions 'time', 'depth', 'longitude', 'latitude'.

        Returns
        -------
        W: 4D DataArray.
            The vertical velocities DataArray of dimensions 'time', 'depth', 'longitude' and 'latitude'.

        Notes
        -----
        This code has been developped by T. Hermilly in the frame of an internship at the IMEDEA
        research center in 2021. It can be mofied and used for non-lucrative purposes but must be
        properly cited. It is adapted to python 3.6 and requires the following packages: numpy,
        matplotlib, time, mpl_toolkits, warnings, gsw, xarray and scipy. The xarray DataArrays
        used have always the dimensions 'time', 'depth', 'longitude' and 'latitude' in that order
        and are therefore 4D DataArrays.
        
        Last updated: 2018-08-27
    """

    x, y = (U['longitude'].values - U['longitude'].values[0]) * (2 * np.pi * 6.4e6 *
                                                                 np.cos(36 * np.pi / 180) / 360), \
           (U['latitude'].values - U['latitude'].values[0]) * (2 * np.pi * 6.4e6 / 360)

    u, v = U.values, V.values
    Udx = (u[:, :, :, 1:] - u[:, :, :, :-1]) / (x[1:] - x[:-1])[np.newaxis, np.newaxis, np.newaxis, :]
    Vdy = (v[:, :, 1:, :] - v[:, :, :-1, :]) / (y[1:] - y[:-1])[np.newaxis, np.newaxis, :, np.newaxis]
    Udx = (Udx[:, :, 1:, :] + Udx[:, :, :-1, :]) / 2.
    Vdy = (Vdy[:, :, :, 1:] + Vdy[:, :, :, :-1]) / 2.
    Hdiv = Udx + Vdy

    W_hat = np.zeros(np.shape(Hdiv))
    dlist = U['depth'].values
    dz = dlist[1:] - dlist[:-1]

    for i in range(len(dz)):
        W_hat[:, i + 1, :, :] = W_hat[:, i, :, :] + Hdiv[:, i, :, :] * dz[i]

    W_hat = W_hat * 3600 * 24

    lon_mid, lat_mid = (U['longitude'].values[1:] + U['longitude'].values[:-1]) / 2., \
                       (U['latitude'].values[1:] + U['latitude'].values[:-1]) / 2.

    dict = {'time': U['time'].values,
            'depth': U['depth'].values,
            'longitude': lon_mid,
            'latitude': lat_mid}

    return xr.DataArray(W_hat, dims=('time', 'depth', 'latitude', 'longitude'), coords=dict)


def p_from_depth(d, rho0=1026., g=9.81):
    """
        Returns the pressure assoiated to a depth.

        Parameters
        ----------
        d: float.
            The depth to which pressure should be computed.
        rho0: float, optional.
            Mean density. Default is 1026.
        g: float, optional.
            Gravity value. Default is 9.81.

        Returns
        -------
        p: float.
            The pressure associated to depth d in dbar.

        Notes
        -----
        This code has been developped by T. Hermilly in the frame of an internship at the IMEDEA
        research center in 2021. It can be mofied and used for non-lucrative purposes but must be
        properly cited. It is adapted to python 3.6 and requires the following packages: numpy,
        matplotlib, time, mpl_toolkits, warnings, gsw, xarray and scipy. The xarray DataArrays
        used have always the dimensions 'time', 'depth', 'longitude' and 'latitude' in that order
        and are therefore 4D DataArrays.
        
        Last updated: 2018-08-27
    """

    return rho0 * g * d / 1e4


def depth_array_build(dlist, nt, nx, ny):
    """
        Builds and return a depth 4D array of dimensions time, depth, longitude, latitude.

        Parameters
        ----------
        dlist: array-like.
            The depth list associated to the discretization.
        nt: int.
            The size of the time dimension.
        nx: int.
            The size of the longitude dimension.
        ny: int.
            The size of the latitude dimension.

        Returns
        -------
        darray: array-like.
            The 4D array containing the right depth at the right node.

        Notes
        -----
        This code has been developped by T. Hermilly in the frame of an internship at the IMEDEA
        research center in 2021. It can be mofied and used for non-lucrative purposes but must be
        properly cited. It is adapted to python 3.6 and requires the following packages: numpy,
        matplotlib, time, mpl_toolkits, warnings, gsw, xarray and scipy. The xarray DataArrays
        used have always the dimensions 'time', 'depth', 'longitude' and 'latitude' in that order
        and are therefore 4D DataArrays.
        
        Last updated: 2018-08-27
    """

    darr = np.zeros((nt, len(dlist), nx, ny))

    for k, d in enumerate(dlist):
        darr[:, k, :, :] = np.ones((nt, nx, ny)) * d

    return darr


def p_array_build(dlist, nt, nx, ny, rho0=1026., g=9.81):
    """
        Builds and return a pressure 4D array of dimensions time, depth, longitude, latitude.

        Parameters
        ----------
        dlist: array-like.
            The depth list associated to the discretization.
        nt: int.
            The size of the time dimension.
        nx: int.
            The size of the longitude dimension.
        ny: int.
            The size of the latitude dimension.
        rho0: float, optional.
            Mean density. Default is 1026.
        g: float, optional.
            Gravity value. Default is 9.81.

        Returns
        -------
        parray: array-like.
            The 4D array containing the right pressure at the right node.

        Notes
        -----
        This code has been developped by T. Hermilly in the frame of an internship at the IMEDEA
        research center in 2021. It can be mofied and used for non-lucrative purposes but must be
        properly cited. It is adapted to python 3.6 and requires the following packages: numpy,
        matplotlib, time, mpl_toolkits, warnings, gsw, xarray and scipy. The xarray DataArrays
        used have always the dimensions 'time', 'depth', 'longitude' and 'latitude' in that order
        and are therefore 4D DataArrays.
        
        Last updated: 2018-08-27
    """

    p_array = depth_array_build(dlist, nt, nx, ny) * rho0 * g

    return p_array / 1e4

def filly(array):
    """
        Fills NaNs of a 4D array of dimensions time, depth, lon and lat with neighbor values
        along lat axis.

        Parameters
        ----------
        array: 4D array-like.
            The array to be filled.

        Returns
        -------
        filled_array: 4D array-like.
            The filled array.

        Notes
        -----
        This code has been developped by T. Hermilly in the frame of an internship at the IMEDEA
        research center in 2021. It can be mofied and used for non-lucrative purposes but must be
        properly cited. It is adapted to python 3.6 and requires the following packages: numpy,
        matplotlib, time, mpl_toolkits, warnings, gsw, xarray and scipy. The xarray DataArrays
        used have always the dimensions 'time', 'depth', 'longitude' and 'latitude' in that order
        and are therefore 4D DataArrays.
        
        Last updated: 2018-08-27
    """

    arr = array

    nt, nz, nx, ny = np.shape(arr)

    tstart = t.time()

    for i in range(nt):
        for j in range(nz):
            print('Array filling: {:.0f}% processed{}'.format((i * nz + j) / (nt * nz) * 50,
                                                              '.' * int(j % 3 + 1)), end='\r')
            for k in range(ny):
                for l in range(1, nx):

                    if np.isnan(arr[i, j, l, k]) & (not (np.isnan(arr[i, j, l - 1, k]))):
                        arr[i, j, l, k] = arr[i, j, l - 1, k]

    for i in range(nt):
        for j in range(nz):
            print('Array filling: {:.0f}% processed{}'.format(50 + (i * nz + j) / (nt * nz) * 50,
                                                              '.' * int(j % 3 + 1)), end='\r')
            for k in range(ny):
                for l in range(nx - 2, -1, -1):

                    if np.isnan(arr[i, j, l, k]) & (not (np.isnan(arr[i, j, l + 1, k]))):
                        arr[i, j, l, k] = arr[i, j, l + 1, k]

    print('Array filling time: {}'.format(secs_to_string(t.time() - tstart)))

    return arr


def get_DH_from_TS(SA, TH, no_motion=300., rho0=1026., g=9.81):
    """
        Computes the dynamic height integrating the specific volume of the Gibbs Sea Water package.

        Parameters
        ----------
        SA: 4D DataArray.
            The salinity DataArray of dimensions 'time', 'depth', 'longitude', 'latitude'.
        TH: 4D DataArray.
            The temperature DataArray of dimensions 'time', 'depth', 'longitude', 'latitude'.
        no_motion: float, optional.
            The no motin level. Default is 300.
        rho0: float, optional.
            Mean density. Default is 1026.
        g: float, optional.
            Gravity value. Default is 9.81.

        Returns
        -------
        DH: 4D DataArray.
            The 4D DataArray of dimensions 'time', 'depth', 'longitude', 'latitude' containing
            the dynamic height variable.

        Notes
        -----
        This code has been developped by T. Hermilly in the frame of an internship at the IMEDEA
        research center in 2021. It can be mofied and used for non-lucrative purposes but must be
        properly cited. It is adapted to python 3.6 and requires the following packages: numpy,
        matplotlib, time, mpl_toolkits, warnings, gsw, xarray and scipy. The xarray DataArrays
        used have always the dimensions 'time', 'depth', 'longitude' and 'latitude' in that order
        and are therefore 4D DataArrays.
        
        Last updated: 2018-08-27
    """

    time = t.time()

    ### Z levels
    i = 0
    while SA['depth'].values[i] < no_motion:
        i += 1
    z_list = SA['depth'].values[:i + 1]
    z_list[-1] = no_motion
    dz_list = z_list[1:] - z_list[:-1]

    ### Resampling salinity and temperature arrays for integration (until no motion level)
    sa = SA.isel(depth=[k for k in range(i + 1)])
    th = TH.isel(depth=[k for k in range(i + 1)])

    ### Bathymetry mask for later
    mask = np.isnan(sa.values)

    ### Building pressure array, in dbar, of same shape as salinity and temperature arrays
    nt, nz, nx, ny = np.shape(sa.values)
    P = p_array_build(z_list, nt, nx, ny, rho0=rho0, g=g)

    ### Filling salinity and temperature arrays along latitude
    print('Beginning to fill arrays...')
    sa_synth, th_synth = filly(sa.values), filly(th.values)
    print('Arrays filled!\n')

    ### Using gsw to have the specific volume according to sea water thermodynamic equation
    specvol = sw.density.specvol(sa_synth, th_synth, P)

    ### Integrating the specific volume from no motion level until surface level. Pressure in Pa.
    DH = np.zeros((nt, nz, nx, ny))

    for k in range(nz - 1, 0, -1):
        DH[:, k - 1] = DH[:, k] + specvol[:, k - 1] * dz_list[k - 1] * rho0 * g

    DH = DH / g  # To have dynamic meters

    ### Masking DH with bathymetry mask
    DH = np.where(mask, np.nan, DH)

    ### Dynamic height coordinates
    dict = {'time': sa['time'].values,
            'depth': z_list,
            'longitude': sa['longitude'].values,
            'latitude': sa['latitude'].values}
    DH = xr.DataArray(DH, dims=('time', 'depth', 'latitude', 'longitude'), coords=dict)

    print('Dynamic height computation: {}\n'.format(secs_to_string(t.time() - time)))

    return DH


def get_DH_from_sigma(sigma, no_motion=300., rho0=1026., g=9.81):
    """
        Computes the dynamic height integrating the specific volume of the Gibbs Sea Water package.

        Parameters
        ----------
        sigma: 4D DataArray.
            The potential density DataArray of dimensions 'time', 'depth', 'longitude', 'latitude'.
        no_motion: float, optional.
            The no motin level. Default is 300.
        rho0: float, optional.
            Mean density. Default is 1026.
        g: float, optional.
            Gravity value. Default is 9.81.

        Returns
        -------
        DH: 4D DataArray.
            The 4D DataArray of dimensions 'time', 'depth', 'longitude', 'latitude' containing
            the dynamic height variable.

        Notes
        -----
        This code has been developped by T. Hermilly in the frame of an internship at the IMEDEA
        research center in 2021. It can be mofied and used for non-lucrative purposes but must be
        properly cited. It is adapted to python 3.6 and requires the following packages: numpy,
        matplotlib, time, mpl_toolkits, warnings, gsw, xarray and scipy. The xarray DataArrays
        used have always the dimensions 'time', 'depth', 'longitude' and 'latitude' in that order
        and are therefore 4D DataArrays.

        Last updated: 2018-08-27
    """

    time = t.time()

    ### Z levels
    i = 0
    while sigma['depth'].values[i] < no_motion:
        i += 1
    z_list = sigma['depth'].values[:i + 1]
    z_list[-1] = no_motion
    dz_list = z_list[1:] - z_list[:-1]

    ### Resampling sigma array for integration (until no motion level)
    sig = sigma.isel(depth=[k for k in range(i + 1)])
    nt, nz, nx, ny = np.shape(sig.values)

    ### Bathymetry mask for later
    mask = np.isnan(sig.values)

    ### Filling salinity and temperature arrays along latitude
    print('Beginning to fill array...')
    sig_synth = filly(sig.values)
    print('Array filled!\n')

    ### Computing specific volume
    specvol = 1/(sig_synth+1000)

    ### Integrating the specific volume from no motion level until surface level. Pressure in Pa.
    DH = np.zeros((nt, nz, nx, ny))

    for k in range(nz - 1, 0, -1):
        DH[:, k - 1] = DH[:, k] + specvol[:, k - 1] * dz_list[k - 1] * rho0 * g

    DH = DH / g  # To have dynamic meters

    ### Masking DH with bathymetry mask
    DH = np.where(mask, np.nan, DH)

    ### Dynamic height coordinates
    dict = {'time': sig['time'].values,
            'depth': z_list,
            'longitude': sig['longitude'].values,
            'latitude': sig['latitude'].values}
    DH = xr.DataArray(DH, dims=('time', 'depth', 'latitude', 'longitude'), coords=dict)

    print('Dynamic height computation: {}\n'.format(secs_to_string(t.time() - time)))

    return DH

def get_Ugeo_from_DH(DH, g=9.81, f=8.57e-5):
    """
        Computes the geostrophic velocities from the dynamic height.

        Parameters
        ----------
        DH: 4D DataArray.
            The dynamic height DataArray of dimensions 'time', 'depth', 'longitude', 'latitude'.
        g: float, optional.
            Gravity value. Default is 9.81.
        f: float, optional.
            The Coriolis parameter. Default is f=8.57e-5.

        Returns
        -------
        Ugeo: 4D DataArray.
            The 4D DataArray of dimensions 'time', 'depth', 'longitude', 'latitude' containing
            the zonal component of the geostrophic velocity.
        Vgeo: 4D DataArray.
            The 4D DataArray of dimensions 'time', 'depth', 'longitude', 'latitude' containing
            the meridional component of the geostrophic velocity.

        Notes
        -----
        This code has been developped by T. Hermilly in the frame of an internship at the IMEDEA
        research center in 2021. It can be mofied and used for non-lucrative purposes but must be
        properly cited. It is adapted to python 3.6 and requires the following packages: numpy,
        matplotlib, time, mpl_toolkits, warnings, gsw, xarray and scipy. The xarray DataArrays
        used have always the dimensions 'time', 'depth', 'longitude' and 'latitude' in that order
        and are therefore 4D DataArrays.
        
        Last updated: 2018-08-27
    """

    time = t.time()
    x, y = (DH['longitude'].values - DH['longitude'].values[0]) * (2 * np.pi * 6.4e6 * np.cos(36 * np.pi / 180) / 360), \
           (DH['latitude'].values - DH['latitude'].values[0]) * (2 * np.pi * 6.4e6 / 360)

    dynh = DH.values

    Ugeo, Vgeo = - (dynh[:, :, 1:, :] - dynh[:, :, :-1, :]) / (y[1:] - y[:-1])[np.newaxis, np.newaxis, :, np.newaxis], \
                 (dynh[:, :, :, 1:] - dynh[:, :, :, :-1]) / (x[1:] - x[:-1])[np.newaxis, np.newaxis, np.newaxis, :]
    Ugeo, Vgeo = (Ugeo[:, :, :, 1:] + Ugeo[:, :, :, :-1]) / 2., (Vgeo[:, :, 1:, :] + Vgeo[:, :, :-1, :]) / 2.
    Ugeo, Vgeo = g * Ugeo / f, g * Vgeo / f

    lon_mid, lat_mid = (DH['longitude'].values[1:] + DH['longitude'].values[:-1]) / 2., \
                       (DH['latitude'].values[1:] + DH['latitude'].values[:-1]) / 2.
    dict = {'time': DH['time'].values,
            'depth': DH['depth'].values,
            'longitude': lon_mid,
            'latitude': lat_mid}
    Ugeo, Vgeo = xr.DataArray(Ugeo, dims=('time', 'depth', 'latitude', 'longitude'), coords=dict), \
                 xr.DataArray(Vgeo, dims=('time', 'depth', 'latitude', 'longitude'), coords=dict)
    print('Geostrophic velocities computation: {}\n'.format(secs_to_string(t.time() - time)))

    return Ugeo, Vgeo


def get_mean_N(rho, rho0=1026., g=9.81):
    """
        Returns the 1D DataArray time, longitude and latitude averaged Brunt-Vassaila frequency.

        Parameters
        ----------
        rho: 4D DataArray.
            The potential density DataArray of dimensions 'time', 'depth', 'longitude' and 'latitude'.
        rho0: float, optional.
            Mean density. Default is 1026.
        g: float, optional.
            Gravity value. Default is 9.81.

        Returns
        -------
        N: 1D DataArray.
            The time, longitude and latitude averaged Brunt-Vassaila frequency array (hence is same size as
            rho 'depth' dimension).

        Notes
        -----
        This code has been developped by T. Hermilly in the frame of an internship at the IMEDEA
        research center in 2021. It can be mofied and used for non-lucrative purposes but must be
        properly cited. It is adapted to python 3.6 and requires the following packages: numpy,
        matplotlib, time, mpl_toolkits, warnings, gsw, xarray and scipy. The xarray DataArrays
        used have always the dimensions 'time', 'depth', 'longitude' and 'latitude' in that order
        and are therefore 4D DataArrays.
        
        Last updated: 2018-08-27
    """

    time = t.time()
    dlist = rho['depth'].values
    rho4d = rho.values
    rhodz = (rho4d[:, 1:, :, :] - rho4d[:, :-1, :, :]) / \
            (dlist[1:] - dlist[:-1])[np.newaxis, :, np.newaxis, np.newaxis]
    rhodz = np.nanmean(rhodz, axis=(0, 2, 3))
    N = np.sqrt(np.abs(g / rho0 * rhodz))
    N = xr.DataArray(N, dims='depth', coords={'depth': (dlist[1:] + dlist[:-1]) / 2})

    print('Average Brunt-Vassaila frequency computation: {}\n'.format(secs_to_string(t.time() - time)))

    return N


def get_forcing_from_Ugeo_sigma(XUgeo, XVgeo, Xsigma, rho0=1026., g=9.81):
    """
        Returns the quasi-geostrophic omega equation forcing term DataArray of dimensions 'time', 'depth',
        'longitude' and 'latitude'.

        Parameters
        ----------
        XUgeo: 4D DataArray.
            The geostrophic velocity zonal component DataArray.
        XVgeo: 4D DataArray.
            The geostrophic velocity meridional component DataArray of same size as XUgeo.
        Xsigma: 4D DataArray.
            The potential density DataArray, which the code linearily interpolates to the XUgeo and XVgeo
            nodes.
        rho0: float, optional.
            Mean density. Default is 1026.
        g: float, optional.
            Gravity value. Default is 9.81.

        Returns
        -------
        forcing: 4D DataArray.
            The forcing DataArray derived as 2div(Q), with Q = (du/dx*drho/dx + dv/dx*drho/dy,
            du/dy*drho/dx + dv/dy*drho/dy).

        Notes
        -----
        This code has been developped by T. Hermilly in the frame of an internship at the IMEDEA
        research center in 2021. It can be mofied and used for non-lucrative purposes but must be
        properly cited. It is adapted to python 3.6 and requires the following packages: numpy,
        matplotlib, time, mpl_toolkits, warnings, gsw, xarray and scipy. The xarray DataArrays
        used have always the dimensions 'time', 'depth', 'longitude' and 'latitude' in that order
        and are therefore 4D DataArrays.
        
        Last updated: 2018-08-27
    """

    time = t.time()
    lon_list, lat_list = XUgeo['longitude'].values, XUgeo['latitude'].values
    x, y = lon_list * (2 * np.pi * 6.4e6 * np.cos(36 * np.pi / 180) / 360), \
           lat_list * (2 * np.pi * 6.4e6 / 360)
    Xsigma_interp = Xsigma.interp(depth=XUgeo['depth'].values, longitude=XUgeo['longitude'].values,
                                  latitude=XUgeo['latitude'].values, kwargs={'fill_value': 'extrapolate'})
    Ugeo, Vgeo, sigma = XUgeo.values, XVgeo.values, Xsigma_interp.values

    Udx = (Ugeo[:, :, :, 1:] - Ugeo[:, :, :, :-1]) / (x[1:] - x[:-1])[np.newaxis, np.newaxis, np.newaxis, :]
    Vdx = (Vgeo[:, :, :, 1:] - Vgeo[:, :, :, :-1]) / (x[1:] - x[:-1])[np.newaxis, np.newaxis, np.newaxis, :]
    Udy = (Ugeo[:, :, 1:, :] - Ugeo[:, :, :-1, :]) / (y[1:] - y[:-1])[np.newaxis, np.newaxis, :, np.newaxis]
    Vdy = (Vgeo[:, :, 1:, :] - Vgeo[:, :, :-1, :]) / (y[1:] - y[:-1])[np.newaxis, np.newaxis, :, np.newaxis]
    sigmadx = (sigma[:, :, :, 1:] - sigma[:, :, :, :-1]) / (x[1:] - x[:-1])[np.newaxis, np.newaxis, np.newaxis, :]
    sigmady = (sigma[:, :, 1:, :] - sigma[:, :, :-1, :]) / (y[1:] - y[:-1])[np.newaxis, np.newaxis, :, np.newaxis]
    Udx = (Udx[:, :, 1:, :] + Udx[:, :, :-1, :]) / 2.
    Vdx = (Vdx[:, :, 1:, :] + Vdx[:, :, :-1, :]) / 2.
    Udy = (Udy[:, :, :, 1:] + Udy[:, :, :, :-1]) / 2.
    Vdy = (Vdy[:, :, :, 1:] + Vdy[:, :, :, :-1]) / 2.
    sigmadx = (sigmadx[:, :, 1:, :] + sigmadx[:, :, :-1, :]) / 2.
    sigmady = (sigmady[:, :, :, 1:] + sigmady[:, :, :, :-1]) / 2.

    Qx = (g / rho0) * (Udx * sigmadx + Vdx * sigmady)
    Qy = (g / rho0) * (Udy * sigmadx + Vdy * sigmady)

    x, y = (x[1:] + x[:-1]) / 2., (y[1:] + y[:-1]) / 2.

    Qxdx = (Qx[:, :, :, 1:] - Qx[:, :, :, :-1]) / (x[1:] - x[:-1])[np.newaxis, np.newaxis, np.newaxis, :]
    Qydy = (Qy[:, :, 1:, :] - Qy[:, :, :-1, :]) / (y[1:] - y[:-1])[np.newaxis, np.newaxis, :, np.newaxis]
    Qxdx = (Qxdx[:, :, 1:, :] + Qxdx[:, :, :-1, :]) / 2.
    Qydy = (Qydy[:, :, :, 1:] + Qydy[:, :, :, :-1]) / 2.

    forcing = 2 * (Qxdx + Qydy)
    dict = {'time': XUgeo['time'],
            'depth': XUgeo['depth'].values,
            'longitude': lon_list[1:-1],
            'latitude': lat_list[1:-1]}
    forcing = xr.DataArray(forcing, dims=('time', 'depth', 'latitude', 'longitude'), coords=dict)

    print('Forcing computation time: {}\n'.format(secs_to_string(t.time() - time)))

    return forcing


def get_forcing_N_from_Ugeo_sigma(XUgeo, XVgeo, Xsigma, rho0=1026., g=9.81):
    """
        Returns both the time, longitude and latitude averaged Brunt-Vassaila frequency 1D DataArray of
        dimension 'depth' and the quasi-geostrophic omega equation forcing term 4D DataArray of dimensions
        'time', 'depth', 'longitude' and 'latitude'.

        Parameters
        ----------
        XUgeo: 4D DataArray.
            The geostrophic velocity zonal component DataArray.
        XVgeo: 4D DataArray.
            The geostrophic velocity meridional component DataArray of same size as XUgeo.
        Xsigma: 4D DataArray.
            The potential density DataArray, which the code linearily interpolates to the XUgeo and XVgeo
            nodes.
        rho0: float, optional.
            Mean density. Default is 1026.
        g: float, optional.
            Gravity value. Default is 9.81.

        Returns
        -------
        forcing: 4D DataArray.
            The forcing DataArray derived as 2div(Q), with Q = (du/dx*drho/dx + dv/dx*drho/dy,
            du/dy*drho/dx + dv/dy*drho/dy).
        N: 1D DataArray.
            The time, longitude and latitude averaged Brunt-Vassaila frequency array (hence is same size as
            rho 'depth' dimension).

        Notes
        -----
        This code has been developped by T. Hermilly in the frame of an internship at the IMEDEA
        research center in 2021. It can be mofied and used for non-lucrative purposes but must be
        properly cited. It is adapted to python 3.6 and requires the following packages: numpy,
        matplotlib, time, mpl_toolkits, warnings, gsw, xarray and scipy. The xarray DataArrays
        used have always the dimensions 'time', 'depth', 'longitude' and 'latitude' in that order
        and are therefore 4D DataArrays.
        
        Last updated: 2018-08-27
    """

    forcing = get_forcing_from_Ugeo_sigma(XUgeo, XVgeo, Xsigma, rho0=rho0, g=g)
    Xsigma_interp = Xsigma.interp(depth=XUgeo['depth'].values, longitude=XUgeo['longitude'].values,
                                  latitude=XUgeo['latitude'].values, kwargs={'fill_value': 'extrapolate'})
    N = get_mean_N(Xsigma_interp, rho0=rho0, g=g)

    return forcing, N


def get_W_atzlevel_from_QGOE(Forcing, BVF, depth=None, layer=None, n_levels=6, epsilon=0.1, max_loops=500,
                             min_loops=20, return_all=False, f=8.57e-5):
    """
        Returns the vertical velocities DataArray of dimensions 'time', 'depth', 'longitude' and 'latitude'
        at a single depth layer integrated from the quasi-geostrophic omega equation on a sub-domain in depth.
        This is an auxiliary function used by get_W_from_QGOE that can be used very carefully to go faster.
        Keep in mind that the output is drastically underestimated.

        Parameters
        ----------
        Forcing: 4D DataArray.
            The forcing DataArray derived as 2div(Q), with Q = (du/dx*drho/dx + dv/dx*drho/dy,
            du/dy*drho/dx + dv/dy*drho/dy).
        BVF: 1D DataArray.
            The time, longitude and latitude averaged Brunt-Vassaila frequency array (hence is same size as
            rho 'depth' dimension).
        depth: float.
            The depth of interest.
        layer: int.
            The layer of interest. As dz is fixed to 10, the layer number 3 corresponds to a depth of 30m.
        n_levels: int, optional.
            The number of layers to use for the integration, above and down. If n_levels=3, the sub-domain
            will have a 'depth' dimension of size 3*2+1 = 7. A n_levels higher than 5 is highly recommanded.
        return_all: bool, optional.
            Whether to return all the sub-domain or just the layer in question.
        epsilon: float, optional.
            The percentage threshold to stop the solving iterations. Epsilon represents the percentage of
            change in W after a thousand iterations. This means that if epsilon is 1%, at current changing
            rate (which is strictly decreasing), the code will stop iterating when omega will only change by
            1% after a thousand iterations.
        max_loops: int, optional.
            The maximum number of loops.
        min_loops: int, optional.
            The minimum number of loops.
        f: float, optional.
            The Coriolis parameter. Default is f=8.57e-5.

        Returns
        -------
        W: 4D DataArray.
            The vertical velocities from the quasi-geostrophic omega equation integration.

        Notes
        -----
        This code has been developped by T. Hermilly in the frame of an internship at the IMEDEA
        research center in 2021. It can be mofied and used for non-lucrative purposes but must be
        properly cited. It is adapted to python 3.6 and requires the following packages: numpy,
        matplotlib, time, mpl_toolkits, warnings, gsw, xarray and scipy. The xarray DataArrays
        used have always the dimensions 'time', 'depth', 'longitude' and 'latitude' in that order
        and are therefore 4D DataArrays.
        
        Last updated: 2018-08-27
    """

    time = t.time()

    # Z discretization of 10m. Too high or too small dz changes to convergence (see finite differences loop)
    dz = 10.

    # Getting the index of the layer in question
    if layer is None:
        if depth is None:
            if not return_all:
                print('Give the algorithm a layer, a depth or choose return all!')
                return None
            else:
                ind = n_levels
        else:
            ind = int(np.round(depth / dz))
    else:
        ind = layer

    # Interpolating forcing and BVF with splines
    interp_levels = np.arange(0, Forcing['depth'].values[-1] + dz / 2., dz)
    mask = np.isnan(Forcing.interp(depth=interp_levels).values)
    spl_forc, spl_N = sci.interp1d(Forcing['depth'].values, np.nan_to_num(Forcing.values, nan=0), axis=1, kind='cubic',
                                   fill_value='extrapolate', assume_sorted=True), \
                      sci.interp1d(BVF['depth'].values, BVF.values, axis=0, kind='cubic',
                                   fill_value='extrapolate', assume_sorted=True)
    Forc, N = spl_forc(interp_levels), spl_N(interp_levels)

    # Masking forcing after interpolation
    Forc = np.where(mask, np.nan, Forc)

    nt, nz, ny, nx = np.shape(Forc)
    dx, dy = (Forcing['longitude'].values[1] - Forcing['longitude'].values[0]) * (2 * np.pi * 6.4e6 *
                                                                                  np.cos(36 * np.pi / 180) / 360), \
             (Forcing['latitude'].values[1] - Forcing['latitude'].values[0]) * (2 * np.pi * 6.4e6 / 360)

    # Getting the index limits of the sub-domain
    indmin = max(0, ind - n_levels)
    indmax = min(np.shape(Forc)[1] - 1, ind + n_levels)
    length = indmax - indmin + 1


    # Setting the boundary condition just outside to keep the first and last rows non zero. Restricting the
    # BVF and forcing arrays
    temp = np.nan * np.zeros((nt, length + 2, ny + 2, nx + 2))
    temp[:, 1:-1, 1:-1, 1:-1] = Forc[:, indmin:indmax + 1, :, :]
    Forc = temp
    N = N[indmin:indmax + 1]

    # If indmin is 0, the layer #1 is the surface, and so has no forcing
    if indmin == 0:
        Forc[:, 1] = np.nan * np.zeros(np.shape(Forc[:, 1]))

    # Initializing variables
    W = np.zeros(np.shape(Forc))
    max_diff = 1.
    max_ddiff = 1.
    count = 0

    # Keeping mask in memory for boundary condition
    mask = np.isnan(Forc)

    # Finding the ind position in the sub-domain frame
    if indmin == 0:
        if return_all:
            level = ind // 2
        else:
            level = ind
    else:
        level = n_levels

    # Starting to loop, count will stop between min_loops and max_loops
    print('Starting to update vertical velocity...')

    while ((max_ddiff * 1000 > epsilon / 100 * np.nanmax(np.abs(W[:, level + 1]))) or (count < min_loops)) and \
            (count < max_loops):

        # Keeping old variables
        count += 1
        old_W = W
        old_max_diff = max_diff

        if count > 1:
            print('Loading: iteration #{} | processing >{:.0f}%'.format(count, count / max_loops * 100), end='\r')

        def update(VV, BVfreq, forc, thickness, Nx, Ny, Dx, Dy, Dz, cor, ma):
            for j in range(1, thickness):

                n = BVfreq[j - 1]
                K = 1 / (n ** 2 / Dx ** 2 + n ** 2 / Dy ** 2 + cor ** 2 / Dz ** 2)

                for k in range(1, Ny + 1):

                    for l in range(1, Nx + 1):
                        VV[:, j, k, l] = -1. / 2. * K * (forc[:, j, k, l] -
                                                         n ** 2 * (VV[:, j, k, l - 1] + VV[:, j, k, l + 1]) / Dx ** 2 -
                                                         n ** 2 * (VV[:, j, k - 1, l] + VV[:, j, k + 1, l]) / Dy ** 2 -
                                                         cor ** 2 * (VV[:, j - 1, k, l] + VV[:, j + 1, k, l]) / Dz ** 2)
                        VV = np.where(ma, 0., VV)
            return VV

        W = update(W, N, Forc, length, nx, ny, dx, dy, dz, f, mask)

        if count == 1:

            if return_all:
                max_diff = np.nanmax(np.abs(old_W - W))
            else:
                max_diff = np.nanmax(np.abs(old_W[:, level + 1] - W[:, level + 1]))

            t1 = t.time() - time

            print('Min processing time: {}\n'
                  'Max processing time: {}'.format(secs_to_string(min_loops * t1),
                                                   secs_to_string(max_loops * t1)))

        else:

            if return_all:
                max_diff = np.nanmax(np.abs(old_W - W))
            else:
                max_diff = np.nanmax(np.abs(old_W[:, level + 1] - W[:, level + 1]))

            max_ddiff = old_max_diff - max_diff

            print('W change after 1K iteration: {:02.2f}%'.format(
                max_ddiff / np.nanmax(np.abs(W[:, level + 1])) * 100 * 1000))
            print('Loading: iteration #{} | {:.0f}% processed{}'.format(count, count / max_loops * 100,
                                                                        '.' * ((count % 3) + 1)), end='\r')

    if not return_all:

        W = W[:, level + 1, 1:-1, 1:-1] * 3600 * 24
        W = np.where(np.isnan(Forc[:, level + 1, 1:-1, 1:-1]), np.nan, W)[:, np.newaxis]

        dict = {'time': Forcing['time'].values,
                'depth': np.array([interp_levels[ind]]),
                'longitude': Forcing['longitude'].values,
                'latitude': Forcing['latitude'].values}

        W = xr.DataArray(W, dims=('time', 'depth', 'latitude', 'longitude'), coords=dict)

    else:

        W = W[:, 1:-1, 1:-1, 1:-1] * 3600 * 24
        W = np.where(np.isnan(Forc[:, 1:-1, 1:-1, 1:-1]), np.nan, W)

        dict = {'time': Forcing['time'].values,
                'depth': interp_levels[indmin:indmax + 1],
                'longitude': Forcing['longitude'].values,
                'latitude': Forcing['latitude'].values}

        W = xr.DataArray(W, dims=('time', 'depth', 'latitude', 'longitude'), coords=dict)

    print('Vertical velocity computation: {}\n'.format(secs_to_string(t.time() - time)))

    return W


def get_W_from_QGOE(Forcing, BVF, epsilon=1., max_loops=300, min_loops=20, f=8.57e-5):
    """
        Returns the vertical velocities DataArray of dimensions 'time', 'depth', 'longitude' and 'latitude'
        integrated from the quasi-geostrophic omega equation.

        Parameters
        ----------
        Forcing: 4D DataArray.
            The forcing DataArray derived as 2div(Q), with Q = (du/dx*drho/dx + dv/dx*drho/dy,
            du/dy*drho/dx + dv/dy*drho/dy).
        BVF: 1D DataArray.
            The time, longitude and latitude averaged Brunt-Vassaila frequency array (hence is same size as
            rho 'depth' dimension).
        epsilon: float, optional.
            The percentage threshold to stop the solving iterations. Epsilon represents the percentage of
            change in W after a thousand iterations. This means that if epsilon is 1%, at current changing
            rate (which is strictly decreasing), the code will stop iterating when omega will only change by
            1% after a thousand iterations.
        max_loops: int, optional.
            The maximum number of loops.
        min_loops: int, optional.
            The minimum number of loops.
        f: float, optional.
            The Coriolis parameter. Default is f=8.57e-5.

        Returns
        -------
        W: 4D DataArray.
            The vertical velocities from the quasi-geostrophic omega equation integration.

        Notes
        -----
        This code has been developped by T. Hermilly in the frame of an internship at the IMEDEA
        research center in 2021. It can be mofied and used for non-lucrative purposes but must be
        properly cited. It is adapted to python 3.6 and requires the following packages: numpy,
        matplotlib, time, mpl_toolkits, warnings, gsw, xarray and scipy. The xarray DataArrays
        used have always the dimensions 'time', 'depth', 'longitude' and 'latitude' in that order
        and are therefore 4D DataArrays.
        
        Last updated: 2018-08-27
    """

    dz = 10
    n_levels = int(Forcing['depth'].values[-1] / dz) + 1

    return get_W_atzlevel_from_QGOE(Forcing, BVF, n_levels=n_levels, return_all=True, epsilon=epsilon,
                                    max_loops=max_loops, min_loops=min_loops, f=f)

### Graphic functions

def draw_alboran_sea(ax=None, bounds=None, coast_res='i', grid=True, skipgrid=1.,
                     scale=True, scalesize=100, scalepos=None):
    """
        Plots the backgroud map on a given ax and returns the created basemap object
        to keep the conversion (lon, lat) -> (x, y) possible.

        Parameters
        ----------
        ax : matplotlib axes, optional.
            The axes on which to draw the map. If None, just uses the current axes.
        bounds: float tuple (lonmin, lonmax, latmin, latmax), optional.
            Custom map limits.
        coast_res: coast resolution. From low to high: 'l', 'i', 'h', 'f' (str).
        grid: choose whether to draw parallels and meridians or not (bool).
        skipgrid: float, optional.
            Default is 1., can be smaller. If one changes scale drastically, one can need
            to skip some parallels and meridians.
        scale: bool, optional.
            Wether to print a scale or not.
        scalesize: {int, float}, optional.
            Size of the scale in km. Useless when scale is False.
        scalepos: tuple (lon, lat), optional.
            The location of the scale on the map.

        Returns
        -------
        chart : basemap object
            The map. But "chart" is more posh. I like that.

        Notes
        -----
        This code has been developped by T. Hermilly in the frame of an internship at the IMEDEA
        research center in 2021. It can be mofied and used for non-lucrative purposes but must be
        properly cited. It is adapted to python 3.6 and requires the following packages: numpy,
        matplotlib, time, mpl_toolkits, warnings, gsw, xarray and scipy. The xarray DataArrays
        used have always the dimensions 'time', 'depth', 'longitude' and 'latitude' in that order
        and are therefore 4D DataArrays.
        
        Last updated: 2018-08-27
    """

    time = t.time()

    if ax is None:
        ax = plt.gca()

    ax.set_facecolor((0., 0.5, 0.9, 0.5))

    if bounds is None:
        bounds = (-4.5, 0.6, 34.5, 38.)

    chart = Basemap(llcrnrlon=bounds[0], llcrnrlat=bounds[2], urcrnrlon=bounds[1], urcrnrlat=bounds[3],
                    resolution=coast_res, projection='lcc', epsg=3395, ax=ax)
    chart.fillcontinents(ax=ax, zorder=2)
    chart.drawcoastlines(ax=ax, zorder=2)

    # Draw parallels and meridians

    if grid:
        meridians = np.arange(round(bounds[0])-1, round(bounds[1]) + 1, skipgrid)
        parallels = np.arange(round(bounds[2])-1, round(bounds[3]) + 1, skipgrid)
        chart.drawmeridians(meridians, labels=[0, 1, 1, 0], fontsize=10, ax=ax, zorder=2, color=(0., 0., 0., 0.5))
        chart.drawparallels(parallels, labels=[0, 1, 1, 0], fontsize=10, ax=ax, zorder=2, color=(0., 0., 0., 0.5))

    if scale:
        matplotlib.rcParams['lines.linestyle'] = '-'
        xsize = bounds[1] - bounds[0]
        ysize = bounds[3] - bounds[2]

        if scalepos is not None:
            chart.drawmapscale(scalepos[0], scalepos[1], 0., 0.,
                               scalesize, fontsize=6, ax=ax, zorder=3)
        else:
            chart.drawmapscale(bounds[0] + 0.78 * xsize, bounds[2] + 0.1 * ysize, 0., 0.,
                               scalesize, fontsize=6, ax=ax, zorder=3)

    ax.set_xlabel('Longitude (deg)', labelpad=5)
    ax.set_ylabel('Latitude (deg)', labelpad=5)

    print('Map drawing: {}'.format(secs_to_string(t.time() - time)))

    return chart


def draw_map_trajectory(lon, lat, ax=None, map=None, bounds=None,
                        color='b', trajlab=None, print_start=False):
    """
        Draws a trajectory on a given axis.

        Parameters
        ----------
        lon: 1D array-like.
            The longitude successive positions.
        lat: 1D array-like.
            The latitude successive positions.
        map: Basemap object, optional.
            The coordinate transform as a Basemap map.
        ax: matplotlib axes, optional.
            The axes on which to draw the trajectory. If None uses the current axes.
        map: Basemap object, optional.
            The coordinate transform as a Basemap map.
        bounds: float tuple (lonmin, lonmax, latmin, latmax), optional.
            Custom map limits.
        color: matplotlib color, optional.
            The trajectory color.
        trajlab: str, optional.
            The trajectory label.
        print_start: bool, optional.
            Whether to print the starting point as a red diamond or not.

        Notes
        -----
        This code has been developped by T. Hermilly in the frame of an internship at the IMEDEA
        research center in 2021. It can be mofied and used for non-lucrative purposes but must be
        properly cited. It is adapted to python 3.6 and requires the following packages: numpy,
        matplotlib, time, mpl_toolkits, warnings, gsw, xarray and scipy. The xarray DataArrays
        used have always the dimensions 'time', 'depth', 'longitude' and 'latitude' in that order
        and are therefore 4D DataArrays.
        
        Last updated: 2018-08-27
    """

    time = t.time()

    if ax is None:
        ax = plt.gca()

    if map is None:

        if bounds is None:
            default_bounds = (-4.5, 0.6, 34.5, 38.)
        else:
            default_bounds = bounds

        map = draw_alboran_sea(ax=ax, bounds=default_bounds, scale=False)

    ### Coordinates conversions and plot

    x, y = map(lon, lat)
    ax.plot(x, y, c=color, label=trajlab)

    if print_start:
        ax.scatter(x[0], y[0], s=20, marker='D', c='red')

    ### Legend

    if trajlab is not None:
        ax.legend()

    print('Trajectory drawing: {}'.format(secs_to_string(t.time() - time)))


def draw_map_scalar_field(lon, lat, C, bounds=None, values_bounds=None, ax=None, map=None, cmap='jet', bins=None,
                          print_cbar=True, cbar_label='', cbar_dir='horizontal', shading='flat', levels=None,
                          isolines=None, contour_label=False, contour_legend=None, centered=False, extend='both',
                          cbarticks=None):
    """
        Plots the given scalar field. Parameters lon, lat and C should have same dimensions.

        Parameters
        ----------
        lon: 2D array-like
            The longitude meshgrid positions.
        lat: 2D array-like
            The latitude meshgrid positions.
        C: 2D array-like
            The scalar field at lon, lat positions.
        ax: matplotlib axes, optional.
            The axes on which to draw the scalar field. If None uses the current axes.
        map: Basemap object, optional.
            The coordinate transform as a Basemap map.
        bounds: float tuple (lonmin, lonmax, latmin, latmax), optional.
            Custom drawing limits if map is None (else uses the map limits).
        values_bounds: tuple (min, max), optional.
            The scalar values limits.
        cmap: matplotlib cmap, optional.
            The colormap to use. Default is jet.
        bins: int, optional.
            The number of color category to use. If None, the maximum will be used.
        print_cbar: bool, optional.
            Whether to print the colorbar.
        cbar_label: str, optional.
            The colorbar label.
        cbar_dir: {'horizontal', 'vertical'}, optional.
            The colorbar orientation.
        cbarticks: 1D array-like, optional.
            The colorbar ticks to use if you are disatisfied with the default ones.
        shading: {'flat', 'gouraud'}, optional.
            'gouraud' makes it smooth.
        levels: 1D array-like, optional.
            The isolines levels to use. If None, no isolines will be drawn.
        isolines: 2D array-like, optional.
            The variable isolines to use. If None, uses the C array.
        contour_label: bool, optional.
            Whether to print the isoline label.
        contour_legend: str, optional.
            String to put in the legend regarding the isolines.
        centered: bool, optional.
            Centers the colormap on 0.
        extend: {'neither', 'both', 'min', 'max'}, optional.
            Which end of the colorbar to extend to a larger span.

        Notes
        -----
        This code has been developped by T. Hermilly in the frame of an internship at the IMEDEA
        research center in 2021. It can be mofied and used for non-lucrative purposes but must be
        properly cited. It is adapted to python 3.6 and requires the following packages: numpy,
        matplotlib, time, mpl_toolkits, warnings, gsw, xarray and scipy. The xarray DataArrays
        used have always the dimensions 'time', 'depth', 'longitude' and 'latitude' in that order
        and are therefore 4D DataArrays.
        
        Last updated: 2018-08-27
    """

    time = t.time()

    if ax is None:
        ax = plt.gca()

    if map is None:
        if bounds is None:
            bounds = (-4.5, 0.6, 34.5, 38.)

        map = Basemap(llcrnrlon=bounds[0], llcrnrlat=bounds[2],
                      urcrnrlon=bounds[1], urcrnrlat=bounds[3],
                      resolution='i', projection='lcc', epsg=3395, ax=ax)

    Xmap, Ymap = map(lon, lat)

    cmap = cm.get_cmap(cmap, bins)

    if values_bounds is None:

        if centered:
            max = np.nanmax(np.abs(C))
            values_bounds = (-max, max)
        else:
            values_bounds = (np.nanmin(C), np.nanmax(C))

        mappable = ax.pcolormesh(Xmap, Ymap, C, cmap=cmap, edgecolors='none',
                                 vmin=values_bounds[0], vmax=values_bounds[1], shading=shading)
    else:
        mappable = ax.pcolormesh(Xmap, Ymap, C, cmap=cmap, edgecolors='none',
                                 vmin=values_bounds[0], vmax=values_bounds[1], shading=shading)

    if levels is not None:

        if isolines is None:
            cs = ax.contour(Xmap, Ymap, C, levels=levels, colors='k', linewidths=0.5)

        else:
            cs = ax.contour(Xmap, Ymap, isolines, levels=levels, colors='k', linewidths=0.5)

        if contour_legend is not None:
            from matplotlib.lines import Line2D
            line = Line2D([0], [0], color='k', lw=0.5)
            ax.legend([line], [contour_legend], loc='upper left')

        if contour_label:
            ax = plt.gca()
            ax.clabel(cs, fmt='%1.2f', fontsize=6, inline_spacing=30)

    if print_cbar:

        fig = plt.gcf()

        if cbarticks is None:
            cbar = fig.colorbar(mappable, ax=ax, fraction=0.046, pad=0.15, orientation=cbar_dir, extend=extend)
        else:
            cbar = fig.colorbar(mappable, ax=ax, fraction=0.046, pad=0.15, orientation=cbar_dir, extend=extend,
                                ticks=cbarticks)

        cbar.set_label(cbar_label)

    print('Scalar field drawing: {}'.format(secs_to_string(t.time() - time)))


def quick_figure_scalar(xarr4d, layer=None, depth=None, day=0, ax=None, bounds=(-3.25, 0.25, 35., 37.), shading='flat',
                        values_bounds=None, title=None, cmap='jet', cbar_label=None, levels=None, contour_label=False,
                        centered=False):
    """
        The lazy (let's call it "automatic") way of ploting a scalar in a DataArray.

        Parameters
        ----------
        xarr4d: 4D DataArray.
            The DataArray of dimensions 'time', 'depth', 'longitude', 'latitude' containing the variable to plot.
        layer: int, optional.
            The z-layer to print. If None, depth param is used.
        depth: float, optional.
            The depth to print of layer is None. If None, surface layer is used.
        day: int, optional.
            The number of the day in the time dimension.
        ax: matplotlib axes, optional.
            The axes on which to draw the scalar field. If None uses the current axes.
        title: str, optional.
            The plot title.
        bounds: float tuple (lonmin, lonmax, latmin, latmax), optional.
            Custom drawing limits if map is None (else uses the map limits).
        values_bounds: tuple (min, max), optional.
            The scalar values limits.
        cmap: matplotlib cmap, optional.
            The colormap to use. Default is jet.
        cbar_label: str, optional.
            The colorbar label.
        shading: {'flat', 'gouraud'}, optional.
            'gouraud' makes it smooth.
        levels: 1D array-like, optional.
            The isolines levels to use. If None, no isolines will be drawn.
        contour_label: bool, optional.
            Whether to print the isoline label.
        centered: bool, optional.
            Centers the colormap on 0.

        Notes
        -----
        This code has been developped by T. Hermilly in the frame of an internship at the IMEDEA
        research center in 2021. It can be mofied and used for non-lucrative purposes but must be
        properly cited. It is adapted to python 3.6 and requires the following packages: numpy,
        matplotlib, time, mpl_toolkits, warnings, gsw, xarray and scipy. The xarray DataArrays
        used have always the dimensions 'time', 'depth', 'longitude' and 'latitude' in that order
        and are therefore 4D DataArrays.
        
        Last updated: 2018-08-27
    """

    if layer is None:
        if depth is None:
            layer = 0
        else:
            d = 0
            while xarr4d['depth'].values[d] < depth:
                d += 1
            layer = d

    lon, lat = np.meshgrid(xarr4d['longitude'].values, xarr4d['latitude'].values)
    F = xarr4d.values[day, layer]

    if ax is None:
        fig, ax = plt.subplots()
        fig.suptitle(title)
        chart = draw_alboran_sea(ax=ax, bounds=bounds, scalesize=50)
        draw_map_scalar_field(lon, lat, F, bounds=bounds, values_bounds=values_bounds, ax=ax, map=chart, cmap=cmap,
                              cbar_label=cbar_label, levels=levels, shading=shading, contour_label=contour_label,
                              centered=centered)
    else:
        ax.set_title(title)
        chart = draw_alboran_sea(ax=ax, bounds=bounds, scalesize=50)
        draw_map_scalar_field(lon, lat, F, bounds=bounds, values_bounds=values_bounds, ax=ax, map=chart, cmap=cmap,
                              cbar_label=cbar_label, levels=levels, shading=shading, contour_label=contour_label,
                              centered=centered)


def draw_map_vector_field(lon, lat, U, V, ax=None, map=None, skip=1, intensity_color=True,
                          bounds=(-3.25, 0.25, 35., 37.), shading='flat', bins=None, cmap='jet', print_cbar=True,
                          intensity_lim=None, cbar_label=None, cbar_dir='horizontal', color='k', zoom=1.):
    """
        Plots the given scalar field. Parameters lon, lat and C should have same dimensions.

        Parameters
        ----------
        lon: 2D array-like.
            The longitude meshgrid positions.
        lat: 2D array-like.
            The latitude meshgrid positions.
        U: 2D array-like.
            The scalar field at lon, lat positions corresponding to the zonal vector field component.
        V: 2D array-like.
            The scalar field at lon, lat positions corresponding to the meridional vector field component.
        ax: matplotlib axes, optional.
            The axes on which to draw the scalar field. If None uses the current axes.
        map: Basemap object, optional.
            The coordinate transform as a Basemap map.
        intensity_color: bool, optional.
            Whether to print a background scalar field corresponding to vector field intensity.
        skip: int, optional.
            If the resolution is too high, skip some vectors to make the plot understandable and faster.
        zoom: float, optional.
            The zoom factor for arrow size.
        bounds: float tuple (lonmin, lonmax, latmin, latmax), optional.
            Custom drawing limits if map is None (else uses the map limits).
        intensity_lim: float, optional.
            The vector max intensity value.
        cmap: matplotlib cmap, optional.
            The colormap to use. Default is jet.
        bins: int, optional.
            The number of color category to use. If None, the maximum will be used.
        print_cbar: bool, optional.
            Whether to print the colorbar.
        cbar_label: str, optional.
            The colorbar label.
        cbar_dir: {'horizontal', 'vertical'}, optional.
            The colorbar orientation.
        shading: {'flat', 'gouraud'}, optional.
            'gouraud' makes it smooth.
        color: matplotlib color, optional.
        The arrows color.

        Notes
        -----
        This code has been developped by T. Hermilly in the frame of an internship at the IMEDEA
        research center in 2021. It can be mofied and used for non-lucrative purposes but must be
        properly cited. It is adapted to python 3.6 and requires the following packages: numpy,
        matplotlib, time, mpl_toolkits, warnings, gsw, xarray and scipy. The xarray DataArrays
        used have always the dimensions 'time', 'depth', 'longitude' and 'latitude' in that order
        and are therefore 4D DataArrays.
        
        Last updated: 2018-08-27
    """

    time = t.time()

    if ax is None:
        ax = plt.gca()

    if map is None:
        map = Basemap(llcrnrlon=bounds[0], llcrnrlat=bounds[2],
                      urcrnrlon=bounds[1], urcrnrlat=bounds[3],
                      resolution='i', projection='lcc', epsg=3395, ax=ax)

    Xmap, Ymap = map(lon, lat)

    if intensity_color:

        C = np.sqrt(U ** 2 + V ** 2)
        max_intensity = np.nanmax(C)

        if intensity_lim is None:
            intensity_lim = (0., max_intensity)
        else:
            intensity_lim = (0., intensity_lim)

        draw_map_scalar_field(lon, lat, C, ax=ax, map=map, cmap=cmap, bins=bins, values_bounds=intensity_lim,
                              cbar_label=cbar_label, print_cbar=print_cbar, cbar_dir=cbar_dir, shading=shading,
                              extend='max')

    ax.quiver(Xmap[::skip, ::skip], Ymap[::skip, ::skip], U[::skip, ::skip], V[::skip, ::skip],
              scale=15. / zoom, width=0.002, headwidth=4, color=color)

    print('Vector field drawing: {}'.format(secs_to_string(t.time() - time)))


def draw_ts_diagram(SA, TH, depth=None, ax=None, background=True, color='k', cmap='winter_r', print_cbar=True,
                    ts_label='', print_legend=True, s=3, dsig=1.):
    """
        Plots a TS diagram.

        Parameters
        ----------
        SA: 1D array-like.
            The salinity array.
        TH: 1D array-like.
            The temperature array.
        depth: 1D array-like, optional.
            The depth array that, if given, colors the points according to the cmap.
        ax: matplotlib axes, optional.
            The axes on which to draw the TS diagram.
        background: bool, optional.
            Whether to print the density isolines or not.
        color: matplotlib color, optional.
            The points color.
        cmap: matplotlib colormap, optional.
            The color map if depth is not None.
        print_cbar: bool, optional.
            Whether to print the colorbar.
        ts_label: str, optional.
            The scatter plot label.
        print_legend: bool, optional.
            Whether to print the legend.
        s: float, optional.
            The points size.
        dsig: float, optional.
            The sigma isolines spacings.

        Notes
        -----
        This code has been developped by T. Hermilly in the frame of an internship at the IMEDEA
        research center in 2021. It can be mofied and used for non-lucrative purposes but must be
        properly cited. It is adapted to python 3.6 and requires the following packages: numpy,
        matplotlib, time, mpl_toolkits, warnings, gsw, xarray and scipy. The xarray DataArrays
        used have always the dimensions 'time', 'depth', 'longitude' and 'latitude' in that order
        and are therefore 4D DataArrays.
        
        Last updated: 2018-08-27
    """

    if ax is None:
        ax = plt.gca()

    ax.set_xlabel('Salinity (ppt)')
    ax.set_ylabel('Temperature (°C)')

    if depth is None:
        ax.scatter(SA, TH, s=s, c=color, label=ts_label)
        if print_legend:
            plt.legend()
    else:
        plt.scatter(SA, TH, s=s, c=depth, cmap=cmap, norm=matplotlib.colors.LogNorm())

        if print_cbar:
            cbar = plt.colorbar(fraction=0.046, pad=0.15)
            cbar.ax.invert_yaxis()
            cbar.set_label('Depth (m)')

    smin, smax = ax.get_xlim()
    tmin, tmax = ax.get_ylim()

    if background:

        X, Y = np.meshgrid(np.linspace(smin, smax, 100), np.linspace(tmin, tmax, 100))
        sigma = sw.density.sigma0(X, Y)
        levels = np.arange(18, 35, dsig)

        CS = ax.contour(X, Y, sigma, colors='k', levels=levels, linewidths=0.5, zorder=1)
        CS.collections[0].set_label('$\sigma_0$ (kg.m-3)')
        plt.legend()
        ax.clabel(CS, levels=levels, inline=True, inline_spacing=30, fmt='%1.1f')


def draw_map_ts_diagram(SA, TH, coords, Xarr_2D=None, bounds=(-3.25, 0.25, 35., 37.), tolerance=15,
                        title='TS diagrams at specific locations', map_title='Chosen locations',
                        background=True, map_cmap='jet', bins=None, map_cbar_label=None, colors=None, s=2, dsig=1.):
    """
        Plots a TS diagram.

        Parameters
        ----------
        SA: 4D DataArray.
            The salinity DataArray of dimensions 'time', 'depth', 'longitude', 'latitude'.
        TH: 4D DataArray.
            The temperature DataArray of dimensions 'time', 'depth', 'longitude', 'latitude'.
        coords: N*2 array-like.
            The locations list (lon, lat).
        Xarr_2D: 2D DataArray, optional.
            The DataArray containing the variable to plot on the map (coords should be only 'longitude'
            and 'latitude').
        bounds: float tuple (lonmin, lonmax, latmin, latmax), optional.
            Custom drawing limits if map is None (else uses the map limits).
        tolerance: float, optional.
            The square side size of the horizontal area of each profile (centered and coords locations).
        title: str, optional.
            The figure title. Default is 'TS diagrams at specific locations'.
        map_title: str, optional.
            The map title. Default is 'Chosen locations'.
        map_cmap: matplotlib colormap, optional.
            The map colormap. Default is 'jet'.
        bins: int, optional.
            The number of color category to use. If None, the maximum will be used.
        map_cbar_label: str, optional.
            The map colorbar label.
        background: bool, optional.
            Whether to print the density isolines or not.
        colors: matplotlib color list, optional.
            The points color list in same order as the locations.
        s: float, optional.
            The points size.
        dsig: float, optional.
            The sigma isolines spacings.

        Notes
        -----
        This code has been developped by T. Hermilly in the frame of an internship at the IMEDEA
        research center in 2021. It can be mofied and used for non-lucrative purposes but must be
        properly cited. It is adapted to python 3.6 and requires the following packages: numpy,
        matplotlib, time, mpl_toolkits, warnings, gsw, xarray and scipy. The xarray DataArrays
        used have always the dimensions 'time', 'depth', 'longitude' and 'latitude' in that order
        and are therefore 4D DataArrays.
        
        Last updated: 2018-08-27
    """

    from matplotlib.patches import Circle

    N = np.shape(coords)[0]

    tolerance *= 1e3

    tol_lon, tol_lat = tolerance / (2 * np.pi * 6.4e6 * np.cos(36 * np.pi / 180) / 360), \
                       tolerance / (2 * np.pi * 6.4e6 / 360)

    lon, lat = np.meshgrid(SA['longitude'], SA['latitude'])

    SA_list, TH_list = [], []
    for k in range(N):
        SA_list.append(SA.where((SA['longitude'] > coords[k, 0] - tol_lon) &
                                (SA['longitude'] < coords[k, 0] + tol_lon) &
                                (SA['latitude'] > coords[k, 1] - tol_lat) &
                                (SA['latitude'] < coords[k, 1] + tol_lon), drop=True))
        TH_list.append(TH.where((TH['longitude'] > coords[k, 0] - tol_lon) &
                                (TH['longitude'] < coords[k, 0] + tol_lon) &
                                (TH['latitude'] > coords[k, 1] - tol_lat) &
                                (TH['latitude'] < coords[k, 1] + tol_lon), drop=True))

    if colors is None:
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    fig, axs = plt.subplots(1, 2)
    fig.suptitle(title, y=0.96)
    axs[0].set_title(map_title, pad=20)
    chart = draw_alboran_sea(ax=axs[0], bounds=bounds)
    x_p, y_p = chart(coords[:, 0], coords[:, 1])
    for k in range(N):
        axs[0].scatter(x_p[k], y_p[k], marker='v', c=colors[k], s=40, edgecolor='black', zorder=3,
                       label='location n°{}'.format(k + 1))
        axs[0].add_patch(Circle((x_p[k], y_p[k]), tolerance, fill=False, color='r', ls='--'))

    axs[0].legend()

    if Xarr_2D is not None:
        draw_map_scalar_field(lon, lat, Xarr_2D.values.squeeze(), ax=axs[0], map=chart, print_cbar=True,
                              cbar_label=map_cbar_label, cmap=map_cmap, bins=bins)

    axs[1].set_title('Associated TS diagram')
    for k in range(N):
        if k == 0:
            draw_ts_diagram(SA_list[k], TH_list[k], ax=axs[1], color=colors[k], ts_label='location n°{}'.format(k + 1),
                            background=False, s=s, dsig=dsig)
        elif k == N-1:
            draw_ts_diagram(SA_list[k], TH_list[k], ax=axs[1], color=colors[k], ts_label='location n°{}'.format(k + 1),
                            background=background, s=s, dsig=dsig)
        else:
            draw_ts_diagram(SA_list[k], TH_list[k], ax=axs[1], color=colors[k], ts_label='location n°{}'.format(k + 1),
                            background=False, s=s, dsig=dsig)


def draw_vsect(vsect, isolines_vsect=None, ax=None, cmap='jet', bins=None, print_cbar=True, cbar_label=None,
               title=None, print_contour=False, levels=4, contour_label=True, contour_legend=None, shading='flat',
               values_bounds=None):
    """
        Plots a vertical section. See vsect_interp function for building the vertical section 2D DataArray.

        Parameters
        ----------
        vsect: 2D DataArray.
            The vertical section as a DataArray of dimensions 'lon' or 'lat' and 'depth'.
        isolines_vsect: 2D DataArray, optional.
            The vertical DataArray section of dimensions 'lon' or 'lat' and 'depth' of the isolines variable.
        ax: matplotlib axes, optional.
            The axes on which to draw the vertical section.
        cmap: matplotlib cmap, optional.
            The vertical section colormap.
        bins: int, optional.
            The number of color category to use. If None, the maximum will be used.
        print_cbar: bool, optional.
            Whether to print the colorbar.
        cbar_label: str, optional.
            The colorbar label.
        title: str, optional.
            The figure title.
        print_contour: bool, optional.
            Whether to print the contour lines.
        contour_legend: str, optional.
            The char describing the contours in the legend.
        levels: {int, 1D array-like}, optional.
            The levels to use for the density contour. If an int, will draw the given number of levels.
        contour_label: str, optional.
            The contour label to print in the legend.
        shading: {'flat', 'gouraud'}, optional.
            'gouraud' makes it smooth.
        values_bounds: iterable, optional.
            Defines the printed values limits.

        Notes
        -----
        This code has been developped by T. Hermilly in the frame of an internship at the IMEDEA
        research center in 2021. It can be mofied and used for non-lucrative purposes but must be
        properly cited. It is adapted to python 3.6 and requires the following packages: numpy,
        matplotlib, time, mpl_toolkits, warnings, gsw, xarray and scipy. The xarray DataArrays
        used have always the dimensions 'time', 'depth', 'longitude' and 'latitude' in that order
        and are therefore 4D DataArrays.
        
        Last updated: 2018-08-27
    """

    if ax is None:
        ax = plt.gca()

    ax.set_facecolor((0.25, 0.10, 0.25))
    ax.set_title(title, pad=13)

    ax.set_ylabel('Depth (m)')
    ax.invert_yaxis()

    if vsect.dims[1] == 'lat':

        X, Y = np.meshgrid(vsect['lat'].values, vsect['depth'].values)
        ax.set_xlabel('Latitude (deg)', labelpad=5)
    else:

        X, Y = np.meshgrid(vsect['lon'].values, vsect['depth'].values)
        ax.set_xlabel('Longitude (deg)', labelpad=5)

    cmap = cm.get_cmap(cmap, bins)
    pi_width, pi_height = np.sort(np.unique(X.flatten()))[1] - np.sort(np.unique(X.flatten()))[0], \
                          np.sort(np.unique(Y.flatten()))[1] - np.sort(np.unique(Y.flatten()))[0]
    if values_bounds is None:
        mappable = ax.pcolormesh(X - pi_width / 2, Y - pi_height / 2, vsect.values, cmap=cmap, shading=shading)
    else:
        mappable = ax.pcolormesh(X - pi_width / 2, Y - pi_height / 2, vsect.values, cmap=cmap, shading=shading,
                                 vmin=values_bounds[0], vmax=values_bounds[1])

    if print_cbar:
        cbar = plt.colorbar(mappable, extend='both', orientation='horizontal', fraction=0.046, pad=0.20)
        cbar.set_label(cbar_label)

    if print_contour:

        if contour_legend is not None:
            from matplotlib.lines import Line2D
            line = Line2D([0], [0], color='k', lw=1)
            ax.legend([line], [contour_legend], loc='lower right')

        if isolines_vsect is not None:
            CS = ax.contour(X, Y, isolines_vsect.values, colors='k', linewidths=1, zorder=1, levels=levels)

        else:
            CS = ax.contour(X, Y, vsect.values, colors='k', linewidths=1, zorder=1, levels=levels)

        if contour_label:
            ax.clabel(CS, inline=True, fmt='%1.1f', inline_spacing=20, fontsize=8)


def draw_map_vsect(vsect, isolines_vsect=None, Xarr_2D=None, title='Vertical section', values_bounds=None,
                   vsect_title='Associated vertical section', bounds=None, shading='flat', scalesize=100,
                   map_title='Vertical section location', color='k', cmap='jet', bins=None, contour_legend=None,
                   vsect_cbar_label=None, map_cbar_label=None, print_contour=True, levels=4, contour_label=True,
                   skipgrid=1.):
    """
        Plots a vertical section along to a map of the location. See vsect_interp function
        for building the vertical section 2D DataArray.

        Parameters
        ----------
        vsect: 2D DataArray.
            The vertical section as a DataArray of dimensions 'lon' or 'lat' and 'depth'.
        Xarr_2D: 2D DataArray, optional.
            The DataArray containing the variable to plot on the map (coords should be only 'longitude'
            and 'latitude').
        isolines_vsect: 2D DataArray, optional.
            The vertical DataArray section of the isolines variable.
        title: str, optional.
            The figure title. Default is 'Vertical section'.
        vsect_title: str, optional.
            The vertical section title. Default is 'Associated vertical section'.
        bounds: float tuple (lonmin, lonmax, latmin, latmax), optional.
            Custom drawing limits if map is None (else uses the map limits).
        cmap: matplotlib cmap, optional.
            The vertical section colormap.
        bins: int, optional.
            The number of color category to use. If None, the maximum will be used.
        map_title: str, optional.
            The map title. Default is 'Vertical section location'.
        color: matlotlib color, optional.
            The color of the vertical section line on the map.
        vsect_cbar_label: str, optional.
            The vertical section colorbar label.
        map_cbar_label: str, optional.
            The map scalar field colorbar label.
        print_contour: bool, optional.
            Whether to print the contour lines.
        levels: {int, 1D array-like}, optional.
            The levels to use for the density contour. If an int, will draw the given number of levels.
        contour_label: bool, optional.
            Whether to label the contours.
        contour_legend: str, optional.
            String to put in the legend regarding the isolines.
        shading: {'flat', 'gouraud'}, optional.
            'gouraud' makes it smooth.
        scalesize: {int, float}, optional.
            Size of the scale in km. Useless when scale is False.
        skipgrid: float, optional.
            Default is 1., can be smaller. If one changes scale drastically, one can need
            to skip some parallels and meridians.
        values_bounds: iterable, optional.
            Defines the printed values limits.

        Notes
        -----
        This code has been developped by T. Hermilly in the frame of an internship at the IMEDEA
        research center in 2021. It can be mofied and used for non-lucrative purposes but must be
        properly cited. It is adapted to python 3.6 and requires the following packages: numpy,
        matplotlib, time, mpl_toolkits, warnings, gsw, xarray and scipy. The xarray DataArrays
        used have always the dimensions 'time', 'depth', 'longitude' and 'latitude' in that order
        and are therefore 4D DataArrays.
        
        Last updated: 2018-08-27
    """

    fig, axs = plt.subplots(1, 2, figsize=(12, 7))
    fig.suptitle(title, y=0.94)
    axs[0].set_title(map_title, pad=20)
    chart = draw_alboran_sea(ax=axs[0], bounds=bounds, skipgrid=skipgrid, scalesize=scalesize)

    A, B = vsect.attrs['start'], vsect.attrs['stop']
    Xmap, Ymap = chart([A[0], B[0]], [A[1], B[1]])
    axs[0].plot(Xmap, Ymap, c=color)

    if Xarr_2D is not None:
        lon = Xarr_2D['longitude'].values
        lat = Xarr_2D['latitude'].values
        lon, lat = np.meshgrid(lon, lat)

        scalar_field = Xarr_2D.values.squeeze()

        draw_map_scalar_field(lon, lat, scalar_field, ax=axs[0], bounds=bounds, cbar_label=map_cbar_label, bins=bins,
                              cmap=cmap, shading=shading, values_bounds=values_bounds)

    draw_vsect(vsect, cmap=cmap, isolines_vsect=isolines_vsect, shading=shading, values_bounds=values_bounds,
               cbar_label=vsect_cbar_label, ax=axs[1], title=vsect_title, print_contour=print_contour,
               levels=levels, bins=bins, contour_label=contour_label, contour_legend=contour_legend)


def corr_plot(arr1, arr2, c='b', ax=None, xlabel=None, ylabel=None):
    """
        Plots a correlation diagram.

        Parameters
        ----------
        arr1: array-like.
            The first array.
        arr2: array-like.
            The second array of same shape as arr1 to compare.
        c: matplotlib color, optional.
            The color of the scatter plot. Default is blue.
        ax: matplotlib axes, optional.
            The axes on which to draw the correlation diagram. If None, current axes are used.
        xlabel: str, optional.
            The x-axis label.
        ylabel: str, optional.
            The y-axis label.

        Notes
        -----
        This code has been developped by T. Hermilly in the frame of an internship at the IMEDEA
        research center in 2021. It can be mofied and used for non-lucrative purposes but must be
        properly cited. It is adapted to python 3.6 and requires the following packages: numpy,
        matplotlib, time, mpl_toolkits, warnings, gsw, xarray and scipy. The xarray DataArrays
        used have always the dimensions 'time', 'depth', 'longitude' and 'latitude' in that order
        and are therefore 4D DataArrays.
        
        Last updated: 2018-08-27
    """

    from scipy.stats import linregress

    a1, a2 = arr1.flatten(), arr2.flatten()
    xy = np.vstack([a1, a2])
    xy = xy[:, ~np.isnan(xy).any(axis=0)]
    a1, a2 = xy[0], xy[1]

    sl, inter, r, p, stderr = linregress(a1, a2)

    if ax is None:
        ax = plt.gca()

    ax.set_title('r-val: {:0.3f} / p-val: {:0.3f} / stdev: {:0.3f}'.format(r, p, stderr))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.scatter(a1, a2, c=c, s=2)
    xmin, xmax = np.nanmin(a1), np.nanmax(a1)
    ax.plot([xmin, xmax], [inter + sl * xmin, inter + sl * xmax], label='lstsq: y={:.2f}x+{:.2f}'.format(sl, inter))
    ax.plot([xmin, xmax], [xmin, xmax], c='k', label='Identity', linestyle='dashed')
    ax.set_xlim(1.3 * xmin, 1.3 * xmax)
    ax.set_ylim(1.3 * xmin, 1.3 * xmax)
    ax.set_aspect(1.)
    ax.legend()


if __name__ == '__main__':

    ### Do your tests here

    ### Opening data

    folder = 'data/'
    fn = '20180531to20180602_CMEMS_daily_masked.nc'

    data = xr.open_dataset(folder + fn)

    ### Tests

    ### Test 1

    # field = data['thetao'].values[0, 0]
    # lon, lat = np.meshgrid(data['longitude'], data['latitude'].values)
    # fig, ax = plt.subplots()
    # chart = draw_alboran_sea(ax=ax, bounds=(-4., 0., 34.5, 37.5))
    # draw_map_scalar_field(lon, lat, field, map=chart, bins=10, cbar_label='Temperature', print_cbar=False)

    ### Test 2

    # U, V = data['uo'].values[0, 0], data['vo'].values[0, 0]
    # lon, lat = np.meshgrid(data['longitude'], data['latitude'].values)
    # fig, ax = plt.subplots()
    # chart = draw_alboran_sea(ax=ax, bounds=(-4., 0., 34.5, 37.5))
    # draw_map_vector_field(lon, lat, U, V, map=chart, skip=3, bins=30, intensity_color=True,
    #                       cbar_label='Temperature')

    ### Test 3

    # sigma = (sw.density.sigma0(data['so'], data['thetao'])).where(data['so'].depth < 550, drop=True) + 1000
    # dh = get_DH_from_sigma(sigma, 300, 1026., 9.81)
    # lon, lat = np.meshgrid(dh['longitude'].values, dh['latitude'].values)
    # _, ax = plt.subplots()
    # chart = draw_alboran_sea(ax=ax, bounds=(-4., 0., 34.5, 37.5))
    # draw_map_scalar_field(lon, lat, dh.values[0, 0], ax=ax, map=chart)

    ### Test 4

    # A = np.array([[np.nan, np.nan, np.nan, np.nan, np.nan],
    #               [np.nan, 1, 1, np.nan, 4],
    #               [1, 1, 2, 3, 2],
    #               [1, 2, 3, 3, 2],
    #               [1, 2, 2, 3, 3],
    #               [1, 1, 2, 3, 3],
    #               [np.nan, 2, 12, 2, np.nan]]).reshape((1, 1, 7, 5))
    #
    # B = np.concatenate([A, A, A, A], axis=1)
    # C = np.concatenate([B, B, B], axis=0)
    #
    # D = filly(C)

    ### Test 5

    # XA = data['so'].where((data['depth'] <= 320) & (data['time'] == data['time'].values[0]), drop=True)
    # XA_2D = data['so'].where((data['depth'] == data['depth'].values[0]) &
    #                          (data['time'] == data['time'].values[0]), drop=True)
    #
    # sigma = sw.density.sigma0(data['so'].where((data['depth'] <= 320) &
    #                                            (data['time'] == data['time'].values[0]), drop=True),
    #                           data['thetao'].where((data['depth'] <= 320) &
    #                                                (data['time'] == data['time'].values[0]), drop=True))
    #
    # vsect = vsect_interp(XA, dim='lon', A_B_depth=[(-1.75, 35.7), (-0.9, 36.2), (0., 300.)], nz=100)
    # density_vsect = vsect_interp(sigma, dim='lon', A_B_depth=[(-1.75, 35.7), (-0.9, 36.2), (0., 300.)], nz=100)
    # draw_map_vsect(vsect, Xarr_2D=XA_2D,
    #                isolines_vsect=density_vsect,
    #                title='Vertical section of salinity', levels=3, map_cbar_label='Salinity (ppt)',
    #                contour_label='Density contour', vsect_cbar_label='Salinity (ppt)')

    ### Test 6

    # SA, TH = data['so'].where(data['time'] == data['time'].values[2], drop=True),\
    #          data['thetao'].where(data['time'] == data['time'].values[2], drop=True)
    #
    # coords = np.array([[-2.5, 36.5],
    #                    [-1.9, 36.],
    #                    [-1., 36.6]])
    #
    # TH_2D = TH.where(TH['depth'] == TH['depth'].values[0], drop=True)
    # colors=['purple', 'green', 'orange']
    # draw_map_ts_diagram(SA, TH, coords)

    ### Test 7

    # T = data['thetao']
    #
    # quick_figure_scalar(T, depth=200, title='Temperature at {}m depth'.format(200), cbar_label='Temperature (°C)',
    #                     levels=np.arange(13, 14, 0.1), contour_label=True, shading='gouraud')