"""
Geostrophic balance : If Rossby number is small, acceleration term can be neglected relative to Coriolis force in
the horizontal momentum equations. If forcing term can also be neglected, valid away from boundaries,
the horizontal momentum equations reduce to geostrophic balance:
u = -1/p0*f dp/dy
v = 1/p0*f dp/dy

Near the surface of the ocean, the shallow water pressure is related hydrostatically to elevation : P = gp0n
relative to equilibrium geopotential z=0. The near surface geo-strophic flow can be calculated from
u_surface = -g/f dndy
v_surface = g/f dndx
https://uw.pressbooks.pub/ocean285/chapter/geostrophic-balance/
@author : Laura Risley, 2022
"""

import numpy as np
from general_functions import *
from gyre_setup import *


def geostrophic_balance_D(eta: np.ndarray, vel: str, lat, dz):
    """
    Use geostrophic balance to find the horizontal velocity vectors.
    Inputs: eta, elevation
            vel, either 'u' or 'v'
            lat, lattitude coordinates
            dz, spatial grid length (either dy or dx)
    Output: u/v, the horizontal velocity vector
    """
    f_0, g, beta = f0_calc(), param['g'], beta_calc()
    if vel == 'u':
        dy = dz
        # interpolate eta to v grid
        eta_vgrid = interp_merid(eta)
        # eta = 0 on the v boundary and outside the u boundary #TODO: need to change this assumption!?
        eta_vgrid = outside_boundary(eta_vgrid, 'nesw')
        # find dndy on the eta grid
        dndy = dzdy(eta_vgrid, dy)
        # interp to the u grid
        dndy_ugrid = interp_zonal(dndy)
        cor = f_0 + beta * lat
        return - g * np.divide(dndy_ugrid, cor)

    elif vel == 'v':
        dx = dz
        # interpolate eta to u grid
        eta_ugrid = interp_zonal(eta)
        # eta = 0 on the u boundary and outside the v boundary
        eta_ugrid = outside_boundary(eta_ugrid, 'nesw')
        # find dndx on the eta grid
        dndx = dzdx(eta_ugrid, dx)
        # interp to the u grid
        dndx_vgrid = interp_merid(dndx)
        # calculate the coriolis term on v-grid
        cor = f_0 + beta * lat
        return g * np.divide(dndx_vgrid, cor)


def geostrophic_balance(eta: np.ndarray, vel: str, lat, dz):
    """
    Use geostrophic balance to find the horizontal velocity vectors.
    Inputs: eta, elevation
            vel, either 'u' or 'v'
            lat, lattitude coordinates
            dz, spatial grid length (either dy or dx)
    Output: u/v, the horizontal velocity vector
    """
    print('Calculating geostrophic balance.')
    f_0, g, beta = f0_calc(), param['g'], beta_calc()
    if vel == 'u':
        dy = dz
        # find dndy on the v grid
        dndy = dzdy(eta, dy)
        ## interp to the u grid using four point averaging
        # u will have the same shape as eta, with zeros on the boundary
        dndy_ugrid = np.zeros_like(eta)
        dndy_ugrid[1:-1, :-1] = interp_merid(interp_zonal(dndy))
        ## calculate the coriolis term on the u grid
        # f = f0 + by where y is the distance between the most southern point (where f) is calculated).
        cor = f_0 + beta * ((lat - param['phi0']) * param['rad'])
        return - g * np.divide(dndy_ugrid, cor)

    elif vel == 'v':
        dx = dz
        # find dndx on the u grid
        dndx = dzdx(eta, dx)
        ## interp to the v grid using four point averaging
        # v will have the same shape as eta, with zeros on the boundary
        dndx_vgrid = np.zeros_like(eta)
        dndx_vgrid[:-1, 1:-1] = interp_zonal(interp_merid(dndx))
        ## calculate the coriolis term on the vgrid
        # f = f0 + by where y is the distance between the most southern point (where f) is calculated).
        cor = f_0 + beta * ((lat - param['phi0']) * param['rad'])
        return g * np.divide(dndx_vgrid, cor)
