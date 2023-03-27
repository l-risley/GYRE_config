import xarray
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cartopy.crs as ccrs
import numpy as np

def set_cbar_range(variable):
    ''' Set the range of the colour bar
    '''
    if variable == "sossheig":
        vmin = -0.4 ; vmax = 0.4
    elif variable in ["votemper", "sosstsst"]:
        vmin = 15. ; vmax = 25.
    elif variable in ["vosaline", "sossssss"]:
        vmin = 36. ; vmax = 37.
    elif variable in ["vozocrtx", "vomecrty", "sozocrtx", "somecrty"]:
        vmin = -0.5 ; vmax = 0.5
    elif variable in ["sozotaux", "sometauy"]:
        vmin = -0.1 ; vmax = 0.1
    else:
        raise ValueError("Variable name not recognised")

    return vmin, vmax

def plot_field(da, da2=None, output_dir=".", variable=None):
    '''
    Plot xarray data array
    '''
    datetime = da.time_counter.dt.strftime("%Y%m%dT%H%M").values
    print(datetime)

    vmin, vmax = set_cbar_range(da.name)
    cmap = cm.get_cmap('RdBu_r')

    if da2 is not None:
        plt_fld = np.sqrt((da.values*da.values) + (da2.values*da2.values))
    else:
        plt_fld = da

    fig = plt.figure(figsize=[8,8])
    ax = plt.subplot(projection=ccrs.PlateCarree())
    im = ax.pcolormesh(da.nav_lon.values, da.nav_lat.values, plt_fld,
                  vmin=vmin, vmax=vmax, cmap=cmap, transform=ccrs.PlateCarree())
    ax.gridlines(draw_labels=True)
    fig.colorbar(im, orientation='horizontal', pad=0.05, extend="both")

    if variable is not None:
        var_name = variable
    else:
        var_name = da.name
    ax.set_title(f"{var_name} ({da.units}): {datetime}\n")
    plt.tight_layout()

    out_fname = f"{output_dir}/{var_name}_{datetime}.png"
    plt.savefig(out_fname)

if __name__ == "__main__":

    #variable = "velocity"
    #variable = "sossheig"
    #variable = "votemper"
    variable = "vosaline"

    config = "gyre12"

    input_dir = f"/hpc/scratch/d00/frwn/cylc-run/u-cu190/share/cycle/20190101T0000Z/results_{config}"
    output_dir = f"/data/users/frwn/GYRE/plots/{config}"

    if variable in ["sozocrtx", "sozotaux"]:
        input_file = f"{input_dir}/mersea.grid_U.nc"
        ds = xarray.open_mfdataset(input_file)
        das = ds[variable].isel(depthu=0)
    elif variable in ["somecrty", "sometauy"]:
        input_file = f"{input_dir}/mersea.grid_V.nc"
        ds = xarray.open_mfdataset(input_file)
        das = ds[variable].isel(depthv=0)
    elif variable in ["velocity"]:
        input_file1 = f"{input_dir}/mersea.grid_U.nc"
        input_file2 = f"{input_dir}/mersea.grid_V.nc"
        ds1 = xarray.open_mfdataset(input_file1)
        ds2 = xarray.open_mfdataset(input_file2)
        das1 = ds1["vozocrtx"].isel(depthu=0)
        das2 = ds2["vomecrty"].isel(depthv=0)
    else:
        input_file = f"{input_dir}/mersea.grid_T.nc"
        ds = xarray.open_mfdataset(input_file)
        das = ds[variable].isel(deptht=0)

    if variable == "velocity":
        for da1, da2 in zip(das1, das2):
            plot_field(da1, da2=da2, output_dir=output_dir, variable=variable)
    else:
        ds = xarray.open_mfdataset(input_file)
        for da in das:
            plot_field(da, output_dir=output_dir)

