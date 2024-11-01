import numpy as np
import xarray
import matplotlib.pyplot as plt


def read_fields(in_files, variable):
    ds = xarray.open_mfdataset(in_files)
    fields = ds[variable]

    return fields


def calc_rmse(fields1, fields2):
    sq_errors = (fields1 - fields2) ** 2

    if fields1.name == "sossheig":
        mse = np.mean(sq_errors, axis=(1, 2))
    else:
        mse = np.mean(sq_errors, axis=(2, 3))

    return np.sqrt(mse)


def calc_spatial_mean(field):
    abs_field = abs(field)

    return np.mean(abs_field, axis=(2, 3))


def plot_spatial_mean(mean_list, expt_ids, plot_dir, month):
    plot_levels = [4, 8, 20]

    # Plot time-series of the errors at selected levels
    for plot_level in plot_levels:

        fig = plt.figure(figsize=[10, 6])

        for i, mean in enumerate(mean_list):
            plot_vals = mean["vovecrtz"].isel(depthw=plot_level)
            plt.plot(mean.time_counter, plot_vals, label=expt_ids[i])

        plt.ylabel("Spatial mean of the absolute values")
        plt.legend()
        plt.title(f"Vertical velocity at {plot_vals.depthw.values:.2f}m depth")

        plt.savefig(f"{plot_dir}/spatial_mean_vovecrtz_{plot_level}_{month}.png")


def plot_rmse(rmse_list, variable, expt_ids, plot_dir, month):
    if variable == "sossheig":
        plot_levels = [0]
        variable_name = "Sea-surface Height"
    else:
        plot_levels = [0, 8, 20]

    # Plot time-series of the errors at selected levels
    for plot_level in plot_levels:

        fig = plt.figure(figsize=[10, 6])

        for i, rmse in enumerate(rmse_list):

            if variable == "votemper":
                plot_vals = rmse[variable].isel(deptht=plot_level)
                variable_name = "Temperature"
            elif variable == "vosaline":
                plot_vals = rmse[variable].isel(deptht=plot_level)
                variable_name = "Salinty"
            elif variable == "vozocrtx":
                plot_vals = rmse[variable].isel(depthu=plot_level)
                variable_name = "Zonal Velocity"
            elif variable == "vomecrty":
                plot_vals = rmse[variable].isel(depthv=plot_level)
                variable_name = "Meridional Velocity"
            elif variable == "vovecrtz":
                plot_vals = rmse[variable].isel(depthw=plot_level)
                variable_name = "Vertical Velocity"
            else:
                plot_vals = rmse[variable]

            plt.plot(rmse.time_counter, plot_vals, label=expt_ids[i])

        plt.ylabel("RMSE")
        plt.legend()
        # try:
        # plt.title(f"{variable_name} at {plot_vals.deptht.values:.2f}m depth")
        # except:
        # plt.title(f"{variable_name}")

        if variable in ["votemper", "vosaline"]:
            plt.title(f"{variable_name} at {plot_vals.deptht.values:.2f}m depth")
        elif variable == "vozocrtx":
            plt.title(f"{variable_name} at {plot_vals.depthu.values:.2f}m depth")
        elif variable == "vomecrty":
            plt.title(f"{variable_name} at {plot_vals.depthv.values:.2f}m depth")
        elif variable == "vovecrtz":
            plt.title(f"{variable_name} at {plot_vals.depthw.values:.2f}m depth")
        else:
            plt.title(f"{variable_name}")
        plt.savefig(f"{plot_dir}/rmse_{variable}_{plot_level}_{month}.png")

    # Plot average errors as a function of depth
    if variable != "sossheig":
        fig = plt.figure(figsize=[10, 6])
        for i, rmse in enumerate(rmse_list):
            plot_vals = rmse[variable].mean(dim="time_counter")
            if variable == "votemper":
                depth_array = rmse.deptht
                variable_name = "Temperature"
            elif variable == "vosaline":
                depth_array = rmse.deptht
                variable_name = "Salinty"
            elif variable == "vozocrtx":
                depth_array = rmse.depthu
                variable_name = "Zonal Velocity"
            elif variable == "vomecrty":
                depth_array = rmse.depthv
                variable_name = "Meridional Velocity"
            elif variable == "vovecrtz":
                depth_array = rmse.depthw
                variable_name = "Vertical Velocity"
            print(plot_vals)
            plt.plot(plot_vals, depth_array, label=expt_ids[i])
        plt.gca().invert_yaxis()
        plt.xlabel("RMSE")
        plt.ylabel("Depth (m)")
        plt.legend()
        plt.title(f"{variable_name}")
        plt.savefig(f"{plot_dir}/rmse_{variable}_depths_{month}.png")


if __name__ == "__main__":

    date_str = "200903??T0000Z"
    month = "March"
    variables = ["sossheig", "votemper", "vosaline", "vozocrtx", "vomecrty", "vovecrtz"]
    expt_ids = ["u-di464", "u-di283", "u-di338", "u-di336", "u-di463"]
    expt_nos = [7, 11, 12]  # for u-di463
    expt_name = ["Control 1", "Free-run", "Control 2", "New CVs", "New CVs (no chi)", "New CVs (double sdv)", "New CVs (half_sdv)"]

    input_dir = "/projects/jodap/lrisley/GYRE/experiments"
    plot_dir = "plots"

    for variable in variables:

        if variable in ["votemper", "vosaline", "sossheig"]:
            grid = "T"
        elif variable in ["vozocrtx"]:
            grid = "U"
        elif variable in ["vomecrty"]:
            grid = "V"
        elif variable in ["vovecrtz"]:
            grid = "W"

        for expt_id in expt_ids[:-1]:
            if variable == "vovecrtz":
                stats_file = f"stats/stats_{expt_id}_{variable}_{month}.nc"
                mean_file = f"stats/mean_{expt_id}_{variable}_{month}.nc"
            else:
                stats_file = f"stats/stats_{expt_id}_{variable}_{month}.nc"
            try:
                rmse = xarray.open_dataset(stats_file)
                if variable == "vovecrtz":
                    mean = xarray.open_dataset(mean_file)
                print(f"Reading existing stats file {stats_file}")

            except:
                print(f"No stats file exists so generating it for variable {variable} and experiment {expt_id}")

                in_nr_files = f"{input_dir}/nature_run_gyre12/{date_str}_mersea.grid_{grid}_gyre12.nc"
                in_da_files = f"{input_dir}/{expt_id}/{date_str}_mersea.grid_{grid}.nc"

                nr_fields = read_fields(in_nr_files, variable)
                da_fields = read_fields(in_da_files, variable)

                new_rmse = calc_rmse(da_fields, nr_fields)
                new_rmse.to_netcdf(path=stats_file)
                rmse = xarray.open_dataset(stats_file)

                if variable == "vovecrtz":
                    new_mean = calc_spatial_mean(da_fields)
                    new_mean.to_netcdf(path=mean_file)
                    mean = xarray.open_dataset(mean_file)

        for expt_no in expt_nos:
            expt_id = 'u-di463'
            if variable == "vovecrtz":
                stats_file = f"stats/stats_{expt_id}_{expt_no}_{variable}_{month}.nc"
                mean_file = f"stats/mean_{expt_id}_{expt_no}_{variable}_{month}.nc"
            else:
                stats_file = f"stats/stats_{expt_id}_{expt_no}_{variable}.nc"
            try:
                rmse = xarray.open_dataset(stats_file)
                if variable == "vovecrtz":
                     mean = xarray.open_dataset(mean_file)
                print(f"Reading existing stats file {stats_file}")

            except:
                print(f"No stats file exists so generating it for variable {variable} and experiment {expt_no}")

                in_nr_files = f"{input_dir}/nature_run_gyre12/{date_str}_mersea.grid_{grid}_gyre12.nc"
                in_da_files = f"{input_dir}/{expt_id}/experiment_{expt_no}/{date_str}_mersea.grid_{grid}.nc"

                nr_fields = read_fields(in_nr_files, variable)
                da_fields = read_fields(in_da_files, variable)

                new_rmse = calc_rmse(da_fields, nr_fields)
                new_rmse.to_netcdf(path=stats_file)
                rmse = xarray.open_dataset(stats_file)

                if variable == "vovecrtz":
                    new_mean = calc_spatial_mean(da_fields)
                    new_mean.to_netcdf(path=mean_file)
                    mean = xarray.open_dataset(mean_file)

            if expt_id == expt_ids[0]:
                rmse_list = [rmse]
            else:
                rmse_list.append(rmse)

            if variable in ["vovecrtz"]:
                if expt_id == expt_ids[0]:
                    mean_list = [mean]
                else:
                    mean_list.append(mean)

        plot_rmse(rmse_list, variable, expt_name, plot_dir, month)

        if variable in ["vovecrtz"]:
            plot_spatial_mean(mean_list, expt_name, plot_dir, month)
