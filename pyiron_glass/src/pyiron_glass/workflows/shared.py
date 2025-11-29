def get_lammps_command(server_kwargs=None):
    lmp_command = "mpiexec -n 1 --oversubscribe lmp_mpi -in lmp.in"
    if server_kwargs is not None:
        if isinstance(server_kwargs, dict) and len(server_kwargs) == 1:
            if "cores" in server_kwargs:
                lmp_command = "mpiexec -n {} --oversubscribe lmp_mpi -in lmp.in".format(str(server_kwargs["cores"]))
            else:
                raise ValueError("Server dictionary error: " + str(server_kwargs))
        elif isinstance(server_kwargs, dict) and len(server_kwargs) == 0:
            pass
        else:
            raise ValueError("Server dictionary error: " + str(server_kwargs))
    return lmp_command
