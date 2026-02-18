"""Shared module for amorphouspy simulation workflows.

This module contains shared functionality which is reused in the individual workflows.
"""


def get_lammps_command(server_kwargs: dict | None = None) -> str:
    """Generate a LAMMPS command, by default this function returns: "mpiexec -n 1 --oversubscribe lmp_mpi -in lmp.in".

    Args:
        server_kwargs: Server dictionary for example: {"cores": 2}.

    Returns:
        LAMMPS command as a string.

    """
    lmp_command = "mpiexec -n 1 --oversubscribe lmp_mpi -in lmp.in"
    if server_kwargs is not None:
        if isinstance(server_kwargs, dict) and len(server_kwargs) == 1:
            if "cores" in server_kwargs:
                lmp_command = "mpiexec -n {} --oversubscribe lmp_mpi -in lmp.in".format(str(server_kwargs["cores"]))
            else:
                raise ValueError("Server dictionary error: " + str(server_kwargs))
        elif isinstance(server_kwargs, (dict, list)) and len(server_kwargs) == 0:
            pass
        else:
            raise ValueError("Server dictionary error: " + str(server_kwargs))
    return lmp_command
