"""Functions written to help with pyms"""
import matplotlib.pyplot as plt
import numpy as np

def nyquist_probe_spacing(resolution_limit=None, eV=None, alpha=None):
    """
        Calculate maximum probe spacing based on a naive use of nyquist theorem.
    """
    if resolution_limit is None:
        if eV is None and alpha is None:
            raise ValueError
        else:
            resolution_limit = wavev(eV) * alpha * 1e-3
    return 1 / (4 * resolution_limit)

def get_STEM_raster(
        object_real_size,
        object_scan_size        = None,
        probe_spacing           = None,
        n_probes                = None,
        ROI_0_to_1              = None,
        ROI_tiles               = None,
        tiling                  = None,
        ROI_shift_0_to_1        = None,
        ROI_shift_to_tiles      = None,
        probe_spacing_noise_std = 0,
        rot_angle               = None,
        rot_center              = None,
        return_probe_spacing    = False,
        ):
    """ Get probe grid
        Returns the probe positions for a sampled STEM raster by numbers
        between 0 and 1. 
        
        object_real_size: 2-tuple
            The object real size in Angstrom in which the probe positions will
            be reported. The output of this function is numbers betwen
            0 to 1. You will be passing a tilling and crystal to 
            pyms pattern maker and pyms will use it to define the
            real size of the object and uses the provided scan positions
            according to real dimensions. You have to provide here, the
            multiplication of tilling x crystal unit cell dimensions. 
                        
        object_scan_size: 2-tuple
            The object scanning size in Angstrom.
            This is the size to scan, for example you have a 
            probe spacing but you don't know the number of probes.
             
        probe_spacing: float or a 2-tuple of floats
             You need to know the probe spacing capability of your experiment.
             This has real physical meaning and is possible to set for every
             microscope. Still, if you are curious to use n_probes you can
             leave this as None.
        
        n_probes: int or a 2-tuple of intd
             You may wish to provide n_probes in each direction instead of
             probe_spacing...It is unrealistic but hey! it is simulation.
             If you don;t provide probe_spacing, we use this to linearly space
             the ROI. If you provide the probe_spacing we use it as the step 
             size to cover as many steps in the ROI as possible.
        
        ROI_0_to_1: 4-tuple of floats
            should be a tuple of four floats. 0 to 1 is the object 0 to 1.
            the hint for Python 3.11 would be tuple[float, float, float, flaot].
            If given, n_probes and ROI_shift_0_to_1 will be disregarded. 
        
        ROI_tiles: a tuple of two tuples each would be two integers.
            Instead of ROI, you may like to specify specific tiles.
            In total there would be four integers in the form of:
            ((start_tile_r, start_tile_c), (end_tile_r, end_tile_c))
            Then you must provide tiling a 2-tuple you used to make the object.
        
        tiling: a 2-tuple of two integeres
            You use this for making the object. You have to pass it if you
            are using ROI_tiles
        
        ROI_shift_0_to_1: 2-tuple between 0 and 1
            when not using the tiling, When giving n_probes 
            you may wish to shift the ROI, this would be your option. 
            Must be between 0 and 1 which is the size of entire
            object. So choose tiling to be twice as large and choose a middle 
            region for ROI by setting this to ROI_shift_0_to_1 = (0.25, 0.25)
        
        ROI_shift_to_tiles: 2-tuple for specific tile numbers
            to begin the sampling from specific tile numbers, use this along
            with tiling which must be provided. We will divide the objects
            real size by tiling to get the unit_cell size and start the 
            sampling using ROI_shift_to_tiles.
        
        probe_spacing_noise_std:float
            Your prbe sure is noisy, give it a std, maybe some 1/10th of your
            probe_spacing?
        
        rot_angle; float
            Rotation angle in radian to rotate the grid around its center.
            
        rot_center: 2-tuple
            If provided, the rotation will happen around this center
        
        return_probe_spacing: bool
            if probe_spacing is not provided but return_probe_spacing is 
            set to True, the output is a 2-tuple, the first element is 
            the grid and the second is a 2-tuple of probe_spacing)
    """
    assert object_real_size is not None, \
        'You have to provide the real size of the object as you will' + \
        ' be passing tiling and crystal to pyms pattern maker.'
    
    if probe_spacing is not None:
        try:
            probe_spacing = float(probe_spacing)
            probe_spacing = (probe_spacing, probe_spacing)            
        except:
            try:
                psx, psy = probe_spacing
                psx = float(psx)
                psy = float(psy)
            except Exception:
                print(f'probe_spacing is given as {probe_spacing}, it '\
                      ' must be either a float or a tuple of two floats')
                raise Exception

    if n_probes is not None:
        try:
            n_probes = int(n_probes)
            n_probes = (n_probes, n_probes)            
        except:
            try:
                npx, npy = n_probes
                n_probes = (int(npx), int(npy))
            except Exception:
                print(f'n_probes is given as {n_probes}, it '\
                      ' must be either an int or a tuple of two ints')
                raise Exception

    if object_scan_size is None:
        object_scan_size = object_real_size

    if tiling is not None:
        try:
            tile_r, tile_c = tiling
        except Exception as e:
            print(f'tiling is given as {tiling}.'
                  ' But it should be a 2-tuple you used to make the object.')
            raise e

    if ROI_0_to_1 is not None:
        assert (0 <= np.array(ROI_0_to_1)).all() \
                and (np.array(ROI_0_to_1) <= 1).all(), \
                'ROI_0_to_1 should be between 0 and 1'

    if ROI_shift_0_to_1 is not None:
        assert ROI_shift_to_tiles is None, \
            'You cannot provide both ROI_shift_to_tiles and ROI_shift_0_to_1'
        assert (0 <= np.array(ROI_shift_0_to_1)).all() \
                and (np.array(ROI_shift_0_to_1) <= 1).all(), \
                'ROI_shift_0_to_1 should be between 0 and 1'

    if ROI_shift_to_tiles is not None:
        assert ROI_shift_0_to_1 is None, \
            'You cannot provide both ROI_shift_to_tiles and ROI_shift_0_to_1'
        assert tiling is not None,\
            'tiling must be provided when ROI_shift_to_tiles is used'
        try:
            start_tile_r, start_tile_c = ROI_shift_to_tiles
        except Exception as e:
            print(f'ROI_shift_to_tiles must be a 2-tuple of two ints for '
                  f'tile numbers, but it is {ROI_shift_to_tiles}.')
            raise e
        else:
            ROI_shift_0_to_1 = [start_tile_r/tile_r, start_tile_c/tile_c]
            
    assert (probe_spacing is not None) | (n_probes is not None), \
        'At least one of probe_spacing or n_probes must be given.'

    if (ROI_tiles is not None) | (ROI_0_to_1 is not None):
        assert (probe_spacing is None) | (n_probes is None), \
                'One of ROI_0_to_1 or ROI_tiles are given. '\
                'Then only one of probe_spacing or n_probes can be given.'

    if ROI_tiles is not None:
        assert tiling is not None,\
            'tiling must be provided when ROI_tiles is used'
        assert ROI_0_to_1 is None, \
            'When using ROI_tiles, do not provide ROI_0_to_1'
        assert ROI_shift_0_to_1 is None, \
            'When using ROI_tiles, do not provide ROI_shift_0_to_1'
        try:
            start_tile, end_tile = ROI_tiles
            start_tile_r, start_tile_c = start_tile
            end_tile_r, end_tile_c = end_tile
        except Exception as e:
            print(f'ROI_tiles is given as {ROI_tiles}.'
                  ' But it should be a 2-tuple of two 2-tiples. e.g.:'
                  ' ((start_tile_r, start_tile_c), (end_tile_r, end_tile_c))')
            raise e
        else:
            ROI_0_to_1 = [start_tile_r/tile_r, start_tile_c/tile_c, 
                          end_tile_r/tile_r,   end_tile_c/tile_c]
        
    else:        
        if ROI_0_to_1 is None:
            ROI_0_to_1 = [0, 0, 1, 1]
            if (n_probes is not None) & (probe_spacing is not None):
                ROI_0_to_1 = [0, 0,
                    n_probes[0] * probe_spacing[0] / object_scan_size[0],
                    n_probes[1] * probe_spacing[1] / object_scan_size[1]]
            if (ROI_0_to_1[2] > 1) | (ROI_0_to_1[3] > 1):
                print(f'The object_scan_size must be larger than the product'
                      ' n_probes x probe_spacing. but now it is '
                      f' {object_scan_size}.')
                object_scan_size = [n_probes[0] * probe_spacing[0],
                                    n_probes[1] * probe_spacing[1]]
                print(f'The new object_scan_size is: {object_scan_size}')
                print(f'The  object_real_size stays: {object_real_size}')
                print(f'The scanning positions will be reported within the '
                      ' object_real_size area.')
                ROI_0_to_1 = [0, 0, 1, 1]
                
    if ROI_shift_0_to_1 is not None:        
        ROI=[ROI_0_to_1[0] + ROI_shift_0_to_1[0],
             ROI_0_to_1[1] + ROI_shift_0_to_1[1], 
             ROI_0_to_1[2] + ROI_shift_0_to_1[0],
             ROI_0_to_1[3] + ROI_shift_0_to_1[1]]
    else:
        ROI = ROI_0_to_1
    
    ROI = list(ROI) 
    ROI[0] = np.maximum(ROI[0], 0)
    ROI[1] = np.minimum(ROI[1], 1)
    ROI[2] = np.maximum(ROI[2], 0)
    ROI[3] = np.minimum(ROI[3], 1)
    
    if n_probes is None:    
        rows = np.arange(ROI[0]*object_scan_size[0],
                         ROI[2]*object_scan_size[0],
                         probe_spacing[0], dtype = 'float64')
        clms = np.arange(ROI[1]*object_scan_size[1],
                         ROI[3]*object_scan_size[1],
                         probe_spacing[1], dtype = 'float64')
    else:
        rows = np.linspace(ROI[0]*object_scan_size[0],
                           ROI[2]*object_scan_size[0],
                           n_probes[0] + 1, dtype = 'float64')[:-1]
        clms = np.linspace(ROI[1]*object_scan_size[1],
                           ROI[3]*object_scan_size[1],
                           n_probes[1] + 1, dtype = 'float64')[:-1]
        if (len(rows) > 1) & (len(clms) > 1):
            probe_spacing = [rows[1] - rows[0], rows[1] - rows[0]]
        else:
            probe_spacing = None

    cc, rr = np.meshgrid(clms, rows)
    rr = np.expand_dims(rr, -1)
    cc = np.expand_dims(cc, -1)
    grid = np.concatenate((rr, cc), axis=2)
    
    if rot_angle is not None:
        R_matrix = np.array(
            [[np.cos(rot_angle), np.sin(rot_angle)], 
             [-np.sin(rot_angle), np.cos(rot_angle)]])
        grid_algebraic = grid.reshape(grid.shape[0] * grid.shape[1], 2).T
        grid_algebraic_T = grid_algebraic.mean(1)
        if rot_center is not None:
            grid_algebraic_T = np.array(rot_center)
        grid_algebraic[0] -= grid_algebraic_T[0]
        grid_algebraic[1] -= grid_algebraic_T[1]
        grid_algebraic = R_matrix @ grid_algebraic 
        grid_algebraic[0] += grid_algebraic_T[0]
        grid_algebraic[1] += grid_algebraic_T[1]
        grid = grid_algebraic.T.reshape(grid.shape[0], grid.shape[1], 2)

    if probe_spacing_noise_std:
        grid += probe_spacing_noise_std * np.random.randn(*grid.shape)

    if ((object_scan_size[0] > object_real_size[0]) |
        (object_scan_size[1] > object_real_size[1])):
        grid[..., 0] = grid[..., 0] % object_real_size[0]
        grid[..., 1] = grid[..., 1] % object_real_size[1]

    grid[..., 0] /= object_real_size[0]
    grid[..., 1] /= object_real_size[1]
    # If there is a noise in probe positions it is half Gaussian for the edges
    grid[grid < 0] =   - grid[grid < 0]
    grid[grid > 1] = 2 - grid[grid > 1]
    
    if return_probe_spacing:
        return grid, probe_spacing
    else:
        return grid

def plot_scan_positions(
        atoms_xyz, atoms_Z, unitcell, object_real_size,
        probe_spacing, scan_posn, markersize = 1, atom_markersize = 8,
        text_size = 'xx-small', fig_ax = None, limit_figure = False):
    """
    Plots the scan positions and atoms in a STEM (Scanning Transmission Electron Microscopy) simulation.

    Parameters:
    ----------
    atoms_xyz : np.ndarray
        An (N, 3) array representing the positions of atoms in 3D space.
        Each row contains the (x, y, z) coordinates of an atom, with x and y
        in fractional coordinates relative to the unit cell, and z representing the depth.
    
    atoms_Z : np.ndarray
        A (N,) array representing the atomic numbers Z of the atoms. Each entry
        corresponds to an atom in `atoms_xyz`.
    
    unitcell : tuple of float
        A tuple (a, b) representing the dimensions of the unit cell along the x and y axes.
    
    object_real_size : tuple of float
        A tuple (width, height) representing the real size of the object (in physical units).
    
    probe_spacing : float or tuple of float
        The spacing between the probe points in the scan grid. If a single float
        is provided, the same spacing is used for both x and y directions.
    
    scan_posn : np.ndarray
        A (M, 2) array representing the real-world positions of the scanning probe.
        Each row contains the (x, y) coordinates of a scan position.
    
    markersize : float, optional (default: 1)
        The size of the markers used to plot the scan positions.
    
    fig_ax : tuple (matplotlib.figure.Figure, matplotlib.axes.Axes), optional
        If provided, this tuple (fig, ax) will be used for plotting. If None,
        a new figure and axis will be created.

    Returns:
    -------
    fig : matplotlib.figure.Figure
        The figure object containing the plot.

    ax : matplotlib.axes.Axes
        The axes object containing the plot.

    Notes:
    ------
    - The function plots atoms as circles ('o') with a color corresponding to their atomic number.
    - The positions of the scanning probe are plotted in red for the region of interest (ROI).
    - The function also generates a perfect STEM raster grid and overlays it with green 'x' markers.
    - Annotations are included to label specific scan positions and atoms.
    """

    scan_posn_perfect = get_STEM_raster(
        object_real_size = object_real_size,
        probe_spacing = probe_spacing)
    scan_posn_tiling = get_STEM_raster(
        object_real_size = object_real_size,
        probe_spacing = (unitcell[0], unitcell[1]))
            
    unique_atoms = np.unique(atoms_Z)
    atom_colors = np.random.rand(len(unique_atoms), 3)
    color_map = {Z: color for Z, color in zip(unique_atoms, atom_colors)}  

    if fig_ax is None:
        fig = plt.figure()
        ax = fig.add_subplot()
    else:
        fig, ax = fig_ax
    for unit_cell_r in np.arange(np.ceil(object_real_size[0] / unitcell[0])):
      for unit_cell_c in np.arange(np.ceil(object_real_size[1] / unitcell[1])):
        for atom_cnt, atom in enumerate(atoms_xyz):
            x, y, z, Z = ((atom[0] + unit_cell_r) * unitcell[0], 
                          (atom[1] + unit_cell_c) * unitcell[1],
                          atom[2], atoms_Z[atom_cnt])
    
            ax.plot(x, y, marker='o', markersize=atom_markersize, 
                    markerfacecolor='none',
                    markeredgewidth= atom_markersize /5, 
                    markeredgecolor='black', zorder=1)
    
            ax.plot(x, y, '.', markersize=atom_markersize * 1.2, 
                    color=color_map[Z], zorder=-z)
            ax.text(x, y, atom_cnt, size = text_size, color = 'green')

    ax.plot(scan_posn_perfect[..., 0].ravel() * object_real_size[0], 
                scan_posn_perfect[..., 1].ravel() * object_real_size[1], '.', 
                color = 'blue', label = 'object', markersize = markersize)
    ax.plot(scan_posn[..., 0].ravel() * object_real_size[0], 
            scan_posn[..., 1].ravel() * object_real_size[1], '.', 
            color = 'red', label = 'ROI', markersize = markersize)
    
    for pos_cnt in [0, 5, scan_posn.shape[0]*scan_posn.shape[1] - 1]:
        ax.text(scan_posn[..., 0].ravel()[pos_cnt] * object_real_size[0], 
                scan_posn[..., 1].ravel()[pos_cnt] * object_real_size[1], 
                pos_cnt, color = 'purple', size = text_size)
    ax.plot(scan_posn_tiling[..., 0].ravel() * object_real_size[0], 
            scan_posn_tiling[..., 1].ravel() * object_real_size[1], 'x',
            color = 'green', label = 'unit cell', markersize = 5 * markersize)
    
    if (limit_figure):
        ax.set_xlim([0, unitcell[0] * np.ceil((scan_posn[..., 0].ravel(
            ) * object_real_size[0]).max() / unitcell[0])])
        ax.set_ylim([0, unitcell[1] * np.ceil((scan_posn[..., 1].ravel(
            ) * object_real_size[1]).max() / unitcell[1])])
        
    plt.legend()
    ax.set_aspect('equal', 'box')

    return fig, ax

def phase_from_com(com, reg=1e-10, rsize=[1, 1]):
    """
    Integrate 4D-STEM centre of mass (DPC) measurements to calculate object phase.

    Assumes a three dimensional array com, with the final two dimensions
    corresponding to the image and the first dimension of the array corresponding
    to the y and x centre of mass respectively.
    """
    # Get shape of arrays
    ny, nx = com.shape[1:]
    s = (ny, nx)
    s = None

    d = np.asarray(rsize) / np.asarray([ny, nx])
    # Calculate Fourier coordinates for array
    ky = np.fft.fftfreq(ny, d=d[0])
    kx = np.fft.rfftfreq(nx, d=d[1])

    # Calculate numerator and denominator expressions for solution of
    # phase from centre of mass measurements
    numerator = ky[:, None] * np.fft.rfft2(com[0]) + kx[None, :] * np.fft.rfft2(com[1])
    denominator = 1j * ((kx ** 2)[None, :] + (ky ** 2)[:, None]) + reg

    # Avoid a divide by zero for the origin of the Fourier coordinates
    numerator[0, 0] = 0
    denominator[0, 0] = 1

    # Return real part of the inverse Fourier transform
    return np.fft.irfft2(numerator / denominator)