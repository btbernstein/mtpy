import numpy as np

from mtpy.modeling.modem import Model

def get_centers(arr):
    """Get the centers from an array of cell boundaries.

    Args:
        arr (np.ndarray): An array of cell boundaries.

    Returns:
        np.ndarray: An array of cell centers.
    """
    return np.mean([arr[:-1], arr[1:]], axis=0)


def strip_padding(arr, pad, keep_start=False):
    """Strip padding cells from grid data. Padding cells occur at the
    the start and end of north and east grid axes, and at the end of
    grid depth axis.

    Note we handle the special case of `pad=0` by returning the data
    untouched (a `slice(None)` will do nothing when used as an index
    slice).

    Args:
        arr (np.ndarray): Axis of grid data (east, north or depth).
        pad (int): Number of padding cells being stripped.
        keep_start (bool): If True, padding is only stripped from the
            end of the data. Intended for use with depth axis.

    Return:
        np.ndarray: A copy of `arr` with padding cells removed.
    """
    if keep_start:
        pad = slice(None) if pad == 0 else slice(None, -pad)
    else:
        pad = slice(None) if pad == 0 else slice(pad, -pad)

    return arr[pad]


def strip_resgrid(res_model, y_pad, x_pad, z_pad):
    """Similar to `strip_padding` but handles the case of stripping
    padding from a 3D resistivity model.

    """
    y_pad = slice(None) if y_pad == 0 else slice(y_pad, -y_pad)
    x_pad = slice(None) if x_pad == 0 else slice(x_pad, -x_pad)
    z_pad = slice(None) if z_pad == 0 else slice(None, -z_pad)
    return res_model[y_pad, x_pad, z_pad]


def list_depths(model, zpad=None):
    """
    Return a list of available depth slices in the model.

    Args:
        model_file (str): Path to ModEM .rho file.
        zpad (int, optional): Number of padding slices to remove from
            bottom of model. If None, model pad_z value is used.

    Returns:
        list of float: A list of available depth slices.
    """
    # Try to insantiate from model file
    if not isinstance(model, Model):
        model = Model()
        model.read_model_file(model_fn=model)

    cz = get_centers(model.grid_z)
    zpad = model.pad_z if zpad is None else zpad
    return cz[:-zpad]


def get_depth_indicies(centers_z, depths):
    """Finds the indicies for the elements closest to those specified
    in a given list of depths.

    Args:
        centers_z (np.ndarray): Grid centers along the Z axis (i.e.
            available depths in our model).
        depths (list of int): A list of depths to retrieve indicies
            for. If None or empty, a list of all indicies is returned.

    Returns:
        set: A set of indicies closest to provided depths.
        list: If `depths` is None or empty, a list of all indicies.
    """
    def _nearest(array, value):
        """Get index for nearest element to value in an array.
        """
        idx = np.searchsorted(array, value, side="left")
        if idx > 0 and (idx == len(array)
                or math.fabs(value - array[idx - 1]) < math.fabs(value - array[idx])):
            return idx - 1
        else:
            return idx

    if depths:
        res = {_nearest(centers_z, d) for d in depths}
        print("Slices closest to requested depths: {}".format([centers_z[di] for di in res]))
        return res
    else:
        print("No depths provided, getting all slices...")
        return range(len(centers_z))


