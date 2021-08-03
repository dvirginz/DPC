import numpy as np

def create_colormap(VERT):
    """
    Creates a uniform color map on a mesh

    Args:
        VERT (Nx3 ndarray): The vertices of the object to plot

    Returns:
        Nx3: The RGB colors per point on the mesh
    """
    VERT = np.double(VERT)
    minx = np.min(VERT[:, 0])
    miny = np.min(VERT[:, 1])
    minz = np.min(VERT[:, 2])
    maxx = np.max(VERT[:, 0])
    maxy = np.max(VERT[:, 1])
    maxz = np.max(VERT[:, 2])
    colors = np.stack([((VERT[:, 0] - minx) / (maxx - minx)), ((VERT[:, 1] - miny) /
                                                               (maxy - miny)), ((VERT[:, 2] - minz) / (maxz - minz))]).transpose()
    return colors

