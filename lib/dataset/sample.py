import numpy as np

def cross(triangles):
    """
    Returns the cross product of two edges from input triangles
    Parameters
    --------------
    triangles: (n, 3, 3) float
      Vertices of triangles
    Returns
    --------------
    crosses : (n, 3) float
      Cross product of two edge vectors
    """
    vectors = np.diff(triangles, axis=1)
    crosses = np.cross(vectors[:, 0], vectors[:, 1])
    return crosses


def tri_area(triangles=None, crosses=None, sum=False):
    """
    Calculates the sum area of input triangles
    Parameters
    ----------
    triangles : (n, 3, 3) float
      Vertices of triangles
    crosses : (n, 3) float or None
      As a speedup don't re- compute cross products
    sum : bool
      Return summed area or individual triangle area
    Returns
    ----------
    area : (n,) float or float
      Individual or summed area depending on `sum` argument
    """
    if crosses is None:
        crosses = cross(triangles)
    area = (np.sum(crosses**2, axis=1)**.5) * .5
    if sum:
        return np.sum(area)
    return area

def sample_surface(triangles, count, area=None, ignore_face_idxs=None):
    """
    Sample the surface of a mesh, returning the specified
    number of points
    For individual triangle sampling uses this method:
    http://mathworld.wolfram.com/TrianglePointPicking.html
    Parameters
    ---------
    triangles : (n, 3, 3) float
      Vertices of triangles
    count : int
      Number of points to return
    Returns
    ---------
    samples : (count, 3) float
      Points in space on the surface of mesh
    face_index : (count,) int
      Indices of faces for each sampled point
    """

    # len(mesh.faces) float, array of the areas
    # of each face of the mesh
    if area is None:
      area = tri_area(triangles)
    if ignore_face_idxs is not None:
      assert area.shape[0] == ignore_face_idxs.shape[0]
      area[ignore_face_idxs] = 0

    # total area (float)
    area_sum = np.sum(area)
    # cumulative area (len(mesh.faces))
    area_cum = np.cumsum(area)
    face_pick = np.random.random(count) * area_sum
    face_index = np.searchsorted(area_cum, face_pick)

    # pull triangles into the form of an origin + 2 vectors
    tri_origins = triangles[:, 0]
    tri_vectors = triangles[:, 1:].copy()
    tri_vectors -= np.tile(tri_origins, (1, 2)).reshape((-1, 2, 3))

    # pull the vectors for the faces we are going to sample from
    tri_origins = tri_origins[face_index]
    tri_vectors = tri_vectors[face_index]

    # randomly generate two 0-1 scalar components to multiply edge vectors by
    random_lengths = np.random.random((len(tri_vectors), 2, 1))

    # points will be distributed on a quadrilateral if we use 2 0-1 samples
    # if the two scalar components sum less than 1.0 the point will be
    # inside the triangle, so we find vectors longer than 1.0 and
    # transform them to be inside the triangle
    random_test = random_lengths.sum(axis=1).reshape(-1) > 1.0
    random_lengths[random_test] -= 1.0
    random_lengths = np.abs(random_lengths)

    # multiply triangle edge vectors by the random lengths and sum
    sample_vector = (tri_vectors * random_lengths).sum(axis=1)

    # finally, offset by the origin to generate
    # (n,3) points in space on the triangle
    samples = sample_vector + tri_origins

    return samples, face_index
