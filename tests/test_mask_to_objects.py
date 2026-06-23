from pathlib import Path

import tifffile

from deprojpy.mask import mask_to_objects


SAMPLES = Path(__file__).parents[2] / "DeProj-matlab" / "samples"


def test_sample_mask_counts():
    mask = tifffile.imread(SAMPLES / "Segmentation-2.tif")
    objects, graph = mask_to_objects(mask)
    assert len(objects) == 426
    # README prints 1840 because MATLAB numel() counts a 920x2 node table.
    assert graph.number_of_nodes() == 920
    assert all(len(obj.boundary) >= 3 for obj in objects)

