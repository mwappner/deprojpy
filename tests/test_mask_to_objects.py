import numpy as np
import pytest
import tifffile

from deprojpy.mask import mask_to_objects, validate_binary_mask


def test_synthetic_mask_has_known_objects_and_graph():
    mask = np.zeros((21, 21), dtype=np.uint8)
    for row_start in (3, 12):
        for column_start in (3, 12):
            mask[row_start : row_start + 6, column_start : column_start + 6] = 255
    objects, graph = mask_to_objects(mask)
    assert len(objects) == 4
    assert graph.number_of_nodes() == 1
    assert graph.number_of_edges() == 0
    assert all(object_.junction_ids.tolist() == [1] for object_ in objects)


@pytest.mark.parametrize(
    "mask",
    [
        np.arange(100, dtype=float).reshape(10, 10),
        np.ones((10, 10), dtype=np.uint8),
        np.zeros((10, 10), dtype=np.uint8),
        np.full((10, 10), np.nan),
    ],
)
def test_binary_mask_validation_rejects_invalid_inputs(mask):
    with pytest.raises(ValueError, match="mask"):
        validate_binary_mask(mask)


def test_mask_with_no_retained_cells_has_helpful_error():
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[1:9, 1:9] = 255
    with pytest.raises(ValueError, match="no retained cells"):
        mask_to_objects(mask)


def test_sample_mask_counts(deproj_samples):
    mask = tifffile.imread(deproj_samples / "Segmentation-2.tif")
    objects, graph = mask_to_objects(mask)
    assert len(objects) == 426
    # README prints 1840 because MATLAB numel() counts a 920x2 node table.
    assert graph.number_of_nodes() == 920
    assert all(len(obj.boundary) >= 3 for obj in objects)
