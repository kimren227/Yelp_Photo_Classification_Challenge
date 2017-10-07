"""
Automatically run tests for this library.
Simply execute
    python test.py
"""
from __future__ import print_function, division
import imgaug as ia
from imgaug import augmenters as iaa
from imgaug import parameters as iap
import numpy as np
import random
import six
import six.moves as sm
from scipy import misc
from skimage import data

def main():
    test_is_single_integer()
    test_is_single_float()

    test_find()
    test_remove()
    test_hooks()

    test_Noop()
    test_Lambda()
    test_AssertLambda()
    test_AssertShape()
    test_Crop()
    test_Fliplr()
    test_Flipud()
    test_GaussianBlur()
    test_AdditiveGaussianNoise()
    # MultiplicativeGaussianNoise
    # ReplacingGaussianNoise
    test_Dropout()
    test_Multiply()
    test_Affine()
    test_ElasticTransformation()
    test_Sequential()
    test_Sometimes()

    # these functions use various augmenters, so test them last
    test_2d_inputs()
    test_background_augmentation()
    test_determinism()
    test_keypoint_augmentation()
    test_copy_random_state()

    print("Finished without errors.")

def test_is_single_integer():
    assert ia.is_single_integer("A") == False
    assert ia.is_single_integer(None) == False
    assert ia.is_single_integer(1.2) == False
    assert ia.is_single_integer(1.0) == False
    assert ia.is_single_integer(np.ones((1,), dtype=np.float32)[0]) == False
    assert ia.is_single_integer(1) == True
    assert ia.is_single_integer(1234) == True
    assert ia.is_single_integer(np.ones((1,), dtype=np.uint8)[0]) == True
    assert ia.is_single_integer(np.ones((1,), dtype=np.int32)[0]) == True

def test_is_single_float():
    assert ia.is_single_float("A") == False
    assert ia.is_single_float(None) == False
    assert ia.is_single_float(1.2) == True
    assert ia.is_single_float(1.0) == True
    assert ia.is_single_float(np.ones((1,), dtype=np.float32)[0]) == True
    assert ia.is_single_float(1) == False
    assert ia.is_single_float(1234) == False
    assert ia.is_single_float(np.ones((1,), dtype=np.uint8)[0]) == False
    assert ia.is_single_float(np.ones((1,), dtype=np.int32)[0]) == False

def test_find():
    reseed()

    noop1 = iaa.Noop(name="Noop")
    fliplr = iaa.Fliplr(name="Fliplr")
    flipud = iaa.Flipud(name="Flipud")
    noop2 = iaa.Noop(name="Noop2")
    seq2 = iaa.Sequential([flipud, noop2], name="Seq2")
    seq1 = iaa.Sequential([noop1, fliplr, seq2], name="Seq")

    augs = seq1.find_augmenters_by_name("Seq")
    assert len(augs) == 1
    assert augs[0] == seq1

    augs = seq1.find_augmenters_by_name("Seq2")
    assert len(augs) == 1
    assert augs[0] == seq2

    augs = seq1.find_augmenters_by_names(["Seq", "Seq2"])
    assert len(augs) == 2
    assert augs[0] == seq1
    assert augs[1] == seq2

    augs = seq1.find_augmenters_by_name(r"Seq.*", regex=True)
    assert len(augs) == 2
    assert augs[0] == seq1
    assert augs[1] == seq2

    augs = seq1.find_augmenters(lambda aug, parents: aug.name in ["Seq", "Seq2"])
    assert len(augs) == 2
    assert augs[0] == seq1
    assert augs[1] == seq2

    augs = seq1.find_augmenters(lambda aug, parents: aug.name in ["Seq", "Seq2"] and len(parents) > 0)
    assert len(augs) == 1
    assert augs[0] == seq2

    augs = seq1.find_augmenters(lambda aug, parents: aug.name in ["Seq", "Seq2"], flat=False)
    assert len(augs) == 2
    assert augs[0] == seq1
    assert augs[1] == [seq2]

def test_remove():
    reseed()

    def get_seq():
        noop1 = iaa.Noop(name="Noop")
        fliplr = iaa.Fliplr(name="Fliplr")
        flipud = iaa.Flipud(name="Flipud")
        noop2 = iaa.Noop(name="Noop2")
        seq2 = iaa.Sequential([flipud, noop2], name="Seq2")
        seq1 = iaa.Sequential([noop1, fliplr, seq2], name="Seq")
        return seq1

    augs = get_seq()
    augs = augs.remove_augmenters(lambda aug, parents: aug.name == "Seq2")
    seqs = augs.find_augmenters_by_name(r"Seq.*", regex=True)
    assert len(seqs) == 1
    assert seqs[0].name == "Seq"

    augs = get_seq()
    augs = augs.remove_augmenters(lambda aug, parents: aug.name == "Seq2" and len(parents) == 0)
    seqs = augs.find_augmenters_by_name(r"Seq.*", regex=True)
    assert len(seqs) == 2
    assert seqs[0].name == "Seq"
    assert seqs[1].name == "Seq2"

    augs = get_seq()
    augs = augs.remove_augmenters(lambda aug, parents: True)
    assert augs is not None
    assert isinstance(augs, iaa.Noop)

    augs = get_seq()
    augs = augs.remove_augmenters(lambda aug, parents: True, noop_if_topmost=False)
    assert augs is None

def test_hooks():
    reseed()

    image = np.array([[0, 0, 1],
                      [0, 0, 1],
                      [0, 1, 1]], dtype=np.uint8)
    image_lr = np.array([[1, 0, 0],
                         [1, 0, 0],
                         [1, 1, 0]], dtype=np.uint8)
    image_ud = np.array([[0, 1, 1],
                         [0, 0, 1],
                         [0, 0, 1]], dtype=np.uint8)
    image_lrud = np.array([[1, 1, 0],
                           [1, 0, 0],
                           [1, 0, 0]], dtype=np.uint8)
    image = image[:, :, np.newaxis]
    image_lr = image_lr[:, :, np.newaxis]
    image_ud = image_ud[:, :, np.newaxis]
    image_lrud = image_lrud[:, :, np.newaxis]

    seq = iaa.Sequential([iaa.Fliplr(1.0), iaa.Flipud(1.0)])

    # preprocessing
    def preprocessor(images, augmenter, parents):
        img = np.copy(images)
        img[0][1, 1, 0] += 1
        return img
    hooks = ia.HooksImages(preprocessor=preprocessor)
    images_aug = seq.augment_images([image], hooks=hooks)
    expected = np.copy(image_lrud)
    expected[1, 1, 0] = 3
    assert np.array_equal(images_aug[0], expected)

    # postprocessing
    def postprocessor(images, augmenter, parents):
        img = np.copy(images)
        img[0][1, 1, 0] += 1
        return img
    hooks = ia.HooksImages(postprocessor=postprocessor)
    images_aug = seq.augment_images([image], hooks=hooks)
    expected = np.copy(image_lrud)
    expected[1, 1, 0] = 3
    assert np.array_equal(images_aug[0], expected)

    # propagating
    def propagator(images, augmenter, parents, default):
        if "Seq" in augmenter.name:
            return False
        else:
            return default
    hooks = ia.HooksImages(propagator=propagator)
    images_aug = seq.augment_images([image], hooks=hooks)
    assert np.array_equal(images_aug[0], image)

    # activation
    def activator(images, augmenter, parents, default):
        if "Flipud" in augmenter.name:
            return False
        else:
            return default
    hooks = ia.HooksImages(activator=activator)
    images_aug = seq.augment_images([image], hooks=hooks)
    assert np.array_equal(images_aug[0], image_lr)

def test_Noop():
    reseed()

    images = create_random_images((16, 70, 50, 3))
    keypoints = create_random_keypoints((16, 70, 50, 3), 4)
    aug = iaa.Noop()
    aug_det = aug.to_deterministic()

    observed = aug.augment_images(images)
    expected = images
    assert np.array_equal(observed, expected)

    observed = aug_det.augment_images(images)
    expected = images
    assert np.array_equal(observed, expected)

    observed = aug.augment_keypoints(keypoints)
    expected = keypoints
    assert keypoints_equal(observed, expected)

    observed = aug_det.augment_keypoints(keypoints)
    expected = keypoints
    assert keypoints_equal(observed, expected)

def test_Lambda():
    reseed()

    base_img = np.array([[0, 0, 1],
                         [0, 0, 1],
                         [0, 1, 1]], dtype=np.uint8)
    base_img = base_img[:, :, np.newaxis]
    images = np.array([base_img])
    images_list = [base_img]

    images_aug = images + 1
    images_aug_list = [image + 1 for image in images_list]

    keypoints = [ia.KeypointsOnImage([ia.Keypoint(x=0, y=0), ia.Keypoint(x=1, y=1), ia.Keypoint(x=2, y=2)], shape=base_img.shape)]
    keypoints_aug = [ia.KeypointsOnImage([ia.Keypoint(x=1, y=0), ia.Keypoint(x=2, y=1), ia.Keypoint(x=0, y=2)], shape=base_img.shape)]

    def func_images(images, random_state, parents, hooks):
        if isinstance(images, list):
            images = [image + 1 for image in images]
        else:
            images = images + 1
        return images

    def func_keypoints(keypoints_on_images, random_state, parents, hooks):
        for keypoints_on_image in keypoints_on_images:
            for kp in keypoints_on_image.keypoints:
                kp.x = (kp.x + 1) % 3
        return keypoints_on_images

    aug = iaa.Lambda(func_images, func_keypoints)
    aug_det = aug.to_deterministic()

    # check once that the augmenter can handle lists correctly
    observed = aug.augment_images(images_list)
    expected = images_aug_list
    assert array_equal_lists(observed, expected)

    observed = aug_det.augment_images(images_list)
    expected = images_aug_list
    assert array_equal_lists(observed, expected)

    for _ in sm.xrange(10):
        observed = aug.augment_images(images)
        expected = images_aug
        assert np.array_equal(observed, expected)

        observed = aug_det.augment_images(images)
        expected = images_aug
        assert np.array_equal(observed, expected)

        observed = aug.augment_keypoints(keypoints)
        expected = keypoints_aug
        assert keypoints_equal(observed, expected)

        observed = aug_det.augment_keypoints(keypoints)
        expected = keypoints_aug
        assert keypoints_equal(observed, expected)

def test_AssertLambda():
    reseed()

    base_img = np.array([[0, 0, 1],
                         [0, 0, 1],
                         [0, 1, 1]], dtype=np.uint8)
    base_img = base_img[:, :, np.newaxis]
    images = np.array([base_img])
    images_list = [base_img]

    keypoints = [ia.KeypointsOnImage([ia.Keypoint(x=0, y=0), ia.Keypoint(x=1, y=1), ia.Keypoint(x=2, y=2)], shape=base_img.shape)]

    def func_images_succeeds(images, random_state, parents, hooks):
        return images[0][0, 0] == 0 and images[0][2, 2] == 1

    def func_images_fails(images, random_state, parents, hooks):
        return images[0][0, 0] == 1

    def func_keypoints_succeeds(keypoints_on_images, random_state, parents, hooks):
        return keypoints_on_images[0].keypoints[0].x == 0 and keypoints_on_images[0].keypoints[2].x == 2

    def func_keypoints_fails(keypoints_on_images, random_state, parents, hooks):
        return keypoints_on_images[0].keypoints[0].x == 2

    aug_succeeds = iaa.AssertLambda(func_images_succeeds, func_keypoints_succeeds)
    aug_succeeds_det = aug_succeeds.to_deterministic()
    aug_fails = iaa.AssertLambda(func_images_fails, func_keypoints_fails)
    aug_fails_det = aug_fails.to_deterministic()

    # images as numpy array
    observed = aug_succeeds.augment_images(images)
    expected = images
    assert np.array_equal(observed, expected)

    try:
        observed = aug_fails.augment_images(images)
        errored = False
    except AssertionError as e:
        errored = True
    assert errored

    observed = aug_succeeds_det.augment_images(images)
    expected = images
    assert np.array_equal(observed, expected)

    try:
        observed = aug_fails.augment_images(images)
        errored = False
    except AssertionError as e:
        errored = True
    assert errored

    # Lists of images
    observed = aug_succeeds.augment_images(images_list)
    expected = images_list
    assert array_equal_lists(observed, expected)

    try:
        observed = aug_fails.augment_images(images_list)
        errored = False
    except AssertionError as e:
        errored = True
    assert errored

    observed = aug_succeeds_det.augment_images(images_list)
    expected = images_list
    assert array_equal_lists(observed, expected)

    try:
        observed = aug_fails.augment_images(images_list)
        errored = False
    except AssertionError as e:
        errored = True
    assert errored

    # keypoints
    observed = aug_succeeds.augment_keypoints(keypoints)
    expected = keypoints
    assert keypoints_equal(observed, expected)

    try:
        observed = aug_fails.augment_keypoints(keypoints)
        errored = False
    except AssertionError as e:
        errored = True
    assert errored

    observed = aug_succeeds_det.augment_keypoints(keypoints)
    expected = keypoints
    assert keypoints_equal(observed, expected)

    try:
        observed = aug_fails.augment_keypoints(keypoints)
        errored = False
    except AssertionError as e:
        errored = True
    assert errored

def test_AssertShape():
    reseed()

    base_img = np.array([[0, 0, 1, 0],
                         [0, 0, 1, 0],
                         [0, 1, 1, 0]], dtype=np.uint8)
    base_img = base_img[:, :, np.newaxis]
    images = np.array([base_img])
    images_list = [base_img]
    keypoints = [ia.KeypointsOnImage([ia.Keypoint(x=0, y=0), ia.Keypoint(x=1, y=1), ia.Keypoint(x=2, y=2)], shape=base_img.shape)]

    base_img_h4 = np.array([[0, 0, 1, 0],
                            [0, 0, 1, 0],
                            [0, 1, 1, 0],
                            [1, 0, 1, 0]], dtype=np.uint8)
    base_img_h4 = base_img_h4[:, :, np.newaxis]
    images_h4 = np.array([base_img_h4])
    keypoints_h4 = [ia.KeypointsOnImage([ia.Keypoint(x=0, y=0), ia.Keypoint(x=1, y=1), ia.Keypoint(x=2, y=2)], shape=base_img_h4.shape)]

    # image must have exactly shape (1, 3, 4, 1)
    aug = iaa.AssertShape((1, 3, 4, 1))
    aug_det = aug.to_deterministic()

    # check once that the augmenter can handle lists correctly
    observed = aug.augment_images(images_list)
    expected = images_list
    assert array_equal_lists(observed, expected)

    observed = aug_det.augment_images(images_list)
    expected = images_list
    assert array_equal_lists(observed, expected)

    for _ in sm.xrange(10):
        observed = aug.augment_images(images)
        expected = images
        assert np.array_equal(observed, expected)

        observed = aug_det.augment_images(images)
        expected = images
        assert np.array_equal(observed, expected)

        observed = aug.augment_keypoints(keypoints)
        expected = keypoints
        assert keypoints_equal(observed, expected)

        observed = aug_det.augment_keypoints(keypoints)
        expected = keypoints
        assert keypoints_equal(observed, expected)

        try:
            observed = aug.augment_images(images_h4)
            errored = False
        except AssertionError as e:
            errored = True
        assert errored

        try:
            observed = aug.augment_keypoints(keypoints_h4)
            errored = False
        except AssertionError as e:
            errored = True
        assert errored

    # any value for number of images allowed (None)
    aug = iaa.AssertShape((None, 3, 4, 1))
    aug_det = aug.to_deterministic()
    for _ in sm.xrange(10):
        observed = aug.augment_images(images)
        expected = images
        assert np.array_equal(observed, expected)

        observed = aug_det.augment_images(images)
        expected = images
        assert np.array_equal(observed, expected)

        observed = aug.augment_keypoints(keypoints)
        expected = keypoints
        assert keypoints_equal(observed, expected)

        observed = aug_det.augment_keypoints(keypoints)
        expected = keypoints
        assert keypoints_equal(observed, expected)

        try:
            observed = aug.augment_images(images_h4)
            errored = False
        except AssertionError as e:
            errored = True
        assert errored

        try:
            observed = aug.augment_keypoints(keypoints_h4)
            errored = False
        except AssertionError as e:
            errored = True
        assert errored

    # list of possible choices [1, 3, 5] for height
    aug = iaa.AssertShape((1, [1, 3, 5], 4, 1))
    aug_det = aug.to_deterministic()
    for _ in sm.xrange(10):
        observed = aug.augment_images(images)
        expected = images
        assert np.array_equal(observed, expected)

        observed = aug_det.augment_images(images)
        expected = images
        assert np.array_equal(observed, expected)

        observed = aug.augment_keypoints(keypoints)
        expected = keypoints
        assert keypoints_equal(observed, expected)

        observed = aug_det.augment_keypoints(keypoints)
        expected = keypoints
        assert keypoints_equal(observed, expected)

        try:
            observed = aug.augment_images(images_h4)
            errored = False
        except AssertionError as e:
            errored = True
        assert errored

        try:
            observed = aug.augment_keypoints(keypoints_h4)
            errored = False
        except AssertionError as e:
            errored = True
        assert errored

    # range of 1-3 for height (tuple comparison is a <= x < b, so we use (1,4) here)
    aug = iaa.AssertShape((1, (1, 4), 4, 1))
    aug_det = aug.to_deterministic()
    for _ in sm.xrange(10):
        observed = aug.augment_images(images)
        expected = images
        assert np.array_equal(observed, expected)

        observed = aug_det.augment_images(images)
        expected = images
        assert np.array_equal(observed, expected)

        observed = aug.augment_keypoints(keypoints)
        expected = keypoints
        assert keypoints_equal(observed, expected)

        observed = aug_det.augment_keypoints(keypoints)
        expected = keypoints
        assert keypoints_equal(observed, expected)

        try:
            observed = aug.augment_images(images_h4)
            errored = False
        except AssertionError as e:
            errored = True
        assert errored

        try:
            observed = aug.augment_keypoints(keypoints_h4)
            errored = False
        except AssertionError as e:
            errored = True
        assert errored

def test_Crop():
    reseed()

    base_img = np.array([[0, 0, 0],
                         [0, 1, 0],
                         [0, 0, 0]], dtype=np.uint8)
    base_img = base_img[:, :, np.newaxis]

    images = np.array([base_img])
    images_list = [base_img]

    keypoints = [ia.KeypointsOnImage([ia.Keypoint(x=0, y=0), ia.Keypoint(x=1, y=1), ia.Keypoint(x=2, y=2)], shape=base_img.shape)]

    # test crop by 1 pixel on each side
    crops = [
        (1, 0, 0, 0),
        (0, 1, 0, 0),
        (0, 0, 1, 0),
        (0, 0, 0, 1),
    ]
    for crop in crops:
        top, right, bottom, left = crop
        height, width = base_img.shape[0:2]
        aug = iaa.Crop(px=crop, keep_size=False)
        base_img_cropped = base_img[top:height-bottom, left:width-right, :]
        observed = aug.augment_images(images)
        assert np.array_equal(observed, np.array([base_img_cropped]))

        observed = aug.augment_images(images_list)
        assert array_equal_lists(observed, [base_img_cropped])

        keypoints_moved = [keypoints[0].shift(x=-left, y=-top)]
        observed = aug.augment_keypoints(keypoints)
        assert keypoints_equal(observed, keypoints_moved)

    # test crop by range of pixels
    crops = [
        ((0, 2), 0, 0, 0),
        (0, (0, 2), 0, 0),
        (0, 0, (0, 2), 0),
        (0, 0, 0, (0, 2)),
    ]
    for crop in crops:
        top, right, bottom, left = crop
        height, width = base_img.shape[0:2]
        aug = iaa.Crop(px=crop, keep_size=False)
        aug_det = aug.to_deterministic()

        images_cropped = []
        keypoints_cropped = []
        top_range = top if isinstance(top, tuple) else (top, top)
        right_range = right if isinstance(right, tuple) else (right, right)
        bottom_range = bottom if isinstance(bottom, tuple) else (bottom, bottom)
        left_range = left if isinstance(left, tuple) else (left, left)
        for top_val in sm.xrange(top_range[0], top_range[1]+1):
            for right_val in sm.xrange(right_range[0], right_range[1]+1):
                for bottom_val in sm.xrange(bottom_range[0], bottom_range[1]+1):
                    for left_val in sm.xrange(left_range[0], left_range[1]+1):

                        images_cropped.append(base_img[top_val:height-bottom_val, left_val:width-right_val, :])
                        keypoints_cropped.append(keypoints[0].shift(x=-left_val, y=-top_val))

        movements = []
        movements_det = []
        for i in sm.xrange(100):
            observed = aug.augment_images(images)

            matches = [1 if np.array_equal(observed, np.array([base_img_cropped])) else 0 for base_img_cropped in images_cropped]
            movements.append(np.argmax(np.array(matches)))
            assert any([val == 1 for val in matches])

            observed = aug_det.augment_images(images)
            matches = [1 if np.array_equal(observed, np.array([base_img_cropped])) else 0 for base_img_cropped in images_cropped]
            movements_det.append(np.argmax(np.array(matches)))
            assert any([val == 1 for val in matches])

            observed = aug.augment_images(images_list)
            assert any([array_equal_lists(observed, [base_img_cropped]) for base_img_cropped in images_cropped])

            observed = aug.augment_keypoints(keypoints)
            assert any([keypoints_equal(observed, [kp]) for kp in keypoints_cropped])

        assert len(set(movements)) == 3
        assert len(set(movements_det)) == 1

    # test crop by list of exact pixel values
    crops = [
        ([0, 2], 0, 0, 0),
        (0, [0, 2], 0, 0),
        (0, 0, [0, 2], 0),
        (0, 0, 0, [0, 2]),
    ]
    for crop in crops:
        top, right, bottom, left = crop
        height, width = base_img.shape[0:2]
        aug = iaa.Crop(px=crop, keep_size=False)
        aug_det = aug.to_deterministic()

        images_cropped = []
        keypoints_cropped = []
        top_range = top if isinstance(top, list) else [top]
        right_range = right if isinstance(right, list) else [right]
        bottom_range = bottom if isinstance(bottom, list) else [bottom]
        left_range = left if isinstance(left, list) else [left]
        for top_val in top_range:
            for right_val in right_range:
                for bottom_val in bottom_range:
                    for left_val in left_range:
                        images_cropped.append(base_img[top_val:height-bottom_val, left_val:width-right_val, :])
                        keypoints_cropped.append(keypoints[0].shift(x=-left_val, y=-top_val))

        movements = []
        movements_det = []
        for i in sm.xrange(100):
            observed = aug.augment_images(images)
            matches = [1 if np.array_equal(observed, np.array([base_img_cropped])) else 0 for base_img_cropped in images_cropped]
            movements.append(np.argmax(np.array(matches)))
            assert any([val == 1 for val in matches])

            observed = aug_det.augment_images(images)
            matches = [1 if np.array_equal(observed, np.array([base_img_cropped])) else 0 for base_img_cropped in images_cropped]
            movements_det.append(np.argmax(np.array(matches)))
            assert any([val == 1 for val in matches])

            observed = aug.augment_images(images_list)
            assert any([array_equal_lists(observed, [base_img_cropped]) for base_img_cropped in images_cropped])

            observed = aug.augment_keypoints(keypoints)
            assert any([keypoints_equal(observed, [kp]) for kp in keypoints_cropped])

        assert len(set(movements)) == 2
        assert len(set(movements_det)) == 1

    # TODO
    print("[Note] Crop by percentages is currently not tested.")
    print("[Note] Landmark projection after crop with resize is currently not tested.")

def test_Fliplr():
    reseed()

    base_img = np.array([[0, 0, 1],
                         [0, 0, 1],
                         [0, 1, 1]], dtype=np.uint8)
    base_img = base_img[:, :, np.newaxis]

    base_img_flipped = np.array([[1, 0, 0],
                                 [1, 0, 0],
                                 [1, 1, 0]], dtype=np.uint8)
    base_img_flipped = base_img_flipped[:, :, np.newaxis]

    images = np.array([base_img])
    images_flipped = np.array([base_img_flipped])

    keypoints = [ia.KeypointsOnImage([ia.Keypoint(x=0, y=0), ia.Keypoint(x=1, y=1), ia.Keypoint(x=2, y=2)], shape=base_img.shape)]
    keypoints_flipped = [ia.KeypointsOnImage([ia.Keypoint(x=2, y=0), ia.Keypoint(x=1, y=1), ia.Keypoint(x=0, y=2)], shape=base_img.shape)]

    # 0% chance of flip
    aug = iaa.Fliplr(0)
    aug_det = aug.to_deterministic()

    for _ in sm.xrange(10):
        observed = aug.augment_images(images)
        expected = images
        assert np.array_equal(observed, expected)

        observed = aug_det.augment_images(images)
        expected = images
        assert np.array_equal(observed, expected)

        observed = aug.augment_keypoints(keypoints)
        expected = keypoints
        assert keypoints_equal(observed, expected)

        observed = aug_det.augment_keypoints(keypoints)
        expected = keypoints
        assert keypoints_equal(observed, expected)

    # 100% chance of flip
    aug = iaa.Fliplr(1.0)
    aug_det = aug.to_deterministic()

    for _ in sm.xrange(10):
        observed = aug.augment_images(images)
        expected = images_flipped
        assert np.array_equal(observed, expected)

        observed = aug_det.augment_images(images)
        expected = images_flipped
        assert np.array_equal(observed, expected)

        observed = aug.augment_keypoints(keypoints)
        expected = keypoints_flipped
        assert keypoints_equal(observed, expected)

        observed = aug_det.augment_keypoints(keypoints)
        expected = keypoints_flipped
        assert keypoints_equal(observed, expected)

    # 50% chance of flip
    aug = iaa.Fliplr(0.5)
    aug_det = aug.to_deterministic()

    nb_iterations = 1000
    nb_images_flipped = 0
    nb_images_flipped_det = 0
    nb_keypoints_flipped = 0
    nb_keypoints_flipped_det = 0
    for _ in sm.xrange(nb_iterations):
        observed = aug.augment_images(images)
        if np.array_equal(observed, images_flipped):
            nb_images_flipped += 1

        observed = aug_det.augment_images(images)
        if np.array_equal(observed, images_flipped):
            nb_images_flipped_det += 1

        observed = aug.augment_keypoints(keypoints)
        if keypoints_equal(observed, keypoints_flipped):
            nb_keypoints_flipped += 1

        observed = aug_det.augment_keypoints(keypoints)
        if keypoints_equal(observed, keypoints_flipped):
            nb_keypoints_flipped_det += 1

    assert int(nb_iterations * 0.3) <= nb_images_flipped <= int(nb_iterations * 0.7)
    assert int(nb_iterations * 0.3) <= nb_keypoints_flipped <= int(nb_iterations * 0.7)
    assert nb_images_flipped_det in [0, nb_iterations]
    assert nb_keypoints_flipped_det in [0, nb_iterations]

    # 50% chance of flipped, multiple images, list as input
    images_multi = [base_img, base_img]
    aug = iaa.Fliplr(0.5)
    aug_det = aug.to_deterministic()
    nb_iterations = 1000
    nb_flipped_by_pos = [0] * len(images_multi)
    nb_flipped_by_pos_det = [0] * len(images_multi)
    for _ in sm.xrange(nb_iterations):
        observed = aug.augment_images(images_multi)
        for i in sm.xrange(len(images_multi)):
            if np.array_equal(observed[i], base_img_flipped):
                nb_flipped_by_pos[i] += 1

        observed = aug_det.augment_images(images_multi)
        for i in sm.xrange(len(images_multi)):
            if np.array_equal(observed[i], base_img_flipped):
                nb_flipped_by_pos_det[i] += 1

    for val in nb_flipped_by_pos:
        assert int(nb_iterations * 0.3) <= val <= int(nb_iterations * 0.7)

    for val in nb_flipped_by_pos_det:
        assert val in [0, nb_iterations]

def test_Flipud():
    reseed()

    base_img = np.array([[0, 0, 1],
                         [0, 0, 1],
                         [0, 1, 1]], dtype=np.uint8)
    base_img = base_img[:, :, np.newaxis]

    base_img_flipped = np.array([[0, 1, 1],
                                 [0, 0, 1],
                                 [0, 0, 1]], dtype=np.uint8)
    base_img_flipped = base_img_flipped[:, :, np.newaxis]

    images = np.array([base_img])
    images_flipped = np.array([base_img_flipped])

    keypoints = [ia.KeypointsOnImage([ia.Keypoint(x=0, y=0), ia.Keypoint(x=1, y=1), ia.Keypoint(x=2, y=2)], shape=base_img.shape)]
    keypoints_flipped = [ia.KeypointsOnImage([ia.Keypoint(x=0, y=2), ia.Keypoint(x=1, y=1), ia.Keypoint(x=2, y=0)], shape=base_img.shape)]

    # 0% chance of flip
    aug = iaa.Flipud(0)
    aug_det = aug.to_deterministic()

    for _ in sm.xrange(10):
        observed = aug.augment_images(images)
        expected = images
        assert np.array_equal(observed, expected)

        observed = aug_det.augment_images(images)
        expected = images
        assert np.array_equal(observed, expected)

        observed = aug.augment_keypoints(keypoints)
        expected = keypoints
        assert keypoints_equal(observed, expected)

        observed = aug_det.augment_keypoints(keypoints)
        expected = keypoints
        assert keypoints_equal(observed, expected)

    # 100% chance of flip
    aug = iaa.Flipud(1.0)
    aug_det = aug.to_deterministic()

    for _ in sm.xrange(10):
        observed = aug.augment_images(images)
        expected = images_flipped
        assert np.array_equal(observed, expected)

        observed = aug_det.augment_images(images)
        expected = images_flipped
        assert np.array_equal(observed, expected)

        observed = aug.augment_keypoints(keypoints)
        expected = keypoints_flipped
        assert keypoints_equal(observed, expected)

        observed = aug_det.augment_keypoints(keypoints)
        expected = keypoints_flipped
        assert keypoints_equal(observed, expected)

    # 50% chance of flip
    aug = iaa.Flipud(0.5)
    aug_det = aug.to_deterministic()

    nb_iterations = 1000
    nb_images_flipped = 0
    nb_images_flipped_det = 0
    nb_keypoints_flipped = 0
    nb_keypoints_flipped_det = 0
    for _ in sm.xrange(nb_iterations):
        observed = aug.augment_images(images)
        if np.array_equal(observed, images_flipped):
            nb_images_flipped += 1

        observed = aug_det.augment_images(images)
        if np.array_equal(observed, images_flipped):
            nb_images_flipped_det += 1

        observed = aug.augment_keypoints(keypoints)
        if keypoints_equal(observed, keypoints_flipped):
            nb_keypoints_flipped += 1

        observed = aug_det.augment_keypoints(keypoints)
        if keypoints_equal(observed, keypoints_flipped):
            nb_keypoints_flipped_det += 1

    assert int(nb_iterations * 0.3) <= nb_images_flipped <= int(nb_iterations * 0.7)
    assert int(nb_iterations * 0.3) <= nb_keypoints_flipped <= int(nb_iterations * 0.7)
    assert nb_images_flipped_det in [0, nb_iterations]
    assert nb_keypoints_flipped_det in [0, nb_iterations]

    # 50% chance of flipped, multiple images, list as input
    images_multi = [base_img, base_img]
    aug = iaa.Flipud(0.5)
    aug_det = aug.to_deterministic()
    nb_iterations = 1000
    nb_flipped_by_pos = [0] * len(images_multi)
    nb_flipped_by_pos_det = [0] * len(images_multi)
    for _ in sm.xrange(nb_iterations):
        observed = aug.augment_images(images_multi)
        for i in sm.xrange(len(images_multi)):
            if np.array_equal(observed[i], base_img_flipped):
                nb_flipped_by_pos[i] += 1

        observed = aug_det.augment_images(images_multi)
        for i in sm.xrange(len(images_multi)):
            if np.array_equal(observed[i], base_img_flipped):
                nb_flipped_by_pos_det[i] += 1

    for val in nb_flipped_by_pos:
        assert int(nb_iterations * 0.3) <= val <= int(nb_iterations * 0.7)

    for val in nb_flipped_by_pos_det:
        assert val in [0, nb_iterations]

def test_GaussianBlur():
    reseed()

    base_img = np.array([[0, 0, 0],
                         [0, 255, 0],
                         [0, 0, 0]], dtype=np.uint8)
    base_img = base_img[:, :, np.newaxis]

    images = np.array([base_img])
    images_list = [base_img]
    outer_pixels = ([], [])
    for i in sm.xrange(base_img.shape[0]):
        for j in sm.xrange(base_img.shape[1]):
            if i != j:
                outer_pixels[0].append(i)
                outer_pixels[1].append(j)

    keypoints = [ia.KeypointsOnImage([ia.Keypoint(x=0, y=0), ia.Keypoint(x=1, y=1), ia.Keypoint(x=2, y=2)], shape=base_img.shape)]

    # no blur, shouldnt change anything
    aug = iaa.GaussianBlur(sigma=0)
    aug_det = aug.to_deterministic()

    observed = aug.augment_images(images)
    expected = images
    assert np.array_equal(observed, expected)

    # weak blur of center pixel
    aug = iaa.GaussianBlur(sigma=0.5)
    aug_det = aug.to_deterministic()

    #np.set_printoptions(formatter={'float_kind': lambda x: "%.6f" % x})
    #from scipy import ndimage
    #images2 = np.copy(images).astype(np.float32)
    #images2[0, ...] = ndimage.gaussian_filter(images2[0, ...], 0.4)
    #print(images2)

    # images as numpy array
    observed = aug.augment_images(images)
    assert 100 < observed[0][1, 1] < 255
    assert (observed[0][outer_pixels[0], outer_pixels[1]] > 0).all()
    assert (observed[0][outer_pixels[0], outer_pixels[1]] < 50).all()

    observed = aug_det.augment_images(images)
    assert 100 < observed[0][1, 1] < 255
    assert (observed[0][outer_pixels[0], outer_pixels[1]] > 0).all()
    assert (observed[0][outer_pixels[0], outer_pixels[1]] < 50).all()

    # images as list
    observed = aug.augment_images(images_list)
    assert 100 < observed[0][1, 1] < 255
    assert (observed[0][outer_pixels[0], outer_pixels[1]] > 0).all()
    assert (observed[0][outer_pixels[0], outer_pixels[1]] < 50).all()

    observed = aug_det.augment_images(images_list)
    assert 100 < observed[0][1, 1] < 255
    assert (observed[0][outer_pixels[0], outer_pixels[1]] > 0).all()
    assert (observed[0][outer_pixels[0], outer_pixels[1]] < 50).all()

    # keypoints shouldnt be changed
    observed = aug.augment_keypoints(keypoints)
    expected = keypoints
    assert keypoints_equal(observed, expected)

    observed = aug_det.augment_keypoints(keypoints)
    expected = keypoints
    assert keypoints_equal(observed, expected)

    # varying blur sigmas
    aug = iaa.GaussianBlur(sigma=(0, 1))
    aug_det = aug.to_deterministic()

    last_aug = None
    last_aug_det = None
    nb_changed_aug = 0
    nb_changed_aug_det = 0
    nb_iterations = 1000
    for i in sm.xrange(nb_iterations):
        observed_aug = aug.augment_images(images)
        observed_aug_det = aug_det.augment_images(images)
        if i == 0:
            last_aug = observed_aug
            last_aug_det = observed_aug_det
        else:
            if not np.array_equal(observed_aug, last_aug):
                nb_changed_aug += 1
            if not np.array_equal(observed_aug_det, last_aug_det):
                nb_changed_aug_det += 1
            last_aug = observed_aug
            last_aug_det = observed_aug_det
    assert nb_changed_aug >= int(nb_iterations * 0.8)
    assert nb_changed_aug_det == 0

def test_AdditiveGaussianNoise():
    reseed()

    #base_img = np.array([[128, 128, 128],
    #                     [128, 128, 128],
    #                     [128, 128, 128]], dtype=np.uint8)
    base_img = np.ones((16, 16, 1), dtype=np.uint8) * 128
    #base_img = base_img[:, :, np.newaxis]

    images = np.array([base_img])
    images_list = [base_img]

    keypoints = [ia.KeypointsOnImage([ia.Keypoint(x=0, y=0), ia.Keypoint(x=1, y=1), ia.Keypoint(x=2, y=2)], shape=base_img.shape)]

    # no noise, shouldnt change anything
    aug = iaa.AdditiveGaussianNoise(loc=0, scale=0)
    aug_det = aug.to_deterministic()

    observed = aug.augment_images(images)
    expected = images
    assert np.array_equal(observed, expected)

    # zero-centered noise
    aug = iaa.AdditiveGaussianNoise(loc=0, scale=0.2)
    aug_det = aug.to_deterministic()

    observed = aug.augment_images(images)
    assert not np.array_equal(observed, images)

    observed = aug_det.augment_images(images)
    assert not np.array_equal(observed, images)

    observed = aug.augment_images(images_list)
    assert not array_equal_lists(observed, images_list)

    observed = aug_det.augment_images(images_list)
    assert not array_equal_lists(observed, images_list)

    observed = aug.augment_keypoints(keypoints)
    assert keypoints_equal(observed, keypoints)

    observed = aug_det.augment_keypoints(keypoints)
    assert keypoints_equal(observed, keypoints)

    # std correct?
    aug = iaa.AdditiveGaussianNoise(loc=0, scale=0.2 * 255)
    aug_det = aug.to_deterministic()
    images = np.ones((1, 1, 1, 1), dtype=np.uint8) * 128
    nb_iterations = 10000
    values = []
    for i in sm.xrange(nb_iterations):
        images_aug = aug.augment_images(images)
        values.append(images_aug[0, 0, 0, 0])
    values = np.array(values)
    assert np.min(values) == 0
    assert 0.1 < np.std(values) / 255.0 < 0.4

    # non-zero loc
    aug = iaa.AdditiveGaussianNoise(loc=0.25 * 255, scale=0.01 * 255)
    aug_det = aug.to_deterministic()
    images = np.ones((1, 1, 1, 1), dtype=np.uint8) * 128
    nb_iterations = 10000
    values = []
    for i in sm.xrange(nb_iterations):
        images_aug = aug.augment_images(images)
        values.append(images_aug[0, 0, 0, 0] - 128)
    values = np.array(values)
    assert 54 < np.average(values) < 74 # loc=0.25 should be around 255*0.25=64 average

    # varying locs
    aug = iaa.AdditiveGaussianNoise(loc=(0, 0.5 * 255), scale=0.0001 * 255)
    aug_det = aug.to_deterministic()
    images = np.ones((1, 1, 1, 1), dtype=np.uint8) * 128
    last_aug = None
    last_aug_det = None
    nb_changed_aug = 0
    nb_changed_aug_det = 0
    nb_iterations = 1000
    for i in sm.xrange(nb_iterations):
        observed_aug = aug.augment_images(images)
        observed_aug_det = aug_det.augment_images(images)
        if i == 0:
            last_aug = observed_aug
            last_aug_det = observed_aug_det
        else:
            if not np.array_equal(observed_aug, last_aug):
                nb_changed_aug += 1
            if not np.array_equal(observed_aug_det, last_aug_det):
                nb_changed_aug_det += 1
            last_aug = observed_aug
            last_aug_det = observed_aug_det
    assert nb_changed_aug >= int(nb_iterations * 0.95)
    assert nb_changed_aug_det == 0

    # varying stds
    aug = iaa.AdditiveGaussianNoise(loc=0, scale=(0.01 * 255, 0.2 * 255))
    aug_det = aug.to_deterministic()
    images = np.ones((1, 1, 1, 1), dtype=np.uint8) * 128
    last_aug = None
    last_aug_det = None
    nb_changed_aug = 0
    nb_changed_aug_det = 0
    nb_iterations = 1000
    for i in sm.xrange(nb_iterations):
        observed_aug = aug.augment_images(images)
        observed_aug_det = aug_det.augment_images(images)
        if i == 0:
            last_aug = observed_aug
            last_aug_det = observed_aug_det
        else:
            if not np.array_equal(observed_aug, last_aug):
                nb_changed_aug += 1
            if not np.array_equal(observed_aug_det, last_aug_det):
                nb_changed_aug_det += 1
            last_aug = observed_aug
            last_aug_det = observed_aug_det
    assert nb_changed_aug >= int(nb_iterations * 0.95)
    assert nb_changed_aug_det == 0


#def test_MultiplicativeGaussianNoise():
#    pass

#def test_ReplacingGaussianNoise():
#    pass

def test_Dropout():
    reseed()

    base_img = np.ones((512, 512, 1), dtype=np.uint8) * 255

    images = np.array([base_img])
    images_list = [base_img]

    keypoints = [ia.KeypointsOnImage([ia.Keypoint(x=0, y=0), ia.Keypoint(x=1, y=1), ia.Keypoint(x=2, y=2)], shape=base_img.shape)]

    # no dropout, shouldnt change anything
    aug = iaa.Dropout(p=0)
    observed = aug.augment_images(images)
    expected = images
    assert np.array_equal(observed, expected)

    observed = aug.augment_images(images_list)
    expected = images_list
    assert array_equal_lists(observed, expected)

    # 100% dropout, should drop everything
    aug = iaa.Dropout(p=1.0)
    observed = aug.augment_images(images)
    expected = np.zeros((1, 512, 512, 1), dtype=np.uint8)
    assert np.array_equal(observed, expected)

    observed = aug.augment_images(images_list)
    expected = [np.zeros((512, 512, 1), dtype=np.uint8)]
    assert array_equal_lists(observed, expected)

    # 50% dropout
    aug = iaa.Dropout(p=0.5)
    aug_det = aug.to_deterministic()

    observed = aug.augment_images(images)
    assert not np.array_equal(observed, images)
    percent_nonzero = len(observed.flatten().nonzero()[0]) / (base_img.shape[0] * base_img.shape[1] * base_img.shape[2])
    assert 0.35 <= (1 - percent_nonzero) <= 0.65

    observed = aug_det.augment_images(images)
    assert not np.array_equal(observed, images)
    percent_nonzero = len(observed.flatten().nonzero()[0]) / (base_img.shape[0] * base_img.shape[1] * base_img.shape[2])
    assert 0.35 <= (1 - percent_nonzero) <= 0.65

    observed = aug.augment_images(images_list)
    assert not array_equal_lists(observed, images_list)
    percent_nonzero = len(observed[0].flatten().nonzero()[0]) / (base_img.shape[0] * base_img.shape[1] * base_img.shape[2])
    assert 0.35 <= (1 - percent_nonzero) <= 0.65

    observed = aug_det.augment_images(images_list)
    assert not array_equal_lists(observed, images_list)
    percent_nonzero = len(observed[0].flatten().nonzero()[0]) / (base_img.shape[0] * base_img.shape[1] * base_img.shape[2])
    assert 0.35 <= (1 - percent_nonzero) <= 0.65

    observed = aug.augment_keypoints(keypoints)
    assert keypoints_equal(observed, keypoints)

    observed = aug_det.augment_keypoints(keypoints)
    assert keypoints_equal(observed, keypoints)

    # varying p
    aug = iaa.Dropout(p=(0.0, 1.0))
    aug_det = aug.to_deterministic()
    images = np.ones((1, 8, 8, 1), dtype=np.uint8) * 255
    last_aug = None
    last_aug_det = None
    nb_changed_aug = 0
    nb_changed_aug_det = 0
    nb_iterations = 1000
    for i in sm.xrange(nb_iterations):
        observed_aug = aug.augment_images(images)
        observed_aug_det = aug_det.augment_images(images)
        if i == 0:
            last_aug = observed_aug
            last_aug_det = observed_aug_det
        else:
            if not np.array_equal(observed_aug, last_aug):
                nb_changed_aug += 1
            if not np.array_equal(observed_aug_det, last_aug_det):
                nb_changed_aug_det += 1
            last_aug = observed_aug
            last_aug_det = observed_aug_det
    assert nb_changed_aug >= int(nb_iterations * 0.95)
    assert nb_changed_aug_det == 0

def test_Multiply():
    reseed()

    base_img = np.ones((3, 3, 1), dtype=np.uint8) * 100
    images = np.array([base_img])
    images_list = [base_img]
    keypoints = [ia.KeypointsOnImage([ia.Keypoint(x=0, y=0), ia.Keypoint(x=1, y=1), ia.Keypoint(x=2, y=2)], shape=base_img.shape)]

    # no multiply, shouldnt change anything
    aug = iaa.Multiply(mul=1.0)
    aug_det = aug.to_deterministic()

    observed = aug.augment_images(images)
    expected = images
    assert np.array_equal(observed, expected)

    observed = aug.augment_images(images_list)
    expected = images_list
    assert array_equal_lists(observed, expected)

    observed = aug_det.augment_images(images)
    expected = images
    assert np.array_equal(observed, expected)

    observed = aug_det.augment_images(images_list)
    expected = images_list
    assert array_equal_lists(observed, expected)

    # multiply >1.0
    aug = iaa.Multiply(mul=1.2)
    aug_det = aug.to_deterministic()

    observed = aug.augment_images(images)
    expected = np.ones((1, 3, 3, 1), dtype=np.uint8) * 120
    assert np.array_equal(observed, expected)

    observed = aug.augment_images(images_list)
    expected = [np.ones((3, 3, 1), dtype=np.uint8) * 120]
    assert array_equal_lists(observed, expected)

    observed = aug_det.augment_images(images)
    expected = np.ones((1, 3, 3, 1), dtype=np.uint8) * 120
    assert np.array_equal(observed, expected)

    observed = aug_det.augment_images(images_list)
    expected = [np.ones((3, 3, 1), dtype=np.uint8) * 120]
    assert array_equal_lists(observed, expected)

    # multiply <1.0
    aug = iaa.Multiply(mul=0.8)
    aug_det = aug.to_deterministic()

    observed = aug.augment_images(images)
    expected = np.ones((1, 3, 3, 1), dtype=np.uint8) * 80
    assert np.array_equal(observed, expected)

    observed = aug.augment_images(images_list)
    expected = [np.ones((3, 3, 1), dtype=np.uint8) * 80]
    assert array_equal_lists(observed, expected)

    observed = aug_det.augment_images(images)
    expected = np.ones((1, 3, 3, 1), dtype=np.uint8) * 80
    assert np.array_equal(observed, expected)

    observed = aug_det.augment_images(images_list)
    expected = [np.ones((3, 3, 1), dtype=np.uint8) * 80]
    assert array_equal_lists(observed, expected)

    # keypoints shouldnt be changed
    aug = iaa.Multiply(mul=1.2)
    aug_det = iaa.Multiply(mul=1.2)
    observed = aug.augment_keypoints(keypoints)
    expected = keypoints
    assert keypoints_equal(observed, expected)

    observed = aug_det.augment_keypoints(keypoints)
    expected = keypoints
    assert keypoints_equal(observed, expected)

    # varying multiply factors
    aug = iaa.Multiply(mul=(0, 2.0))
    aug_det = aug.to_deterministic()

    last_aug = None
    last_aug_det = None
    nb_changed_aug = 0
    nb_changed_aug_det = 0
    nb_iterations = 1000
    for i in sm.xrange(nb_iterations):
        observed_aug = aug.augment_images(images)
        observed_aug_det = aug_det.augment_images(images)
        if i == 0:
            last_aug = observed_aug
            last_aug_det = observed_aug_det
        else:
            if not np.array_equal(observed_aug, last_aug):
                nb_changed_aug += 1
            if not np.array_equal(observed_aug_det, last_aug_det):
                nb_changed_aug_det += 1
            last_aug = observed_aug
            last_aug_det = observed_aug_det
    assert nb_changed_aug >= int(nb_iterations * 0.95)
    assert nb_changed_aug_det == 0

def test_Affine():
    reseed()

    base_img = np.array([[0, 0, 0],
                         [0, 255, 0],
                         [0, 0, 0]], dtype=np.uint8)
    base_img = base_img[:, :, np.newaxis]

    images = np.array([base_img])
    images_list = [base_img]
    outer_pixels = ([], [])
    for i in sm.xrange(base_img.shape[0]):
        for j in sm.xrange(base_img.shape[1]):
            if i != j:
                outer_pixels[0].append(i)
                outer_pixels[1].append(j)

    keypoints = [ia.KeypointsOnImage([ia.Keypoint(x=0, y=0), ia.Keypoint(x=1, y=1), ia.Keypoint(x=2, y=2)], shape=base_img.shape)]

    # no translation/scale/rotate/shear, shouldnt change nothing
    aug = iaa.Affine(scale=1.0, translate_px=0, rotate=0, shear=0)
    aug_det = aug.to_deterministic()

    observed = aug.augment_images(images)
    expected = images
    assert np.array_equal(observed, expected)

    observed = aug_det.augment_images(images)
    expected = images
    assert np.array_equal(observed, expected)

    observed = aug.augment_images(images_list)
    expected = images_list
    assert array_equal_lists(observed, expected)

    observed = aug_det.augment_images(images_list)
    expected = images_list
    assert array_equal_lists(observed, expected)

    observed = aug.augment_keypoints(keypoints)
    expected = keypoints
    assert keypoints_equal(observed, expected)

    observed = aug_det.augment_keypoints(keypoints)
    expected = keypoints
    assert keypoints_equal(observed, expected)

    # ---------------------
    # scale
    # ---------------------
    # zoom in
    aug = iaa.Affine(scale=1.75, translate_px=0, rotate=0, shear=0)
    aug_det = aug.to_deterministic()

    observed = aug.augment_images(images)
    assert observed[0][1, 1] > 250
    assert (observed[0][outer_pixels[0], outer_pixels[1]] > 20).all()
    assert (observed[0][outer_pixels[0], outer_pixels[1]] < 150).all()

    observed = aug_det.augment_images(images)
    assert observed[0][1, 1] > 250
    assert (observed[0][outer_pixels[0], outer_pixels[1]] > 20).all()
    assert (observed[0][outer_pixels[0], outer_pixels[1]] < 150).all()

    observed = aug.augment_images(images_list)
    assert observed[0][1, 1] > 250
    assert (observed[0][outer_pixels[0], outer_pixels[1]] > 20).all()
    assert (observed[0][outer_pixels[0], outer_pixels[1]] < 150).all()

    observed = aug_det.augment_images(images_list)
    assert observed[0][1, 1] > 250
    assert (observed[0][outer_pixels[0], outer_pixels[1]] > 20).all()
    assert (observed[0][outer_pixels[0], outer_pixels[1]] < 150).all()

    observed = aug.augment_keypoints(keypoints)
    assert observed[0].keypoints[0].x < 0
    assert observed[0].keypoints[0].y < 0
    assert observed[0].keypoints[1].x == 1
    assert observed[0].keypoints[1].y == 1
    assert observed[0].keypoints[2].x > 2
    assert observed[0].keypoints[2].y > 2

    observed = aug_det.augment_keypoints(keypoints)
    assert observed[0].keypoints[0].x < 0
    assert observed[0].keypoints[0].y < 0
    assert observed[0].keypoints[1].x == 1
    assert observed[0].keypoints[1].y == 1
    assert observed[0].keypoints[2].x > 2
    assert observed[0].keypoints[2].y > 2

    # zoom in only on x axis
    aug = iaa.Affine(scale={"x": 1.75, "y": 1.0}, translate_px=0, rotate=0, shear=0)
    aug_det = aug.to_deterministic()

    observed = aug.augment_images(images)
    assert observed[0][1, 1] > 250
    assert (observed[0][[1, 1], [0, 2]] > 20).all()
    assert (observed[0][[1, 1], [0, 2]] < 150).all()
    assert (observed[0][0, :] < 5).all()
    assert (observed[0][2, :] < 5).all()

    observed = aug_det.augment_images(images)
    assert observed[0][1, 1] > 250
    assert (observed[0][[1, 1], [0, 2]] > 20).all()
    assert (observed[0][[1, 1], [0, 2]] < 150).all()
    assert (observed[0][0, :] < 5).all()
    assert (observed[0][2, :] < 5).all()

    observed = aug.augment_images(images_list)
    assert observed[0][1, 1] > 250
    assert (observed[0][[1, 1], [0, 2]] > 20).all()
    assert (observed[0][[1, 1], [0, 2]] < 150).all()
    assert (observed[0][0, :] < 5).all()
    assert (observed[0][2, :] < 5).all()

    observed = aug_det.augment_images(images_list)
    assert observed[0][1, 1] > 250
    assert (observed[0][[1, 1], [0, 2]] > 20).all()
    assert (observed[0][[1, 1], [0, 2]] < 150).all()
    assert (observed[0][0, :] < 5).all()
    assert (observed[0][2, :] < 5).all()

    observed = aug.augment_keypoints(keypoints)
    assert observed[0].keypoints[0].x < 0
    assert observed[0].keypoints[0].y == 0
    assert observed[0].keypoints[1].x == 1
    assert observed[0].keypoints[1].y == 1
    assert observed[0].keypoints[2].x > 2
    assert observed[0].keypoints[2].y == 2

    observed = aug_det.augment_keypoints(keypoints)
    assert observed[0].keypoints[0].x < 0
    assert observed[0].keypoints[0].y == 0
    assert observed[0].keypoints[1].x == 1
    assert observed[0].keypoints[1].y == 1
    assert observed[0].keypoints[2].x > 2
    assert observed[0].keypoints[2].y == 2

    # zoom in only on y axis
    aug = iaa.Affine(scale={"x": 1.0, "y": 1.75}, translate_px=0, rotate=0, shear=0)
    aug_det = aug.to_deterministic()

    observed = aug.augment_images(images)
    assert observed[0][1, 1] > 250
    assert (observed[0][[0, 2], [1, 1]] > 20).all()
    assert (observed[0][[0, 2], [1, 1]] < 150).all()
    assert (observed[0][:, 0] < 5).all()
    assert (observed[0][:, 2] < 5).all()

    observed = aug_det.augment_images(images)
    assert observed[0][1, 1] > 250
    assert (observed[0][[0, 2], [1, 1]] > 20).all()
    assert (observed[0][[0, 2], [1, 1]] < 150).all()
    assert (observed[0][:, 0] < 5).all()
    assert (observed[0][:, 2] < 5).all()

    observed = aug.augment_images(images_list)
    assert observed[0][1, 1] > 250
    assert (observed[0][[0, 2], [1, 1]] > 20).all()
    assert (observed[0][[0, 2], [1, 1]] < 150).all()
    assert (observed[0][:, 0] < 5).all()
    assert (observed[0][:, 2] < 5).all()

    observed = aug_det.augment_images(images_list)
    assert observed[0][1, 1] > 250
    assert (observed[0][[0, 2], [1, 1]] > 20).all()
    assert (observed[0][[0, 2], [1, 1]] < 150).all()
    assert (observed[0][:, 0] < 5).all()
    assert (observed[0][:, 2] < 5).all()

    observed = aug.augment_keypoints(keypoints)
    assert observed[0].keypoints[0].x == 0
    assert observed[0].keypoints[0].y < 0
    assert observed[0].keypoints[1].x == 1
    assert observed[0].keypoints[1].y == 1
    assert observed[0].keypoints[2].x == 2
    assert observed[0].keypoints[2].y > 2

    observed = aug_det.augment_keypoints(keypoints)
    assert observed[0].keypoints[0].x == 0
    assert observed[0].keypoints[0].y < 0
    assert observed[0].keypoints[1].x == 1
    assert observed[0].keypoints[1].y == 1
    assert observed[0].keypoints[2].x == 2
    assert observed[0].keypoints[2].y > 2

    # zoom out
    # this one uses a 4x4 area of all 255, which is zoomed out to a 4x4 area
    # in which the center 2x2 area is 255
    # zoom in should probably be adapted to this style
    # no seperate tests here for x/y axis, should work fine if zoom in works with that
    aug = iaa.Affine(scale=0.49, translate_px=0, rotate=0, shear=0)
    aug_det = aug.to_deterministic()

    image = np.ones((4, 4, 1), dtype=np.uint8) * 255
    images = np.array([image])
    images_list = [image]
    outer_pixels = ([], [])
    for y in sm.xrange(4):
        xs = sm.xrange(4) if y in [0, 3] else [0, 3]
        for x in xs:
            outer_pixels[0].append(y)
            outer_pixels[1].append(x)
    inner_pixels = ([1, 1, 2, 2], [1, 2, 1, 2])
    keypoints = [ia.KeypointsOnImage([ia.Keypoint(x=0, y=0), ia.Keypoint(x=3, y=0), ia.Keypoint(x=0, y=3), ia.Keypoint(x=3, y=3)], shape=base_img.shape)]
    keypoints_aug = [ia.KeypointsOnImage([ia.Keypoint(x=1, y=1), ia.Keypoint(x=2, y=1), ia.Keypoint(x=1, y=2), ia.Keypoint(x=2, y=2)], shape=base_img.shape)]

    observed = aug.augment_images(images)
    assert (observed[0][outer_pixels] < 25).all()
    assert (observed[0][inner_pixels] > 200).all()

    observed = aug_det.augment_images(images)
    assert (observed[0][outer_pixels] < 25).all()
    assert (observed[0][inner_pixels] > 200).all()

    observed = aug.augment_images(images_list)
    assert (observed[0][outer_pixels] < 25).all()
    assert (observed[0][inner_pixels] > 200).all()

    observed = aug_det.augment_images(images_list)
    assert (observed[0][outer_pixels] < 25).all()
    assert (observed[0][inner_pixels] > 200).all()

    observed = aug.augment_keypoints(keypoints)
    assert keypoints_equal(observed, keypoints_aug)

    observed = aug_det.augment_keypoints(keypoints)
    assert keypoints_equal(observed, keypoints_aug)

    # varying scales
    aug = iaa.Affine(scale={"x": (0.5, 1.5), "y": (0.5, 1.5)}, translate_px=0, rotate=0, shear=0)
    aug_det = aug.to_deterministic()

    image = np.array([[0, 0, 0, 0, 0],
                      [0, 1, 1, 1, 0],
                      [0, 1, 2, 1, 0],
                      [0, 1, 1, 1, 0],
                      [0, 0, 0, 0, 0]], dtype=np.uint8) * 100
    image = image[:, :, np.newaxis]
    images_list = [image]
    images = np.array([image])

    last_aug = None
    last_aug_det = None
    nb_changed_aug = 0
    nb_changed_aug_det = 0
    nb_iterations = 1000
    for i in sm.xrange(nb_iterations):
        observed_aug = aug.augment_images(images)
        observed_aug_det = aug_det.augment_images(images)
        if i == 0:
            last_aug = observed_aug
            last_aug_det = observed_aug_det
        else:
            if not np.array_equal(observed_aug, last_aug):
                nb_changed_aug += 1
            if not np.array_equal(observed_aug_det, last_aug_det):
                nb_changed_aug_det += 1
            last_aug = observed_aug
            last_aug_det = observed_aug_det
    assert nb_changed_aug >= int(nb_iterations * 0.8)
    assert nb_changed_aug_det == 0

    # ---------------------
    # translate
    # ---------------------
    # move one pixel to the right
    aug = iaa.Affine(scale=1.0, translate_px={"x": 1, "y": 0}, rotate=0, shear=0)
    aug_det = aug.to_deterministic()

    image = np.zeros((3, 3, 1), dtype=np.uint8)
    image_aug = np.copy(image)
    image[1, 1] = 255
    image_aug[1, 2] = 255
    images = np.array([image])
    images_aug = np.array([image_aug])
    images_list = [image]
    images_aug_list = [image_aug]
    keypoints = [ia.KeypointsOnImage([ia.Keypoint(x=1, y=1)], shape=base_img.shape)]
    keypoints_aug = [ia.KeypointsOnImage([ia.Keypoint(x=2, y=1)], shape=base_img.shape)]

    observed = aug.augment_images(images)
    assert np.array_equal(observed, images_aug)

    observed = aug_det.augment_images(images)
    assert np.array_equal(observed, images_aug)

    observed = aug.augment_images(images_list)
    assert array_equal_lists(observed, images_aug_list)

    observed = aug_det.augment_images(images_list)
    assert array_equal_lists(observed, images_aug_list)

    observed = aug.augment_keypoints(keypoints)
    assert keypoints_equal(observed, keypoints_aug)

    observed = aug_det.augment_keypoints(keypoints)
    assert keypoints_equal(observed, keypoints_aug)

    # move one pixel to the bottom
    aug = iaa.Affine(scale=1.0, translate_px={"x": 0, "y": 1}, rotate=0, shear=0)
    aug_det = aug.to_deterministic()

    image = np.zeros((3, 3, 1), dtype=np.uint8)
    image_aug = np.copy(image)
    image[1, 1] = 255
    image_aug[2, 1] = 255
    images = np.array([image])
    images_aug = np.array([image_aug])
    images_list = [image]
    images_aug_list = [image_aug]
    keypoints = [ia.KeypointsOnImage([ia.Keypoint(x=1, y=1)], shape=base_img.shape)]
    keypoints_aug = [ia.KeypointsOnImage([ia.Keypoint(x=1, y=2)], shape=base_img.shape)]

    observed = aug.augment_images(images)
    assert np.array_equal(observed, images_aug)

    observed = aug_det.augment_images(images)
    assert np.array_equal(observed, images_aug)

    observed = aug.augment_images(images_list)
    assert array_equal_lists(observed, images_aug_list)

    observed = aug_det.augment_images(images_list)
    assert array_equal_lists(observed, images_aug_list)

    observed = aug.augment_keypoints(keypoints)
    assert keypoints_equal(observed, keypoints_aug)

    observed = aug_det.augment_keypoints(keypoints)
    assert keypoints_equal(observed, keypoints_aug)

    # move 33% (one pixel) to the right
    aug = iaa.Affine(scale=1.0, translate_percent={"x": 0.3333, "y": 0}, rotate=0, shear=0)
    aug_det = aug.to_deterministic()

    image = np.zeros((3, 3, 1), dtype=np.uint8)
    image_aug = np.copy(image)
    image[1, 1] = 255
    image_aug[1, 2] = 255
    images = np.array([image])
    images_aug = np.array([image_aug])
    images_list = [image]
    images_aug_list = [image_aug]
    keypoints = [ia.KeypointsOnImage([ia.Keypoint(x=1, y=1)], shape=base_img.shape)]
    keypoints_aug = [ia.KeypointsOnImage([ia.Keypoint(x=2, y=1)], shape=base_img.shape)]

    observed = aug.augment_images(images)
    assert np.array_equal(observed, images_aug)

    observed = aug_det.augment_images(images)
    assert np.array_equal(observed, images_aug)

    observed = aug.augment_images(images_list)
    assert array_equal_lists(observed, images_aug_list)

    observed = aug_det.augment_images(images_list)
    assert array_equal_lists(observed, images_aug_list)

    observed = aug.augment_keypoints(keypoints)
    assert keypoints_equal(observed, keypoints_aug)

    observed = aug_det.augment_keypoints(keypoints)
    assert keypoints_equal(observed, keypoints_aug)

    # move 33% (one pixel) to the bottom
    aug = iaa.Affine(scale=1.0, translate_percent={"x": 0, "y": 0.3333}, rotate=0, shear=0)
    aug_det = aug.to_deterministic()

    image = np.zeros((3, 3, 1), dtype=np.uint8)
    image_aug = np.copy(image)
    image[1, 1] = 255
    image_aug[2, 1] = 255
    images = np.array([image])
    images_aug = np.array([image_aug])
    images_list = [image]
    images_aug_list = [image_aug]
    keypoints = [ia.KeypointsOnImage([ia.Keypoint(x=1, y=1)], shape=base_img.shape)]
    keypoints_aug = [ia.KeypointsOnImage([ia.Keypoint(x=1, y=2)], shape=base_img.shape)]

    observed = aug.augment_images(images)
    assert np.array_equal(observed, images_aug)

    observed = aug_det.augment_images(images)
    assert np.array_equal(observed, images_aug)

    observed = aug.augment_images(images_list)
    assert array_equal_lists(observed, images_aug_list)

    observed = aug_det.augment_images(images_list)
    assert array_equal_lists(observed, images_aug_list)

    observed = aug.augment_keypoints(keypoints)
    assert keypoints_equal(observed, keypoints_aug)

    observed = aug_det.augment_keypoints(keypoints)
    assert keypoints_equal(observed, keypoints_aug)

    # 0-1px to left/right and 0-1px to top/bottom
    aug = iaa.Affine(scale=1.0, translate_px={"x": (-1, 1), "y": (-1, 1)}, rotate=0, shear=0)
    aug_det = aug.to_deterministic()
    last_aug = None
    last_aug_det = None
    nb_changed_aug = 0
    nb_changed_aug_det = 0
    nb_iterations = 1000
    centers_aug = np.copy(image).astype(np.int32) * 0
    centers_aug_det = np.copy(image).astype(np.int32) * 0
    for i in sm.xrange(nb_iterations):
        observed_aug = aug.augment_images(images)
        observed_aug_det = aug_det.augment_images(images)
        if i == 0:
            last_aug = observed_aug
            last_aug_det = observed_aug_det
        else:
            if not np.array_equal(observed_aug, last_aug):
                nb_changed_aug += 1
            if not np.array_equal(observed_aug_det, last_aug_det):
                nb_changed_aug_det += 1
            last_aug = observed_aug
            last_aug_det = observed_aug_det

        assert len(observed_aug[0].nonzero()[0]) == 1
        assert len(observed_aug_det[0].nonzero()[0]) == 1
        centers_aug += (observed_aug[0] > 0)
        centers_aug_det += (observed_aug_det[0] > 0)

    assert nb_changed_aug >= int(nb_iterations * 0.7)
    assert nb_changed_aug_det == 0
    assert (centers_aug > int(nb_iterations * (1/9 * 0.6))).all()
    assert (centers_aug < int(nb_iterations * (1/9 * 1.4))).all()

    # ---------------------
    # rotate
    # ---------------------
    # rotate by 45 degrees
    aug = iaa.Affine(scale=1.0, translate_px=0, rotate=90, shear=0)
    aug_det = aug.to_deterministic()

    image = np.zeros((3, 3, 1), dtype=np.uint8)
    image_aug = np.copy(image)
    image[1, :] = 255
    image_aug[0, 1] = 255
    image_aug[1, 1] = 255
    image_aug[2, 1] = 255
    images = np.array([image])
    images_aug = np.array([image_aug])
    images_list = [image]
    images_aug_list = [image_aug]
    keypoints = [ia.KeypointsOnImage([ia.Keypoint(x=0, y=1), ia.Keypoint(x=1, y=1), ia.Keypoint(x=2, y=1)], shape=base_img.shape)]
    keypoints_aug = [ia.KeypointsOnImage([ia.Keypoint(x=1, y=0), ia.Keypoint(x=1, y=1), ia.Keypoint(x=1, y=2)], shape=base_img.shape)]

    observed = aug.augment_images(images)
    observed[observed >= 100] = 255
    observed[observed < 100] = 0
    assert np.array_equal(observed, images_aug)

    observed = aug_det.augment_images(images)
    observed[observed >= 100] = 255
    observed[observed < 100] = 0
    assert np.array_equal(observed, images_aug)

    observed = aug.augment_images(images_list)
    observed[0][observed[0] >= 100] = 255
    observed[0][observed[0] < 100] = 0
    assert array_equal_lists(observed, images_aug_list)

    observed = aug_det.augment_images(images_list)
    observed[0][observed[0] >= 100] = 255
    observed[0][observed[0] < 100] = 0
    assert array_equal_lists(observed, images_aug_list)

    observed = aug.augment_keypoints(keypoints)
    assert keypoints_equal(observed, keypoints_aug)

    observed = aug_det.augment_keypoints(keypoints)
    assert keypoints_equal(observed, keypoints_aug)

    # random rotation 0-364 degrees
    aug = iaa.Affine(scale=1.0, translate_px=0, rotate=(0, 364), shear=0)
    aug_det = aug.to_deterministic()
    last_aug = None
    last_aug_det = None
    nb_changed_aug = 0
    nb_changed_aug_det = 0
    nb_iterations = 1000
    pixels_sums_aug = np.copy(image).astype(np.int32) * 0
    pixels_sums_aug_det = np.copy(image).astype(np.int32) * 0
    for i in sm.xrange(nb_iterations):
        observed_aug = aug.augment_images(images)
        observed_aug_det = aug_det.augment_images(images)
        if i == 0:
            last_aug = observed_aug
            last_aug_det = observed_aug_det
        else:
            if not np.array_equal(observed_aug, last_aug):
                nb_changed_aug += 1
            if not np.array_equal(observed_aug_det, last_aug_det):
                nb_changed_aug_det += 1
            last_aug = observed_aug
            last_aug_det = observed_aug_det

        #assert len(observed_aug[0].nonzero()[0]) == 1
        #assert len(observed_aug_det[0].nonzero()[0]) == 1
        pixels_sums_aug += (observed_aug[0] > 100)
        pixels_sums_aug_det += (observed_aug_det[0] > 100)

    assert nb_changed_aug >= int(nb_iterations * 0.9)
    assert nb_changed_aug_det == 0
    # center pixel, should always be white when rotating line around center
    assert pixels_sums_aug[1, 1] > (nb_iterations * 0.98)
    assert pixels_sums_aug[1, 1] < (nb_iterations * 1.02)

    # outer pixels, should sometimes be white
    # the values here had to be set quite tolerant, the middle pixels at top/left/bottom/right get more activation than expected
    outer_pixels = ([0, 0, 0, 1, 1, 2, 2, 2], [0, 1, 2, 0, 2, 0, 1, 2])
    assert (pixels_sums_aug[outer_pixels] > int(nb_iterations * (2/8 * 0.4))).all()
    assert (pixels_sums_aug[outer_pixels] < int(nb_iterations * (2/8 * 2.0))).all()

    # ---------------------
    # shear
    # ---------------------
    # TODO
    print("[Note] There is currently no test for shear in test_Affine().")

    # ---------------------
    # cval
    # ---------------------
    aug = iaa.Affine(scale=1.0, translate_px=100, rotate=0, shear=0, cval=128)
    aug_det = aug.to_deterministic()

    image = np.ones((3, 3, 1), dtype=np.uint8) * 255
    image_aug = np.copy(image)
    images = np.array([image])
    images_list = [image]

    observed = aug.augment_images(images)
    assert (observed[0] > 128 - 30).all()
    assert (observed[0] < 128 + 30).all()

    observed = aug_det.augment_images(images)
    assert (observed[0] > 128 - 30).all()
    assert (observed[0] < 128 + 30).all()

    observed = aug.augment_images(images_list)
    assert (observed[0] > 128 - 30).all()
    assert (observed[0] < 128 + 30).all()

    observed = aug_det.augment_images(images_list)
    assert (observed[0] > 128 - 30).all()
    assert (observed[0] < 128 + 30).all()

    # random cvals
    aug = iaa.Affine(scale=1.0, translate_px=100, rotate=0, shear=0, cval=(0, 255))
    aug_det = aug.to_deterministic()
    last_aug = None
    last_aug_det = None
    nb_changed_aug = 0
    nb_changed_aug_det = 0
    nb_iterations = 1000
    averages = []
    for i in sm.xrange(nb_iterations):
        observed_aug = aug.augment_images(images)
        observed_aug_det = aug_det.augment_images(images)
        if i == 0:
            last_aug = observed_aug
            last_aug_det = observed_aug_det
        else:
            if not np.array_equal(observed_aug, last_aug):
                nb_changed_aug += 1
            if not np.array_equal(observed_aug_det, last_aug_det):
                nb_changed_aug_det += 1
            last_aug = observed_aug
            last_aug_det = observed_aug_det

        averages.append(int(np.average(observed_aug)))

    assert nb_changed_aug >= int(nb_iterations * 0.9)
    assert nb_changed_aug_det == 0
    # center pixel, should always be white when rotating line around center
    assert pixels_sums_aug[1, 1] > (nb_iterations * 0.98)
    assert pixels_sums_aug[1, 1] < (nb_iterations * 1.02)
    assert len(set(averages)) > 200

    # ---------------------
    # order
    # ---------------------
    # TODO
    print("[Note] There is currently no test for (interpolation) order in test_Affine().")


def test_ElasticTransformation():
    reseed()
    # TODO
    print("[Note] Elastic Transformations are currently not tested.")

def test_Sequential():
    reseed()

    image = np.array([[0, 1, 1],
                      [0, 0, 1],
                      [0, 0, 1]], dtype=np.uint8) * 255
    image = image[:, :, np.newaxis]
    images_list = [image]
    images = np.array([image])

    image_lr = np.array([[1, 1, 0],
                         [1, 0, 0],
                         [1, 0, 0]], dtype=np.uint8) * 255
    image_lr = image_lr[:, :, np.newaxis]
    images_lr = np.array([image_lr])

    image_ud = np.array([[0, 0, 1],
                         [0, 0, 1],
                         [0, 1, 1]], dtype=np.uint8) * 255
    image_ud = image_ud[:, :, np.newaxis]
    images_ud = np.array([image_ud])

    image_lr_ud = np.array([[1, 0, 0],
                            [1, 0, 0],
                            [1, 1, 0]], dtype=np.uint8) * 255
    image_lr_ud = image_lr_ud[:, :, np.newaxis]
    images_lr_ud_list = [image_lr_ud]
    images_lr_ud = np.array([image_lr_ud])

    keypoints = [ia.KeypointsOnImage([ia.Keypoint(x=1, y=0), ia.Keypoint(x=2, y=0), ia.Keypoint(x=2, y=1)], shape=image.shape)]
    keypoints_aug = [ia.KeypointsOnImage([ia.Keypoint(x=1, y=2), ia.Keypoint(x=0, y=2), ia.Keypoint(x=0, y=1)], shape=image.shape)]

    aug = iaa.Sequential([
        iaa.Fliplr(1.0),
        iaa.Flipud(1.0)
    ])
    aug_det = aug.to_deterministic()

    observed = aug.augment_images(images)
    assert np.array_equal(observed, images_lr_ud)

    observed = aug_det.augment_images(images)
    assert np.array_equal(observed, images_lr_ud)

    observed = aug.augment_images(images_list)
    assert array_equal_lists(observed, images_lr_ud_list)

    observed = aug_det.augment_images(images_list)
    assert array_equal_lists(observed, images_lr_ud_list)

    observed = aug.augment_keypoints(keypoints)
    assert keypoints_equal(observed, keypoints_aug)

    observed = aug_det.augment_keypoints(keypoints)
    assert keypoints_equal(observed, keypoints_aug)

    # 50% horizontal flip, 50% vertical flip
    aug = iaa.Sequential([
        iaa.Fliplr(0.5),
        iaa.Flipud(0.5)
    ])
    aug_det = aug.to_deterministic()
    last_aug = None
    last_aug_det = None
    nb_changed_aug = 0
    nb_changed_aug_det = 0
    nb_iterations = 1000
    for i in sm.xrange(nb_iterations):
        observed_aug = aug.augment_images(images)
        observed_aug_det = aug_det.augment_images(images)
        if i == 0:
            last_aug = observed_aug
            last_aug_det = observed_aug_det
        else:
            if not np.array_equal(observed_aug, last_aug):
                nb_changed_aug += 1
            if not np.array_equal(observed_aug_det, last_aug_det):
                nb_changed_aug_det += 1
            last_aug = observed_aug
            last_aug_det = observed_aug_det

        assert np.array_equal(observed_aug, images) or np.array_equal(observed_aug, images_lr) or np.array_equal(observed_aug, images_ud) or np.array_equal(observed_aug, images_lr_ud)
        assert np.array_equal(observed_aug_det, images) or np.array_equal(observed_aug_det, images_lr) or np.array_equal(observed_aug_det, images_ud) or np.array_equal(observed_aug_det, images_lr_ud)

    assert (0.25 - 0.10) <= (1 - (nb_changed_aug / nb_iterations)) <= (0.25 + 0.10) # should be the same in roughly 25% of all cases
    assert nb_changed_aug_det == 0

    # random order
    image = np.array([[0, 1, 1],
                      [0, 0, 1],
                      [0, 0, 1]], dtype=np.uint8)
    image = image[:, :, np.newaxis]
    images = np.array([image])

    images_first_second = (images + 10) * 10
    images_second_first = (images * 10) + 10

    keypoints = [ia.KeypointsOnImage([ia.Keypoint(x=0, y=0)], shape=image.shape)]
    keypoints_first_second = [ia.KeypointsOnImage([ia.Keypoint(x=1, y=1)], shape=image.shape)]
    keypoints_second_first = [ia.KeypointsOnImage([ia.Keypoint(x=1, y=0)], shape=image.shape)]

    def images_first(images, random_state, parents, hooks):
        return images + 10

    def images_second(images, random_state, parents, hooks):
        return images * 10

    def keypoints_first(keypoints_on_images, random_state, parents, hooks):
        for keypoints_on_image in keypoints_on_images:
            for keypoint in keypoints_on_image.keypoints:
                keypoint.x = keypoint.x + 1
        return keypoints_on_images

    def keypoints_second(keypoints_on_images, random_state, parents, hooks):
        for keypoints_on_image in keypoints_on_images:
            for keypoint in keypoints_on_image.keypoints:
                keypoint.y = keypoint.y + keypoint.x
        return keypoints_on_images

    aug_unrandom = iaa.Sequential([
        iaa.Lambda(images_first, keypoints_first),
        iaa.Lambda(images_second, keypoints_second)
    ], random_order=False)
    aug_unrandom_det = aug.to_deterministic()
    aug_random = iaa.Sequential([
        iaa.Lambda(images_first, keypoints_first),
        iaa.Lambda(images_second, keypoints_second)
    ], random_order=True)
    aug_random_det = aug.to_deterministic()

    last_aug = None
    last_aug_det = None
    nb_changed_aug = 0
    nb_changed_aug_det = 0
    nb_iterations = 1000

    nb_images_first_second_unrandom = 0
    nb_images_second_first_unrandom = 0
    nb_images_first_second_random = 0
    nb_images_second_first_random = 0
    nb_keypoints_first_second_unrandom = 0
    nb_keypoints_second_first_unrandom = 0
    nb_keypoints_first_second_random = 0
    nb_keypoints_second_first_random = 0

    for i in sm.xrange(nb_iterations):
        observed_aug_unrandom = aug_unrandom.augment_images(images)
        observed_aug_unrandom_det = aug_unrandom_det.augment_images(images)
        observed_aug_random = aug_random.augment_images(images)
        observed_aug_random_det = aug_random_det.augment_images(images)

        keypoints_aug_unrandom = aug_unrandom.augment_keypoints(keypoints)
        keypoints_aug_unrandom_det = aug_unrandom_det.augment_keypoints(keypoints)
        keypoints_aug_random = aug_random.augment_keypoints(keypoints)
        keypoints_aug_random_det = aug_random_det.augment_keypoints(keypoints)

        if i == 0:
            last_aug = observed_aug
            last_aug_det = observed_aug_det
        else:
            if not np.array_equal(observed_aug, last_aug):
                nb_changed_aug += 1
            if not np.array_equal(observed_aug_det, last_aug_det):
                nb_changed_aug_det += 1
            last_aug = observed_aug
            last_aug_det = observed_aug_det

        if np.array_equal(observed_aug_unrandom, images_first_second):
            nb_images_first_second_unrandom += 1
        elif np.array_equal(observed_aug_unrandom, images_second_first):
            nb_images_second_first_unrandom += 1
        else:
            raise Exception("Received output doesnt match any expected output.")

        if np.array_equal(observed_aug_random, images_first_second):
            nb_images_first_second_random += 1
        elif np.array_equal(observed_aug_random, images_second_first):
            nb_images_second_first_random += 1
        else:
            raise Exception("Received output doesnt match any expected output.")

        if keypoints_equal(keypoints_aug_unrandom, keypoints_first_second):
            nb_keypoints_first_second_unrandom += 1
        elif keypoints_equal(keypoints_aug_unrandom, keypoints_second_first):
            nb_keypoints_second_first_unrandom += 1
        else:
            raise Exception("Received output doesnt match any expected output.")

        if keypoints_equal(keypoints_aug_random, keypoints_first_second):
            nb_keypoints_first_second_random += 1
        elif keypoints_equal(keypoints_aug_random, keypoints_second_first):
            nb_keypoints_second_first_random += 1
        else:
            raise Exception("Received output doesnt match any expected output.")

    assert nb_changed_aug == 0
    assert nb_changed_aug_det == 0
    assert nb_images_first_second_unrandom == nb_iterations
    assert nb_images_second_first_unrandom == 0
    assert nb_keypoints_first_second_unrandom == nb_iterations
    assert nb_keypoints_second_first_unrandom == 0
    assert (0.50 - 0.1) <= nb_images_first_second_random / nb_iterations <= (0.50 + 0.1)
    assert (0.50 - 0.1) <= nb_images_second_first_random / nb_iterations <= (0.50 + 0.1)
    assert (0.50 - 0.1) <= nb_keypoints_first_second_random / nb_iterations <= (0.50 + 0.1)
    assert (0.50 - 0.1) <= nb_keypoints_second_first_random / nb_iterations <= (0.50 + 0.1)

def test_Sometimes():
    reseed()

    image = np.array([[0, 1, 1],
                      [0, 0, 1],
                      [0, 0, 1]], dtype=np.uint8) * 255
    image = image[:, :, np.newaxis]
    images_list = [image]
    images = np.array([image])

    image_lr = np.array([[1, 1, 0],
                         [1, 0, 0],
                         [1, 0, 0]], dtype=np.uint8) * 255
    image_lr = image_lr[:, :, np.newaxis]
    images_lr_list = [image_lr]
    images_lr = np.array([image_lr])

    image_ud = np.array([[0, 0, 1],
                         [0, 0, 1],
                         [0, 1, 1]], dtype=np.uint8) * 255
    image_ud = image_ud[:, :, np.newaxis]
    images_ud_list = [image_ud]
    images_ud = np.array([image_ud])

    keypoints = [ia.KeypointsOnImage([ia.Keypoint(x=1, y=0), ia.Keypoint(x=2, y=0), ia.Keypoint(x=2, y=1)], shape=image.shape)]
    keypoints_lr = [ia.KeypointsOnImage([ia.Keypoint(x=1, y=0), ia.Keypoint(x=0, y=0), ia.Keypoint(x=0, y=1)], shape=image.shape)]
    keypoints_ud = [ia.KeypointsOnImage([ia.Keypoint(x=1, y=2), ia.Keypoint(x=2, y=2), ia.Keypoint(x=2, y=1)], shape=image.shape)]

    # 100% chance of if-branch
    aug = iaa.Sometimes(1.0, [iaa.Fliplr(1.0)], [iaa.Flipud(1.0)])
    aug_det = aug.to_deterministic()

    observed = aug.augment_images(images)
    assert np.array_equal(observed, images_lr)

    observed = aug_det.augment_images(images)
    assert np.array_equal(observed, images_lr)

    observed = aug.augment_images(images_list)
    assert array_equal_lists(observed, images_lr_list)

    observed = aug_det.augment_images(images_list)
    assert array_equal_lists(observed, images_lr_list)

    observed = aug.augment_keypoints(keypoints)
    assert keypoints_equal(observed, keypoints_lr)

    observed = aug_det.augment_keypoints(keypoints)
    assert keypoints_equal(observed, keypoints_lr)

    # 100% chance of else-branch
    aug = iaa.Sometimes(0.0, [iaa.Fliplr(1.0)], [iaa.Flipud(1.0)])
    aug_det = aug.to_deterministic()

    observed = aug.augment_images(images)
    assert np.array_equal(observed, images_ud)

    observed = aug_det.augment_images(images)
    assert np.array_equal(observed, images_ud)

    observed = aug.augment_images(images_list)
    assert array_equal_lists(observed, images_ud_list)

    observed = aug_det.augment_images(images_list)
    assert array_equal_lists(observed, images_ud_list)

    observed = aug.augment_keypoints(keypoints)
    assert keypoints_equal(observed, keypoints_ud)

    observed = aug_det.augment_keypoints(keypoints)
    assert keypoints_equal(observed, keypoints_ud)

    # 50% if branch, 50% else branch
    aug = iaa.Sometimes(0.5, [iaa.Fliplr(1.0)], [iaa.Flipud(1.0)])
    aug_det = aug.to_deterministic()
    last_aug = None
    last_aug_det = None
    nb_changed_aug = 0
    nb_changed_aug_det = 0
    nb_iterations = 1000
    nb_images_if_branch = 0
    nb_images_else_branch = 0
    nb_keypoints_if_branch = 0
    nb_keypoints_else_branch = 0
    for i in sm.xrange(nb_iterations):
        observed_aug = aug.augment_images(images)
        observed_aug_det = aug_det.augment_images(images)
        keypoints_aug = aug.augment_keypoints(keypoints)
        keypoints_aug_det = aug.augment_keypoints(keypoints)
        if i == 0:
            last_aug = observed_aug
            last_aug_det = observed_aug_det
        else:
            if not np.array_equal(observed_aug, last_aug):
                nb_changed_aug += 1
            if not np.array_equal(observed_aug_det, last_aug_det):
                nb_changed_aug_det += 1
            last_aug = observed_aug
            last_aug_det = observed_aug_det

        if np.array_equal(observed_aug, images_lr):
            nb_images_if_branch += 1
        elif np.array_equal(observed_aug, images_ud):
            nb_images_else_branch += 1
        else:
            raise Exception("Received output doesnt match any expected output.")

        if keypoints_equal(keypoints_aug, keypoints_lr):
            nb_keypoints_if_branch += 1
        elif keypoints_equal(keypoints_aug, keypoints_ud):
            nb_keypoints_else_branch += 1
        else:
            raise Exception("Received output doesnt match any expected output.")

    assert (0.50 - 0.10) <= nb_images_if_branch / nb_iterations <= (0.50 + 0.10)
    assert (0.50 - 0.10) <= nb_images_else_branch / nb_iterations <= (0.50 + 0.10)
    assert (0.50 - 0.10) <= nb_keypoints_if_branch / nb_iterations <= (0.50 + 0.10)
    assert (0.50 - 0.10) <= nb_keypoints_else_branch / nb_iterations <= (0.50 + 0.10)
    assert (0.50 - 0.10) <= (1 - (nb_changed_aug / nb_iterations)) <= (0.50 + 0.10) # should be the same in roughly 50% of all cases
    assert nb_changed_aug_det == 0

    # 50% if branch, otherwise no change
    aug = iaa.Sometimes(0.5, iaa.Fliplr(1.0))
    aug_det = aug.to_deterministic()
    last_aug = None
    last_aug_det = None
    nb_changed_aug = 0
    nb_changed_aug_det = 0
    nb_iterations = 1000
    nb_images_if_branch = 0
    nb_images_else_branch = 0
    nb_keypoints_if_branch = 0
    nb_keypoints_else_branch = 0
    for i in sm.xrange(nb_iterations):
        observed_aug = aug.augment_images(images)
        observed_aug_det = aug_det.augment_images(images)
        keypoints_aug = aug.augment_keypoints(keypoints)
        keypoints_aug_det = aug.augment_keypoints(keypoints)
        if i == 0:
            last_aug = observed_aug
            last_aug_det = observed_aug_det
        else:
            if not np.array_equal(observed_aug, last_aug):
                nb_changed_aug += 1
            if not np.array_equal(observed_aug_det, last_aug_det):
                nb_changed_aug_det += 1
            last_aug = observed_aug
            last_aug_det = observed_aug_det

        if np.array_equal(observed_aug, images_lr):
            nb_images_if_branch += 1
        elif np.array_equal(observed_aug, images):
            nb_images_else_branch += 1
        else:
            raise Exception("Received output doesnt match any expected output.")

        if keypoints_equal(keypoints_aug, keypoints_lr):
            nb_keypoints_if_branch += 1
        elif keypoints_equal(keypoints_aug, keypoints):
            nb_keypoints_else_branch += 1
        else:
            raise Exception("Received output doesnt match any expected output.")

    assert (0.50 - 0.10) <= nb_images_if_branch / nb_iterations <= (0.50 + 0.10)
    assert (0.50 - 0.10) <= nb_images_else_branch / nb_iterations <= (0.50 + 0.10)
    assert (0.50 - 0.10) <= nb_keypoints_if_branch / nb_iterations <= (0.50 + 0.10)
    assert (0.50 - 0.10) <= nb_keypoints_else_branch / nb_iterations <= (0.50 + 0.10)
    assert (0.50 - 0.10) <= (1 - (nb_changed_aug / nb_iterations)) <= (0.50 + 0.10) # should be the same in roughly 50% of all cases
    assert nb_changed_aug_det == 0

def test_2d_inputs():
    """Test whether inputs of 2D-images (i.e. (H, W) instead of (H, W, C)) work.
    """
    reseed()

    base_img1 = np.array([[0, 0, 1, 1],
                          [0, 0, 1, 1],
                          [0, 1, 1, 1]], dtype=np.uint8)
    base_img2 = np.array([[0, 0, 1, 1],
                          [0, 1, 1, 1],
                          [0, 1, 0, 0]], dtype=np.uint8)

    base_img1_flipped = np.array([[1, 1, 0, 0],
                                  [1, 1, 0, 0],
                                  [1, 1, 1, 0]], dtype=np.uint8)
    base_img2_flipped = np.array([[1, 1, 0, 0],
                                  [1, 1, 1, 0],
                                  [0, 0, 1, 0]], dtype=np.uint8)

    images = np.array([base_img1, base_img2])
    images_flipped = np.array([base_img1_flipped, base_img2_flipped])
    images_list = [base_img1, base_img2]
    images_flipped_list = [base_img1_flipped, base_img2_flipped]
    images_list2d3d = [base_img1, base_img2[:, :, np.newaxis]]
    images_flipped_list2d3d = [base_img1_flipped, base_img2_flipped[:, :, np.newaxis]]

    aug = iaa.Fliplr(1.0)
    noaug = iaa.Fliplr(0.0)

    # one numpy array as input
    observed = aug.augment_images(images)
    assert np.array_equal(observed, images_flipped)

    observed = noaug.augment_images(images)
    assert np.array_equal(observed, images)

    # list of 2d images
    observed = aug.augment_images(images_list)
    assert array_equal_lists(observed, images_flipped_list)

    observed = noaug.augment_images(images_list)
    assert array_equal_lists(observed, images_list)

    # list of images, one 2d and one 3d
    observed = aug.augment_images(images_list2d3d)
    assert array_equal_lists(observed, images_flipped_list2d3d)

    observed = noaug.augment_images(images_list2d3d)
    assert array_equal_lists(observed, images_list2d3d)

def test_background_augmentation():
    reseed()

    image = np.array([[0, 0, 1, 1],
                      [0, 0, 1, 1],
                      [0, 1, 1, 1]], dtype=np.uint8)
    image_flipped = np.fliplr(image)
    keypoint = ia.Keypoint(x=2, y=1)
    keypoints = [ia.KeypointsOnImage([keypoint], shape=image.shape)]
    kp_flipped = ia.Keypoint(
        x=image.shape[1]-1-keypoint.x,
        y=keypoint.y
    )

    seq = iaa.Fliplr(0.5)

    # with images as list, background=False
    nb_flipped_images = 0
    nb_flipped_keypoints = 0
    nb_iterations = 1000
    batches = [ia.Batch(images=[np.copy(image)], keypoints=[keypoints[0].deepcopy()]) for _ in sm.xrange(nb_iterations)]
    batches_aug = list(seq.augment_batches(batches, background=False))
    for batch_aug in batches_aug:
        image_aug = batch_aug.images_aug[0]
        keypoint_aug = batch_aug.keypoints_aug[0].keypoints[0]
        assert np.array_equal(image_aug, image) or np.array_equal(image_aug, image_flipped)
        if np.array_equal(image_aug, image_flipped):
            nb_flipped_images += 1

        assert (keypoint_aug.x == keypoint.x and keypoint_aug.y == keypoint.y) \
               or (keypoint_aug.x == kp_flipped.x and keypoint_aug.y == kp_flipped.y)
        if keypoint_aug.x == kp_flipped.x and keypoint_aug.y == kp_flipped.y:
            nb_flipped_keypoints += 1
    assert 0.4*nb_iterations <= nb_flipped_images <= 0.6*nb_iterations
    assert nb_flipped_images == nb_flipped_keypoints

    # with images as list
    nb_flipped_images = 0
    nb_flipped_keypoints = 0
    nb_iterations = 1000
    batches = [ia.Batch(images=[np.copy(image)], keypoints=[keypoints[0].deepcopy()]) for _ in sm.xrange(nb_iterations)]
    batches_aug = list(seq.augment_batches(batches, background=True))
    for batch_aug in batches_aug:
        image_aug = batch_aug.images_aug[0]
        keypoint_aug = batch_aug.keypoints_aug[0].keypoints[0]
        assert np.array_equal(image_aug, image) or np.array_equal(image_aug, image_flipped)
        if np.array_equal(image_aug, image_flipped):
            nb_flipped_images += 1

        assert (keypoint_aug.x == keypoint.x and keypoint_aug.y == keypoint.y) \
               or (keypoint_aug.x == kp_flipped.x and keypoint_aug.y == kp_flipped.y)
        if keypoint_aug.x == kp_flipped.x and keypoint_aug.y == kp_flipped.y:
            nb_flipped_keypoints += 1
    assert 0.4*nb_iterations <= nb_flipped_images <= 0.6*nb_iterations
    assert nb_flipped_images == nb_flipped_keypoints

    # with images as array
    nb_flipped_images = 0
    nb_flipped_keypoints = 0
    nb_iterations = 1000
    batches = [ia.Batch(images=np.array([np.copy(image)], dtype=np.uint8), keypoints=None) for _ in sm.xrange(nb_iterations)]
    batches_aug = list(seq.augment_batches(batches, background=True))
    for batch_aug in batches_aug:
        #batch = ia.Batch(images=np.array([image], dtype=np.uint8), keypoints=keypoints)
        #batches_aug = list(seq.augment_batches([batch], background=True))
        #batch_aug = batches_aug[0]
        image_aug = batch_aug.images_aug[0]
        assert np.array_equal(image_aug, image) or np.array_equal(image_aug, image_flipped)
        if np.array_equal(image_aug, image_flipped):
            nb_flipped_images += 1
    assert 0.4*nb_iterations <= nb_flipped_images <= 0.6*nb_iterations

    # array (N, H, W) as input
    nb_flipped_images = 0
    nb_iterations = 1000
    batches = [np.array([np.copy(image)], dtype=np.uint8) for _ in sm.xrange(nb_iterations)]
    batches_aug = list(seq.augment_batches(batches, background=True))
    for batch_aug in batches_aug:
        #batch = np.array([image], dtype=np.uint8)
        #batches_aug = list(seq.augment_batches([batch], background=True))
        #image_aug = batches_aug[0][0]
        image_aug = batch_aug[0]
        assert np.array_equal(image_aug, image) or np.array_equal(image_aug, image_flipped)
        if np.array_equal(image_aug, image_flipped):
            nb_flipped_images += 1
    assert 0.4*nb_iterations <= nb_flipped_images <= 0.6*nb_iterations

    # list of list of KeypointsOnImage as input
    nb_flipped_keypoints = 0
    nb_iterations = 1000
    #batches = [ia.Batch(images=[np.copy(image)], keypoints=None) for _ in sm.xrange(nb_iterations)]
    batches = [[keypoints[0].deepcopy()] for _ in sm.xrange(nb_iterations)]
    batches_aug = list(seq.augment_batches(batches, background=True))
    for batch_aug in batches_aug:
        #batch = [keypoints]
        #batches_aug = list(seq.augment_batches([batch], background=True))
        #batch_aug = batches_aug[0]
        #keypoint_aug = batches_aug[0].keypoints[0].keypoints[0]
        keypoint_aug = batch_aug[0].keypoints[0]

        assert (keypoint_aug.x == keypoint.x and keypoint_aug.y == keypoint.y) \
               or (keypoint_aug.x == kp_flipped.x and keypoint_aug.y == kp_flipped.y)
        if keypoint_aug.x == kp_flipped.x and keypoint_aug.y == kp_flipped.y:
            nb_flipped_keypoints += 1
    assert 0.4*nb_iterations <= nb_flipped_keypoints <= 0.6*nb_iterations

    # test all augmenters
    augs = [
        iaa.Sequential([iaa.Fliplr(1.0), iaa.Flipud(1.0)]),
        iaa.SomeOf(1, [iaa.Fliplr(1.0), iaa.Flipud(1.0)]),
        iaa.OneOf([iaa.Fliplr(1.0), iaa.Flipud(1.0)]),
        iaa.Sometimes(1.0, iaa.Fliplr(1)),
        iaa.WithColorspace("HSV", children=iaa.Add((-50, 50))),
        iaa.WithChannels([0], iaa.Add((-50, 50))),
        iaa.Noop(name="Noop-nochange"),
        iaa.Lambda(
            func_images=lambda images, random_state, parents, hooks: images,
            func_keypoints=lambda keypoints_on_images, random_state, parents, hooks: keypoints_on_images,
            name="Lambda-nochange"
        ),
        iaa.AssertLambda(
            func_images=lambda images, random_state, parents, hooks: True,
            func_keypoints=lambda keypoints_on_images, random_state, parents, hooks: True,
            name="AssertLambda-nochange"
        ),
        iaa.AssertShape(
            (None, 64, 64, 3),
            check_keypoints=False,
            name="AssertShape-nochange"
        ),
        iaa.Scale((0.5, 0.9)),
        iaa.CropAndPad(px=(-50, 50)),
        iaa.Pad(px=(1, 50)),
        iaa.Crop(px=(1, 50)),
        iaa.Fliplr(1.0),
        iaa.Flipud(1.0),
        iaa.Superpixels(p_replace=(0.25, 1.0), n_segments=(16, 128)),
        iaa.ChangeColorspace(to_colorspace="GRAY"),
        iaa.Grayscale(alpha=(0.1, 1.0)),
        iaa.GaussianBlur(1.0),
        iaa.AverageBlur(5),
        iaa.MedianBlur(5),
        iaa.Convolve(np.array([[0, 1, 0],
                               [1, -4, 1],
                               [0, 1, 0]])),
        iaa.Sharpen(alpha=(0.1, 1.0), lightness=(0.8, 1.2)),
        iaa.Emboss(alpha=(0.1, 1.0), strength=(0.8, 1.2)),
        iaa.EdgeDetect(alpha=(0.1, 1.0)),
        iaa.DirectedEdgeDetect(alpha=(0.1, 1.0), direction=(0.0, 1.0)),
        iaa.Add((-50, 50)),
        iaa.AddElementwise((-50, 50)),
        iaa.AdditiveGaussianNoise(scale=(0.1, 1.0)),
        iaa.Multiply((0.6, 1.4)),
        iaa.MultiplyElementwise((0.6, 1.4)),
        iaa.Dropout((0.3, 0.5)),
        iaa.CoarseDropout((0.3, 0.5), size_percent=(0.05, 0.2)),
        iaa.Invert(0.5),
        iaa.ContrastNormalization((0.6, 1.4)),
        iaa.Affine(scale=(0.7, 1.3), translate_percent=(-0.1, 0.1), rotate=(-20, 20), shear=(-20, 20), order=ia.ALL, mode=ia.ALL, cval=(0, 255)),
        iaa.PiecewiseAffine(scale=(0.1, 0.3)),
        iaa.ElasticTransformation(alpha=0.5)
    ]

    nb_iterations = 100
    image = ia.quokka(size=(64, 64))
    batch = ia.Batch(images=np.array([image]), keypoints=keypoints)
    batches = [ia.Batch(images=[np.copy(image)], keypoints=[keypoints[0].deepcopy()]) for _ in sm.xrange(nb_iterations)]
    for aug in augs:
        nb_changed = 0
        batches_aug = list(aug.augment_batches(batches, background=True))
        for batch_aug in batches_aug:
            image_aug = batch_aug.images_aug[0]
            if image.shape != image_aug.shape or not np.array_equal(image, image_aug):
                nb_changed += 1
                if nb_changed > 10:
                    break
        if "-nochange" not in aug.name:
            assert nb_changed > 0
        else:
            assert nb_changed == 0

def test_determinism():
    reseed()

    images = [
        ia.quokka(size=(128, 128)),
        ia.quokka(size=(64, 64)),
        misc.imresize(data.astronaut(), (128, 256))
    ]
    keypoints = [
        ia.KeypointsOnImage([
            ia.Keypoint(x=20, y=10), ia.Keypoint(x=5, y=5), ia.Keypoint(x=10, y=43)
            ], shape=(50, 60, 3))
    ]

    augs = [
        iaa.Sequential([iaa.Fliplr(1.0), iaa.Flipud(1.0)]),
        iaa.SomeOf(1, [iaa.Fliplr(1.0), iaa.Flipud(1.0)]),
        iaa.OneOf([iaa.Fliplr(1.0), iaa.Flipud(1.0)]),
        iaa.Sometimes(1.0, iaa.Fliplr(1)),
        iaa.WithColorspace("HSV", children=iaa.Add((-50, 50))),
        iaa.WithChannels([0], iaa.Add((-50, 50))),
        iaa.Noop(name="Noop-nochange"),
        iaa.Lambda(
            func_images=lambda images, random_state, parents, hooks: images,
            func_keypoints=lambda keypoints_on_images, random_state, parents, hooks: keypoints_on_images,
            name="Lambda-nochange"
        ),
        iaa.AssertLambda(
            func_images=lambda images, random_state, parents, hooks: True,
            func_keypoints=lambda keypoints_on_images, random_state, parents, hooks: True,
            name="AssertLambda-nochange"
        ),
        iaa.AssertShape(
            (None, None, None, 3),
            check_keypoints=False,
            name="AssertShape-nochange"
        ),
        iaa.Scale((0.5, 0.9)),
        iaa.CropAndPad(px=(-50, 50)),
        iaa.Pad(px=(1, 50)),
        iaa.Crop(px=(1, 50)),
        iaa.Fliplr(1.0),
        iaa.Flipud(1.0),
        iaa.Superpixels(p_replace=(0.25, 1.0), n_segments=(16, 128)),
        iaa.ChangeColorspace(to_colorspace="GRAY"),
        iaa.Grayscale(alpha=(0.1, 1.0)),
        iaa.GaussianBlur(1.0),
        iaa.AverageBlur(5),
        iaa.MedianBlur(5),
        iaa.Convolve(np.array([[0, 1, 0],
                               [1, -4, 1],
                               [0, 1, 0]])),
        iaa.Sharpen(alpha=(0.1, 1.0), lightness=(0.8, 1.2)),
        iaa.Emboss(alpha=(0.1, 1.0), strength=(0.8, 1.2)),
        iaa.EdgeDetect(alpha=(0.1, 1.0)),
        iaa.DirectedEdgeDetect(alpha=(0.1, 1.0), direction=(0.0, 1.0)),
        iaa.Add((-50, 50)),
        iaa.AddElementwise((-50, 50)),
        iaa.AdditiveGaussianNoise(scale=(0.1, 1.0)),
        iaa.Multiply((0.6, 1.4)),
        iaa.MultiplyElementwise((0.6, 1.4)),
        iaa.Dropout((0.3, 0.5)),
        iaa.CoarseDropout((0.3, 0.5), size_percent=(0.05, 0.2)),
        iaa.Invert(0.5),
        iaa.ContrastNormalization((0.6, 1.4)),
        iaa.Affine(scale=(0.7, 1.3), translate_percent=(-0.1, 0.1), rotate=(-20, 20), shear=(-20, 20), order=ia.ALL, mode=ia.ALL, cval=(0, 255)),
        iaa.PiecewiseAffine(scale=(0.1, 0.3)),
        iaa.ElasticTransformation(alpha=0.5)
    ]

    for aug in augs:
        aug_det = aug.to_deterministic()
        images_aug1 = aug_det.augment_images(images)
        images_aug2 = aug_det.augment_images(images)
        kps_aug1 = aug_det.augment_keypoints(keypoints)
        kps_aug2 = aug_det.augment_keypoints(keypoints)
        assert array_equal_lists(images_aug1, images_aug2), "Images not identical for %s" % (aug.name,)
        assert keypoints_equal(kps_aug1, kps_aug2), "Keypoints not identical for %s" % (aug.name,)

def test_keypoint_augmentation():
    ia.seed(1)

    keypoints = []
    for y in range(40//5):
        for x in range(60//5):
            keypoints.append(ia.Keypoint(y=y*5, x=x*5))

    keypoints_oi = ia.KeypointsOnImage(keypoints, shape=(40, 60, 3))
    augs = [
        iaa.Add((-5, 5), name="Add"),
        iaa.AddElementwise((-5, 5), name="AddElementwise"),
        iaa.AdditiveGaussianNoise(0.01*255, name="AdditiveGaussianNoise"),
        iaa.Multiply((0.95, 1.05), name="Multiply"),
        iaa.Dropout(0.01, name="Dropout"),
        iaa.CoarseDropout(0.01, size_px=6, name="CoarseDropout"),
        iaa.Invert(0.01, per_channel=True, name="Invert"),
        iaa.ContrastNormalization((0.95, 1.05), name="ContrastNormalization"),
        iaa.GaussianBlur(sigma=(0.95, 1.05), name="GaussianBlur"),
        iaa.AverageBlur((3, 5), name="AverageBlur"),
        iaa.MedianBlur((3, 5), name="MedianBlur"),
        #iaa.BilateralBlur((3, 5), name="BilateralBlur"),
        # WithColorspace ?
        #iaa.AddToHueAndSaturation((-5, 5), name="AddToHueAndSaturation"),
        # ChangeColorspace ?
        # Grayscale cannot be tested, input not RGB
        # Convolve ?
        iaa.Sharpen((0.0, 0.1), lightness=(1.0, 1.2), name="Sharpen"),
        iaa.Emboss(alpha=(0.0, 0.1), strength=(0.5, 1.5), name="Emboss"),
        iaa.EdgeDetect(alpha=(0.0, 0.1), name="EdgeDetect"),
        iaa.DirectedEdgeDetect(alpha=(0.0, 0.1), direction=0, name="DirectedEdgeDetect"),
        iaa.Fliplr(0.5, name="Fliplr"),
        iaa.Flipud(0.5, name="Flipud"),
        iaa.Affine(translate_px=(-5, 5), name="Affine-translate-px"),
        iaa.Affine(translate_percent=(-0.05, 0.05), name="Affine-translate-percent"),
        iaa.Affine(rotate=(-20, 20), name="Affine-rotate"),
        iaa.Affine(shear=(-20, 20), name="Affine-shear"),
        iaa.Affine(scale=(0.9, 1.1), name="Affine-scale"),
        iaa.PiecewiseAffine(scale=(0.001, 0.005), name="PiecewiseAffine"),
        #iaa.PerspectiveTransform(scale=(0.01, 0.10), name="PerspectiveTransform"),
        iaa.ElasticTransformation(alpha=(0.1, 0.2), sigma=(0.1, 0.2), name="ElasticTransformation"),
        # Sequential
        # SomeOf
        # OneOf
        # Sometimes
        # WithChannels
        # Noop
        # Lambda
        # AssertLambda
        # AssertShape
        iaa.Alpha((0.0, 0.1), iaa.Add(10), name="Alpha"),
        iaa.AlphaElementwise((0.0, 0.1), iaa.Add(10), name="AlphaElementwise"),
        iaa.SimplexNoiseAlpha(iaa.Add(10), name="SimplexNoiseAlpha"),
        iaa.FrequencyNoiseAlpha(exponent=(-2, 2), first=iaa.Add(10), name="SimplexNoiseAlpha"),
        iaa.Superpixels(p_replace=0.01, n_segments=64),
        iaa.Scale(0.5, name="Scale"),
        iaa.CropAndPad(px=(-10, 10), name="CropAndPad"),
        iaa.Pad(px=(0, 10), name="Pad"),
        iaa.Crop(px=(0, 10), name="Crop")
    ]

    for aug in augs:
        #if aug.name != "PiecewiseAffine":
        #    continue
        dss = []
        for i in range(10):
            aug_det = aug.to_deterministic()
            kp_image = keypoints_oi.to_keypoint_image(size=5)
            kp_image_aug = aug_det.augment_image(kp_image)
            kp_image_aug_rev = ia.KeypointsOnImage.from_keypoint_image(
                kp_image_aug,
                if_not_found_coords={"x": -9999, "y": -9999}
            )
            kp_aug = aug_det.augment_keypoints([keypoints_oi])[0]
            ds = []
            assert len(kp_image_aug_rev.keypoints) == len(kp_aug.keypoints), "Lost keypoints for '%s' (%d vs expected %d)" % (aug.name, len(kp_aug.keypoints), len(kp_image_aug_rev.keypoints))
            for kp_pred, kp_pred_img in zip(kp_aug.keypoints, kp_image_aug_rev.keypoints):
                kp_pred_lost = (kp_pred.x == -9999 and kp_pred.y == -9999)
                kp_pred_img_lost = (kp_pred_img.x == -9999 and kp_pred_img.y == -9999)
                #if kp_pred_lost and not kp_pred_img_lost:
                #    print("lost kp_pred", kp_pred_img)
                #elif not kp_pred_lost and kp_pred_img_lost:
                #    print("lost kp_pred_img", kp_pred)
                #elif kp_pred_lost and kp_pred_img_lost:
                #    print("lost both keypoints")

                if not kp_pred_lost and not kp_pred_img_lost:
                    d = np.sqrt((kp_pred.x - kp_pred_img.x) ** 2 + (kp_pred.y - kp_pred_img.y) ** 2)
                    ds.append(d)
            #print(aug.name, np.average(ds), ds)
            dss.extend(ds)
            if len(ds) == 0:
                print("[INFO] No valid keypoints found for '%s' in test_keypoint_augmentation()" % (str(aug),))
        assert np.average(dss) < 5.0, "Average distance too high (%.2f, with ds: %s)" % (np.average(dss), str(dss))

def test_copy_random_state():
    image = ia.quokka_square(size=(128, 128))
    images = np.array([image] * 64, dtype=np.uint8)

    source = iaa.Sequential([
        iaa.Fliplr(0.5, name="hflip"),
        iaa.Dropout(0.05, name="dropout"),
        iaa.Affine(translate_px=(-10, 10), name="translate", random_state=3),
        iaa.GaussianBlur(1.0, name="blur", random_state=4)
    ], random_state=5)
    target = iaa.Sequential([
        iaa.Fliplr(0.5, name="hflip"),
        iaa.Dropout(0.05, name="dropout"),
        iaa.Affine(translate_px=(-10, 10), name="translate")
    ])

    source.localize_random_state_()

    target_cprs = target.copy_random_state(source, matching="position")
    source_alt = source.remove_augmenters(lambda aug, parents: aug.name == "blur")
    images_aug_source = source_alt.augment_images(images)
    images_aug_target = target_cprs.augment_images(images)
    #misc.imshow(np.hstack([images_aug_source[0], images_aug_source[1], images_aug_target[0], images_aug_target[1]]))
    assert np.array_equal(images_aug_source, images_aug_target)

    target_cprs = target.copy_random_state(source, matching="name")
    source_alt = source.remove_augmenters(lambda aug, parents: aug.name == "blur")
    images_aug_source = source_alt.augment_images(images)
    images_aug_target = target_cprs.augment_images(images)
    assert np.array_equal(images_aug_source, images_aug_target)

    source_alt = source.remove_augmenters(lambda aug, parents: aug.name == "blur")
    source_det = source_alt.to_deterministic()
    target_cprs_det = target.copy_random_state(source_det, matching="name", copy_determinism=True)
    images_aug_source1 = source_det.augment_images(images)
    images_aug_target1 = target_cprs_det.augment_images(images)
    images_aug_source2 = source_det.augment_images(images)
    images_aug_target2 = target_cprs_det.augment_images(images)
    assert np.array_equal(images_aug_source1, images_aug_source2)
    assert np.array_equal(images_aug_target1, images_aug_target2)
    assert np.array_equal(images_aug_source1, images_aug_target1)
    assert np.array_equal(images_aug_source2, images_aug_target2)

def create_random_images(size):
    return np.random.uniform(0, 255, size).astype(np.uint8)

def create_random_keypoints(size_images, nb_keypoints_per_img):
    result = []
    for i in sm.xrange(size_images[0]):
        kps = []
        height, width = size_images[1], size_images[2]
        for i in sm.xrange(nb_keypoints_per_img):
            x = np.random.randint(0, width-1)
            y = np.random.randint(0, height-1)
            kps.append(ia.Keypoint(x=x, y=y))
        result.append(ia.KeypointsOnImage(kps, shape=size_images[1:]))
    return result

def array_equal_lists(list1, list2):
    assert isinstance(list1, list)
    assert isinstance(list2, list)

    if len(list1) != len(list2):
        return False

    for a, b in zip(list1, list2):
        if not np.array_equal(a, b):
            return False

    return True

def keypoints_equal(kps1, kps2):
    if len(kps1) != len(kps2):
        return False

    for i in sm.xrange(len(kps1)):
        a = kps1[i].keypoints
        b = kps2[i].keypoints
        if len(a) != len(b):
            return False

        for j in sm.xrange(len(a)):
            if a[j].x != b[j].x or a[j].y != b[j].y:
                return False

    return True

def reseed(seed=0):
    ia.seed(seed)
    np.random.seed(seed)
    random.seed(seed)

if __name__ == "__main__":
    main()
