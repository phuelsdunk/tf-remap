import operator
import functools
import itertools

import tensorflow as tf


@tf.function
def remap(grid, points,
          batch_dims=1, border_mode='constant', border_value=0,
          indexing='ij',
          name=None):
    '''
    Returns an tensor that contains the values from grid sampled at the given
    points. Specifically, return an output `remapped` that contains
    at the batch index `i` and location `x = (x_1, ... x_n)`, the values

    ~~~
    remapped[i, x_1, ..., x_n] = grid[i, p(i, x)_1, ..., p(i, x)_n]
    ~~~

    from `grid` sampled at locations `p(i, x) = (p(i, x)_1, ..., p(i, x)_n)`
    where

    ~~~
    p(i, x) = points[i, x_1, ..., x_n]
    ~~~

    Parameters:
    -----------
    grid: shape `batch_shape + space_shape + channels_shape`
    points: shape `batch_shape + sample_shape + [space_dims]

    Returns:
    --------
    shape `batch_shape + sample_shape + channels_shape`
    '''
    with tf.name_scope(name or 'remap'):
        space_dims = points.shape[-1]
        sample_dims = len(points.shape) - batch_dims - 1
        channel_dims = len(grid.shape) - batch_dims - space_dims

        grid_shape = tf.shape(grid)
        points_shape = tf.shape(points)

        batch_shape = [
            grid_shape[axis]
            for axis
            in range(batch_dims)
        ]
        space_shape = [
            grid_shape[axis]
            for axis
            in range(batch_dims, batch_dims + space_dims)
        ]
        sample_shape = [
            points_shape[axis]
            for axis
            in range(batch_dims, batch_dims + sample_dims)
        ]

        if border_mode.upper() == 'CONSTANT':
            border_value = tf.cast(border_value, grid.dtype)

        unstacked_points = tf.unstack(points, axis=-1)
        if indexing == 'xy':
            unstacked_points = unstacked_points[::-1]

        # For each dimension two grid indexes with weights are gathered.
        # These pairs will be stored in weights and indexes respectively.
        weights = []
        indexes = []
        masks = []
        for space_axis in range(space_dims):
            with tf.name_scope('dim-%s' % space_axis):
                loc = unstacked_points[space_axis]

                # Get the nearest neighbour grid indexes
                floor_index = tf.cast(tf.floor(loc), tf.int32)
                ceil_index = floor_index + 1

                # The weights are at opposite sides:
                # If loc is near to floor:
                #     low_weight should increase, high weight should decrease
                # if loc is near to ceil:
                #     hight_weight should increase, low wieght should decrease
                #
                #   |-- high_weight ---|--- low_weight --|
                # floor ------------- loc --------------- ceil
                floor_weight = tf.cast(ceil_index, tf.float32) - loc
                ceil_weight = loc - tf.cast(floor_index, tf.float32)
                weights.append((floor_weight, ceil_weight))

                # Possible range of indexes, used below to handle border
                # effects
                grid_size = space_shape[space_axis]
                min_index = tf.constant(0, tf.int32)
                max_index = tf.cast(grid_size - 1, tf.int32)

                # Mark index for border effets
                floor_mask = tf.logical_and(
                    tf.greater_equal(floor_index, min_index),
                    tf.greater_equal(max_index, floor_index))
                ceil_mask = tf.logical_and(
                    tf.greater_equal(ceil_index, min_index),
                    tf.greater_equal(max_index, ceil_index))
                masks.append((floor_mask, ceil_mask))

                # Remap indices to valid ones, respecting the border_mode
                if border_mode.upper() == 'CONSTANT':
                    floor_index = tf.clip_by_value(floor_index,
                                                   min_index, max_index)
                    ceil_index = tf.clip_by_value(ceil_index,
                                                  min_index, max_index)
                elif border_mode.upper() == 'REPLICATE':
                    floor_index = tf.clip_by_value(floor_index,
                                                   min_index, max_index)
                    ceil_index = tf.clip_by_value(ceil_index,
                                                  min_index, max_index)
                elif border_mode.upper() == 'WRAP':
                    floor_index = tf.math.mod(floor_index, grid_size)
                    ceil_index = tf.math.mod(ceil_index, grid_size)
                elif border_mode.upper() == 'REFLECT':
                    floor_index = (tf.minimum(floor_index, max_index)
                                   - tf.maximum(floor_index - max_index, 0))
                    ceil_index = (tf.minimum(ceil_index, max_index)
                                  - tf.maximum(ceil_index - max_index, 0))
                else:
                    raise ValueError('Unknown border_mode \'%s\''
                                     % border_mode)
                indexes.append((floor_index, ceil_index))

        # The locations are inside of a hyper-cube for which all corners
        # need to be iterated over. That will be the product over floor and
        # ceil indexes per dimension:
        weighted_values = []
        iter_weights = itertools.product(*weights)
        iter_indexes = itertools.product(*indexes)
        iter_masks = itertools.product(*masks)
        iter_weights_indexes = zip(iter_weights, iter_indexes, iter_masks)
        for index, item in enumerate(iter_weights_indexes):
            weight_tuple, index_tuple, mask_tuple = item
            with tf.name_scope('corner-%d' % index):
                value_broadcast_shape = (
                    batch_shape + sample_shape + channel_dims * [1]
                )

                mask = functools.reduce(operator.and_, mask_tuple)
                mask = tf.reshape(mask, value_broadcast_shape)

                weight = functools.reduce(operator.mul, weight_tuple)
                weight = tf.reshape(weight, value_broadcast_shape)

                pos = tf.stack(index_tuple, axis=-1)
                value = tf.gather_nd(grid, pos, batch_dims=batch_dims)

                if border_mode.upper() == 'CONSTANT':
                    value = tf.where(mask, value, border_value)

                value = tf.cast(value, weight.dtype)
                weighted_values.append(weight * value)

        mapped_grid = tf.add_n(weighted_values, name='interpolate')
        if grid.dtype.is_integer:
            mapped_grid = tf.clip_by_value(mapped_grid,
                                           grid.dtype.min, grid.dtype.max)
        mapped_grid = tf.cast(mapped_grid, grid.dtype)
        return mapped_grid


@tf.function
def remap_affine(grid, affine, sample_shape,
                 batch_dims=1, border_mode='constant', border_value=0,
                 indexing='ij',
                 name=None):
    '''
    Remap values of grid along an affine transform. Specifically, return an
    output `remapped` that contains at the batch index `i` and location
    `x`, the values

    ~~~
    remapped[i, x] = grid[i, Ax + b]
    ~~~

    where `affine` is composed by `(A|b)`.


    Parameters:
    -----------
    grid: shape `batch_shape + space_shape + channels_shape`
    affine: shape `batch_shape + [space_dims, space_dims + 1]`

    Returns:
    --------
    shape `batch_shape + sample_shape + channels_shape`
    '''
    with tf.name_scope(name or 'remap_affine'):
        space_dims = affine.shape[-2]
        if affine.shape[-1] != space_dims + 1:
            raise ValueError('affine.shape[-2:] has to be of form '
                             '[space_dims, space_dims + 1]')

        affine_shape = tf.shape(affine)

        batch_shape = [
            affine_shape[axis]
            for axis
            in range(batch_dims)
        ]

        ranges = [tf.range(sample_shape[axis], dtype=affine.dtype)
                  for axis in range(space_dims)]
        points = tf.meshgrid(*ranges, indexing='ij')
        if indexing == 'xy':
            points = points[::-1]
        points = tf.stack(points, axis=-1)

        # [height, width, 3]
        ones = tf.ones(sample_shape + [1], points.dtype)
        points_h = tf.concat([points, ones], axis=-1)

        affine = tf.reshape(affine,
                            batch_shape + space_dims * [1]
                            + [space_dims, space_dims + 1])
        points = tf.linalg.matvec(affine, points_h)

        return remap(grid, points, batch_dims=batch_dims,
                     border_mode=border_mode, border_value=border_value,
                     indexing=indexing)


@tf.function
def remap_flow(grid, flow,
               batch_dims=1, border_mode='replicate', border_value=None,
               indexing='ij',
               name=None):
    '''
    Remap `grid` along relative positions given by flow. Specifically, return
    an output `warped` that contains at batch index `i` and location
    `x = (x_1, ..., x_n)

    ~~~
    remapped[i, x_1, ..., x_n]
        = grid[i, x_1 + f(i, x)_1, ..., x_n + f(i, x)_n]
    ~~~

    where

    ~~~
    f(i, x)_i = flow[i, x_1, ..., x_n]
    ~~~

    Parameters:
    -----------
    grid: shape `batch_shape + space_shape + channels_shape`
    flow: shape `batch_shape + space_shape + [space_dims]`

    Returns:
    --------
    shape `batch_shape + space_shape + channels_shape`
    '''
    with tf.name_scope(name or 'remap_flow'):
        space_dims = flow.shape[-1]

        grid_shape = tf.shape(grid)

        space_shape = [
            grid_shape[axis]
            for axis
            in range(batch_dims, batch_dims + space_dims)
        ]

        ranges = [tf.range(size, dtype=flow.dtype) for size in space_shape]
        points = tf.meshgrid(*ranges, indexing='ij')
        if indexing == 'xy':
            points = points[::-1]
        points = tf.stack(points, axis=-1)
        points = tf.reshape(points,
                            batch_dims * [1] + space_shape + [space_dims])
        points += flow

        return remap(grid, points, batch_dims=batch_dims,
                     border_mode=border_mode, border_value=border_value,
                     indexing=indexing)
