# -*- coding: utf-8 -*-
"""
Set of loss functions
"""
import tensorflow as tf

from .types import TensorLike


@tf.function
def npairs_loss(y_true: TensorLike, y_pred: TensorLike) -> tf.Tensor:
    """
    Computes the npairs loss between `y_true` and `y_pred`.
    Parameters
    ----------
    y_true: 1-D integer `Tensor` with shape `[batch_size]` of
        multi-class labels.
    y_pred: 2-D float `Tensor` with shape `[batch_size, batch_size]` of
        similarity matrix between embedding matrices.
    Returns
    -------
    npairs_loss: float scalar.
    """
    y_pred = tf.convert_to_tensor(y_pred)
    y_true = tf.cast(y_true, y_pred.dtype)

    # Expand to [batch_size, 1]
    y_true = tf.expand_dims(y_true, -1)
    y_true = tf.cast(tf.equal(y_true, tf.transpose(y_true)), y_pred.dtype)
    y_true /= tf.math.reduce_sum(y_true, 1, keepdims=True)

    loss = tf.nn.softmax_cross_entropy_with_logits(logits=y_pred, labels=y_true)

    return tf.math.reduce_mean(loss)


@tf.function
def npairs_multilabel_loss(y_true: TensorLike, y_pred: TensorLike) -> tf.Tensor:
    """
    Computes the npairs loss between multilabel data `y_true` and `y_pred`.
    Parameters
    ----------
    y_true: Either 2-D integer `Tensor` with shape
        `[batch_size, num_classes]`, or `SparseTensor` with dense shape
        `[batch_size, num_classes]`. If `y_true` is a `SparseTensor`, then
        it will be converted to `Tensor` via `tf.sparse.to_dense` first.
    y_pred: 2-D float `Tensor` with shape `[batch_size, batch_size]` of
        similarity matrix between embedding matrices.
    Returns
    -------
    npairs_multilabel_loss: float scalar.
    """
    y_pred = tf.convert_to_tensor(y_pred)
    y_true = tf.cast(y_true, y_pred.dtype)

    # Convert to dense tensor if `y_true` is a `SparseTensor`
    if isinstance(y_true, tf.SparseTensor):
        y_true = tf.sparse.to_dense(y_true)

    # Enable efficient multiplication because y_true contains lots of zeros
    y_true = tf.linalg.matmul(
        y_true, y_true, transpose_b=True, a_is_sparse=True, b_is_sparse=True
    )
    y_true /= tf.math.reduce_sum(y_true, 1, keepdims=True)

    loss = tf.nn.softmax_cross_entropy_with_logits(logits=y_pred, labels=y_true)

    return tf.math.reduce_mean(loss)
