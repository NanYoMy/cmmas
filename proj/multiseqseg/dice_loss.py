import keras
import json
import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt
import tensorflow as tf

def single_class_dice_coefficient(y_true, y_pred, axis=(0, 1, 2),
                                  epsilon=0.00001):
    """
    Compute dice coefficient for single class.

    Args:
        y_true (Tensorflow tensor): tensor of ground truth values for single class.
                                    shape: (x_dim, y_dim, z_dim)
        y_pred (Tensorflow tensor): tensor of predictions for single class.
                                    shape: (x_dim, y_dim, z_dim)
        axis (tuple): spatial axes to sum over when computing numerator and
                      denominator of dice coefficient.
                      Hint: pass this as the 'axis' argument to the K.sum function.
        epsilon (float): small constant added to numerator and denominator to
                        avoid divide by 0 errors.
    Returns:
        dice_coefficient (float): computed value of dice coefficient.
    """
    dice_numerator = 2. * tf.sum(y_true * y_pred, axis=axis) + epsilon
    dice_denominator = tf.sum(y_true, axis=axis) + tf.sum(y_pred, axis=axis) + epsilon
    dice_coefficient = (dice_numerator) / (dice_denominator)

    return dice_coefficient


def dice_coefficient(y_true, y_pred, axis=(1, 2, 3),
                     epsilon=0.00001):
    """
    Compute mean dice coefficient over all abnormality classes.

    Args:
        y_true (Tensorflow tensor): tensor of ground truth values for all classes.
                                    shape: (num_classes, x_dim, y_dim, z_dim)
        y_pred (Tensorflow tensor): tensor of predictions for all classes.
                                    shape: (num_classes, x_dim, y_dim, z_dim)
        axis (tuple): spatial axes to sum over when computing numerator and
                      denominator of dice coefficient.
                      Hint: pass this as the 'axis' argument to the K.sum
                            and K.mean functions.
        epsilon (float): small constant add to numerator and denominator to
                        avoid divide by 0 errors.
    Returns:
        dice_coefficient (float): computed value of dice coefficient.
    """

    dice_numerator = 2. * tf.sum(y_true * y_pred, axis=axis) + epsilon
    dice_denominator = tf.sum(y_true, axis=axis) + tf.sum(y_pred, axis=axis) + epsilon
    dice_coefficient = tf.mean((dice_numerator) / (dice_denominator))

    return dice_coefficient

def soft_dice_loss(y_true, y_pred, axis=(1, 2, 3),
                   epsilon=0.00001):
    """
    Compute mean soft dice loss over all abnormality classes.

    Args:
        y_true (Tensorflow tensor): tensor of ground truth values for all classes.
                                    shape: (num_classes, x_dim, y_dim, z_dim)
        y_pred (Tensorflow tensor): tensor of soft predictions for all classes.
                                    shape: (num_classes, x_dim, y_dim, z_dim)
        axis (tuple): spatial axes to sum over when computing numerator and
                      denominator in formula for dice loss.
                      Hint: pass this as the 'axis' argument to the K.sum
                            and K.mean functions.
        epsilon (float): small constant added to numerator and denominator to
                        avoid divide by 0 errors.
    Returns:
        dice_loss (float): computed value of dice loss.
    """
    dice_numerator = 2. * tf.reduce_sum(y_true * y_pred, axis=axis) + epsilon
    dice_denominator = tf.reduce_sum(y_true, axis=axis) + tf.reduce_sum(y_pred, axis=axis) + epsilon
    dice_loss = 1 - tf.reduce_mean(tf.reduce_mean((dice_numerator)/(dice_denominator),axis=-1),axis=-1)
    return dice_loss