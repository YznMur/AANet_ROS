from __future__ import division

import warnings
import math

import torch
import numpy

# epsilon for testing whether a number is close to zero
_EPS = numpy.finfo(float).eps * 4.0

# axis sequences for Euler angles
_NEXT_AXIS = [1, 2, 0, 1]

# map axes strings to/from tuples of inner axis, parity, repetition, frame
_AXES2TUPLE = {
    'sxyz': (0, 0, 0, 0), 'sxyx': (0, 0, 1, 0), 'sxzy': (0, 1, 0, 0),
    'sxzx': (0, 1, 1, 0), 'syzx': (1, 0, 0, 0), 'syzy': (1, 0, 1, 0),
    'syxz': (1, 1, 0, 0), 'syxy': (1, 1, 1, 0), 'szxy': (2, 0, 0, 0),
    'szxz': (2, 0, 1, 0), 'szyx': (2, 1, 0, 0), 'szyz': (2, 1, 1, 0),
    'rzyx': (0, 0, 0, 1), 'rxyx': (0, 0, 1, 1), 'ryzx': (0, 1, 0, 1),
    'rxzx': (0, 1, 1, 1), 'rxzy': (1, 0, 0, 1), 'ryzy': (1, 0, 1, 1),
    'rzxy': (1, 1, 0, 1), 'ryxy': (1, 1, 1, 1), 'ryxz': (2, 0, 0, 1),
    'rzxz': (2, 0, 1, 1), 'rxyz': (2, 1, 0, 1), 'rzyz': (2, 1, 1, 1)}

_TUPLE2AXES = dict((v, k) for k, v in _AXES2TUPLE.items())

def euler_from_quaternion(quaternion, axes='sxyz'):
    """Return Euler angles from quaternion for specified axis sequence.
    >>> angles = euler_from_quaternion([0.06146124, 0, 0, 0.99810947])
    >>> numpy.allclose(angles, [0.123, 0, 0])
    True
    """
    return euler_from_matrix(quaternion_matrix(quaternion), axes)

def euler_from_matrix(matrix, axes='sxyz', iftorch=False):
    """Return Euler angles from rotation matrix for specified axis sequence.
    axes : One of 24 axis sequences as string or encoded tuple
    Note that many Euler angle triplets can describe one matrix.
    >>> R0 = euler_matrix(1, 2, 3, 'syxz')
    >>> al, be, ga = euler_from_matrix(R0, 'syxz')
    >>> R1 = euler_matrix(al, be, ga, 'syxz')
    >>> numpy.allclose(R0, R1)
    True
    >>> angles = (4.0*math.pi) * (numpy.random.random(3) - 0.5)
    >>> for axes in _AXES2TUPLE.keys():
    ...    R0 = euler_matrix(axes=axes, *angles)
    ...    R1 = euler_matrix(axes=axes, *euler_from_matrix(R0, axes))
    ...    if not numpy.allclose(R0, R1): print axes, "failed"
    """
    try:
        firstaxis, parity, repetition, frame = _AXES2TUPLE[axes.lower()]
    except (AttributeError, KeyError):
        _ = _TUPLE2AXES[axes]
        firstaxis, parity, repetition, frame = axes

    i = firstaxis
    j = _NEXT_AXIS[i+parity]
    k = _NEXT_AXIS[i-parity+1]
    
    if iftorch == False:
        M = numpy.array(matrix, dtype=numpy.float64, copy=False)[:3, :3]
    else:
        M = matrix[:3, :3]

    if repetition:
        sy = math.sqrt(M[i, j]*M[i, j] + M[i, k]*M[i, k])
        if sy > _EPS:
            ax = math.atan2( M[i, j],  M[i, k])
            ay = math.atan2( sy,       M[i, i])
            az = math.atan2( M[j, i], -M[k, i])
        else:
            ax = math.atan2(-M[j, k],  M[j, j])
            ay = math.atan2( sy,       M[i, i])
            az = 0.0
    else:
        cy = math.sqrt(M[i, i]*M[i, i] + M[j, i]*M[j, i])
        if cy > _EPS:
            ax = math.atan2( M[k, j],  M[k, k])
            ay = math.atan2(-M[k, i],  cy)
            az = math.atan2( M[j, i],  M[i, i])
        else:
            ax = math.atan2(-M[j, k],  M[j, j])
            ay = math.atan2(-M[k, i],  cy)
            az = 0.0

    if parity:
        ax, ay, az = -ax, -ay, -az
    if frame:
        ax, az = az, ax
    return ax, ay, az

def quaternion_matrix(quaternion, iftorch=False):
    """Return homogeneous rotation matrix from quaternion.
    >>> R = quaternion_matrix([0.06146124, 0, 0, 0.99810947])
    >>> numpy.allclose(R, rotation_matrix(0.123, (1, 0, 0)))
    True
    """
    
    if iftorch == False:
        q = numpy.array(quaternion[:4], dtype=numpy.float64, copy=True)
        nq = numpy.dot(q, q)
    else:
        q = quaternion[:4]
        nq = torch.dot(q, q)
    
    if nq < _EPS:
        if iftorch == False:
            return numpy.identity(4)
        else:
            return torch.diagonal(4)

    q *= math.sqrt(2.0 / nq)
    
    if iftorch==False:
        q = numpy.outer(q, q)
    else:
        q = torch.einsum('i,j->ij', q, q) # torch.outer(q, q)

    if iftorch == False:
        return numpy.array((
            (1.0-q[1, 1]-q[2, 2],     q[0, 1]-q[2, 3],     q[0, 2]+q[1, 3], 0.0),
            (    q[0, 1]+q[2, 3], 1.0-q[0, 0]-q[2, 2],     q[1, 2]-q[0, 3], 0.0),
            (    q[0, 2]-q[1, 3],     q[1, 2]+q[0, 3], 1.0-q[0, 0]-q[1, 1], 0.0),
            (                0.0,                 0.0,                 0.0, 1.0)
            ), dtype=numpy.float64)
    else:
        return torch.tensor((
            (1.0-q[1, 1]-q[2, 2],     q[0, 1]-q[2, 3],     q[0, 2]+q[1, 3], 0.0),
            (    q[0, 1]+q[2, 3], 1.0-q[0, 0]-q[2, 2],     q[1, 2]-q[0, 3], 0.0),
            (    q[0, 2]-q[1, 3],     q[1, 2]+q[0, 3], 1.0-q[0, 0]-q[1, 1], 0.0),
            (                0.0,                 0.0,                 0.0, 1.0)
            ), dtype=torch.float16, device="cuda")

def quaternion_from_euler(ai, aj, ak, axes='sxyz', iftorch=False):
    """Return quaternion from Euler angles and axis sequence.
    ai, aj, ak : Euler's roll, pitch and yaw angles
    axes : One of 24 axis sequences as string or encoded tuple
    >>> q = quaternion_from_euler(1, 2, 3, 'ryxz')
    >>> numpy.allclose(q, [0.310622, -0.718287, 0.444435, 0.435953])
    True
    """
    try:
        firstaxis, parity, repetition, frame = _AXES2TUPLE[axes.lower()]
    except (AttributeError, KeyError):
        _ = _TUPLE2AXES[axes]
        firstaxis, parity, repetition, frame = axes

    i = firstaxis
    j = _NEXT_AXIS[i+parity]
    k = _NEXT_AXIS[i-parity+1]

    if frame:
        ai, ak = ak, ai
    if parity:
        aj = -aj

    ai /= 2.0
    aj /= 2.0
    ak /= 2.0
    ci = math.cos(ai)
    si = math.sin(ai)
    cj = math.cos(aj)
    sj = math.sin(aj)
    ck = math.cos(ak)
    sk = math.sin(ak)
    cc = ci*ck
    cs = ci*sk
    sc = si*ck
    ss = si*sk

    if iftorch == False:
        quaternion = numpy.empty((4, ), dtype=numpy.float64)
    else:
        quaternion = torch.empty((4, ), dtype=torch.float16, device="cuda")

    if repetition:
        quaternion[i] = cj*(cs + sc)
        quaternion[j] = sj*(cc + ss)
        quaternion[k] = sj*(cs - sc)
        quaternion[3] = cj*(cc - ss)
    else:
        quaternion[i] = cj*sc - sj*cs
        quaternion[j] = cj*ss + sj*cc
        quaternion[k] = cj*cs - sj*sc
        quaternion[3] = cj*cc + sj*ss
    if parity:
        quaternion[j] *= -1

    return quaternion


def euler_to_matrix(angles, coords):
    assert len(angles) == len(coords)

    num_samples = len(angles)
    
    coss = numpy.cos(angles)
    sins =numpy.sin(angles)
    nels = numpy.zeros(len(angles))
    ones = numpy.ones(len(angles))

    Rz = numpy.stack((numpy.stack((coss, sins, nels)),
                     numpy.stack((-sins, coss, nels)),
                     numpy.stack((nels, nels, ones)))).T
    
    # coords = torch.cat((coords, ones.unsqueeze(1)), dim=-1)
    # out = torch.cat((Rz, coords.unsqueeze(-1)), dim=-1)

    return Rz #out

# def euler_from_matrix_vec(matrix):
#     pitch = torch.atan2(matrix[:, 2, 1], torch.sqrt(torch.pow(matrix[:, 0, 0], 2) + torch.pow(matrix[:, 1, 0], 2)))

#     deg_pos_90 = torch.tensor([1.5707], device="cuda", dtype = torch.half).repeat(len(pitch))
#     deg_neg_90 = torch.tensor([+1.5707], device="cuda", dtype = torch.half).repeat(len(pitch))
    
#     pos_90_mask = deg_pos_90 == pitch
#     neg_90_mask = deg_neg_90 == pitch

#     not_90 = torch.logical_and(~pos_90_mask, ~neg_90_mask)

#     yaw_not_90 = torch.atan2(matrix[:, 1, 0], matrix[:, 0, 0])
#     yaw_pos_90 = torch.atan2(matrix[:, 1, 2], matrix[:, 0, 2])
#     yaw_neg_90 = torch.atan2(-matrix[:, 1, 2], -matrix[:, 0, 2])

#     yaw = torch.zeros(pitch.shape, device="cuda", dtype = torch.half)

#     yaw[pos_90_mask] = yaw_pos_90[pos_90_mask]
#     yaw[not_90] = yaw_not_90[not_90]
#     yaw[neg_90_mask] = yaw_neg_90[neg_90_mask]

#     return yaw


def euler_from_matrix_vec(matrix):
    pitch = numpy.arctan2(matrix[:, 2, 1], numpy.sqrt(numpy.power(matrix[:, 0, 0], 2) + numpy.power(matrix[:, 1, 0], 2)))

    deg_pos_90 = [1.5707]*len(pitch)
    deg_neg_90 = [+1.5707]*len(pitch)
    
    pos_90_mask = deg_pos_90 == pitch
    neg_90_mask = deg_neg_90 == pitch

    not_90 = numpy.logical_and(~pos_90_mask, ~neg_90_mask)

    yaw_not_90 = numpy.arctan2(matrix[:, 1, 0], matrix[:, 0, 0])
    yaw_pos_90 = numpy.arctan2(matrix[:, 1, 2], matrix[:, 0, 2])
    yaw_neg_90 = numpy.arctan2(-matrix[:, 1, 2], -matrix[:, 0, 2])

    yaw = numpy.zeros(pitch.shape)

    yaw[pos_90_mask] = yaw_pos_90[pos_90_mask]
    yaw[not_90] = yaw_not_90[not_90]
    yaw[neg_90_mask] = yaw_neg_90[neg_90_mask]

    return yaw
