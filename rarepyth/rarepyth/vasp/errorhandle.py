# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 10:20:32 2024

@author: Wang Junhao
"""


class VaspJobError(Exception):
    def __init__(self, error):
        self.error = error

    def __str__(self):
        return self.error


class ReadWavecarError(VaspJobError):
    def __init__(self):
        self.error = 'The WAVECAR provided does not match this job.'


class FexcpError(VaspJobError):
    def __init__(self):
        self.error = 'A FEXCP error occured. This could be due to POTCAR mismatch, or electronic steps being too poor, or the structure is just too unreasonable. '


class ZhegvError(VaspJobError):
    def __init__(self):
        self.error = 'A ZHEGV error occured. Check ALGO and structure.'


class ZbrentError(VaspJobError):
    def __init__(self):
        self.error = 'A ZBRENT error occured. Try to rise accruacy of electronic stpes or just copy CONTCAR to POSCAR and continue.'


class VaspInterruptError(VaspJobError):
    def __init__(self):
        self.error = 'Job terminated accidentally without any error info.'


class NotConvergedError(VaspJobError):
    def __init__(self, count=1):
        self.count = count
        if count == 1:
            self.error = 'Electronic steps converge failed.'
        else:
            self.error = f'{count} times of electronic inconvergence detected.'


class SlabJobError(Exception):
    def __init__(self, error):
        self.error = error

    def __str__(self):
        return self.error


class StaticConvergeFailedError(SlabJobError):
    def __init__(self):
        self.error = 'Oops, static slab convergence failed!'


class RelaxedConvergeFailedError(SlabJobError):
    def __init__(self):
        self.error = 'Oops, relaxed slab convergence failed!'
