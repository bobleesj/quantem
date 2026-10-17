"""Fixed-size Torch FFT preserving the tutorial's NumPy transform order.

The radix-2/radix-8 butterflies and twiddle decomposition are adapted from
pocketfft, NumPy 2.4.6 revision 33ae5dc94c9cdc7f1c78346504a85de87cadaa12.
The upstream BSD-3-Clause license is reproduced below.
"""

import math

import torch


def _twiddle128(index: int) -> tuple[float, float]:
    """Construct data-independent coefficients using pocketfft's decomposition."""

    def coefficients(index: int) -> tuple[float, float]:
        index *= 8
        angle = math.pi / 512
        if index < 256:
            if index < 128:
                return math.cos(index * angle), math.sin(index * angle)
            return math.sin((256 - index) * angle), math.cos((256 - index) * angle)
        index -= 256
        if index < 128:
            return -math.sin(index * angle), math.cos(index * angle)
        return -math.cos((256 - index) * angle), math.sin((256 - index) * angle)

    conjugate = index > 64
    if conjugate:
        index = 128 - index
    first_real, first_imag = coefficients(index & 15)
    second_real, second_imag = coefficients((index >> 4) * 16)
    real = first_real * second_real - first_imag * second_imag
    imag = first_real * second_imag + first_imag * second_real
    return real, -imag if conjugate else imag


def _fft128(values_t: torch.Tensor, dim: int = -1, inverse: bool = False) -> torch.Tensor:
    """Apply the reference 128-point butterflies without moving data off device.

    The tutorial rounds each forward axis to complex64 and evaluates its shifted
    inverse in complex128. Retain that rounding boundary at this function's exit.
    This is a reference arithmetic path, not a replacement for general FFTs.
    """
    output_dtype = (
        torch.complex64 if values_t.dtype in (torch.float32, torch.complex64) else torch.complex128
    )
    data_t = values_t.movedim(dim, -1).to(torch.complex128)
    if inverse:
        data_t = data_t.conj().resolve_conj()
    shape = data_t.shape

    # Keep real and imaginary products separate to preserve pocketfft's rounding.
    def add(first, second):
        return first[0] + second[0], first[1] + second[1]

    def subtract(first, second):
        return first[0] - second[0], first[1] - second[1]

    def rotate90(value):
        return value[1], -value[0]

    def rotate45(value):
        scale = 0.7071067811865475244
        return scale * (value[0] + value[1]), scale * (value[1] - value[0])

    def rotate135(value):
        scale = 0.7071067811865475244
        return scale * (value[1] - value[0]), scale * (-value[0] - value[1])

    groups = 1
    for radix in (2, 8, 8):
        inner_size = 128 // (groups * radix)
        real_t = data_t.real.reshape(*shape[:-1], groups, radix, inner_size)
        imag_t = data_t.imag.reshape(*shape[:-1], groups, radix, inner_size)
        values = [(real_t[..., index, :], imag_t[..., index, :]) for index in range(radix)]
        if radix == 2:
            transformed = [add(values[0], values[1]), subtract(values[0], values[1])]
        else:
            # The numbered pairs follow pocketfft's pass8 butterfly dependency order.
            pair1, pair5 = add(values[1], values[5]), subtract(values[1], values[5])
            pair3, pair7 = add(values[3], values[7]), subtract(values[3], values[7])
            pair1, pair3 = add(pair1, pair3), rotate90(subtract(pair1, pair3))
            pair7 = rotate90(pair7)
            pair5, pair7 = rotate45(add(pair5, pair7)), rotate135(subtract(pair5, pair7))
            pair0, pair4 = add(values[0], values[4]), subtract(values[0], values[4])
            pair2, pair6 = add(values[2], values[6]), subtract(values[2], values[6])
            pair0, pair2 = add(pair0, pair2), subtract(pair0, pair2)
            pair6 = rotate90(pair6)
            pair4, pair6 = add(pair4, pair6), subtract(pair4, pair6)
            transformed = [
                add(pair0, pair1),
                add(pair4, pair5),
                add(pair2, pair3),
                add(pair6, pair7),
                subtract(pair0, pair1),
                subtract(pair4, pair5),
                subtract(pair2, pair3),
                subtract(pair6, pair7),
            ]
        if inner_size > 1:
            for index in range(1, radix):
                weights_t = torch.tensor(
                    [_twiddle128(index * groups * position) for position in range(inner_size)],
                    device=data_t.device,
                    dtype=torch.float64,
                )
                real_t, imag_t = transformed[index]
                transformed[index] = (
                    real_t * weights_t[:, 0] + imag_t * weights_t[:, 1],
                    imag_t * weights_t[:, 0] - real_t * weights_t[:, 1],
                )
        real_t = torch.stack([value[0] for value in transformed], dim=-3)
        imag_t = torch.stack([value[1] for value in transformed], dim=-3)
        data_t = torch.complex(real_t, imag_t).reshape(shape)
        groups *= radix
    if inverse:
        data_t = data_t.conj().resolve_conj() * (1 / 128)
    return data_t.to(output_dtype).movedim(-1, dim)


#
# This file is part of pocketfft.
#
# Copyright (C) 2010-2022 Max-Planck-Society
# Copyright (C) 2019-2020 Peter Bell
#
# For the odd-sized DCT-IV transforms:
#   Copyright (C) 2003, 2007-14 Matteo Frigo
#   Copyright (C) 2003, 2007-14 Massachusetts Institute of Technology
#
# For the prev_good_size search:
#   Copyright (C) 2024 Tan Ping Liang, Peter Bell
#
# Authors: Martin Reinecke, Peter Bell
#
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright notice, this
#   list of conditions and the following disclaimer in the documentation and/or
#   other materials provided with the distribution.
# * Neither the name of the copyright holder nor the names of its contributors may
#   be used to endorse or promote products derived from this software without
#   specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
# ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
# ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
