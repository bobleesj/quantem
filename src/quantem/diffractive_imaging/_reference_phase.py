# SPDX-License-Identifier: LGPL-2.1-or-later
# Derived from the IBM Accurate Mathematical Library in GNU libc.
# Copyright (C) 2001-2026 Free Software Foundation, Inc.
#
# This file is free software: you can redistribute it and/or modify it under
# the terms of the GNU Lesser General Public License, version 2.1 or later.
# It is distributed WITHOUT ANY WARRANTY; without even the implied warranty
# of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See
# LICENSES/LGPL-2.1-or-later-glibc.txt for the license and source provenance.
#
# Local scientific reference compatibility port, not a general replacement
# for device trigonometric functions. The rest of this repository retains its
# existing license; this derived module has the license specified above.
"""Scratch Torch port of glibc scalar sincos, finite arguments |x| < 1e8.

Algorithm and constants from glibc IBM Accurate Mathematical Library,
LGPL-2.1-or-later: sysdeps/ieee754/dbl-64/{s_sin.c,s_sincos.c,usncs.h,sincostab.c}.
"""

import torch

s1 = float.fromhex("-0x1.5555555555555p-3")
s2 = float.fromhex("0x1.1111111110ecep-7")
s3 = float.fromhex("-0x1.a01a019db08b8p-13")
s4 = float.fromhex("0x1.71de27b9a7ed9p-19")
s5 = float.fromhex("-0x1.addffc2fcdf59p-26")
aa = float.fromhex("-0x1.5558000000000p-3")
bb = float.fromhex("0x1.5555555556e24p-18")
big = float.fromhex("0x1.8000000000000p+45")
hp0 = float.fromhex("0x1.921fb54442d18p+0")
hp1 = float.fromhex("0x1.1a62633145c07p-54")
mp1 = float.fromhex("0x1.921fb58000000p+0")
mp2 = float.fromhex("-0x1.dde973c000000p-27")
mp3 = float.fromhex("-0x1.cb3b399d747f2p-55")
pp3 = float.fromhex("-0x1.cb3b398000000p-55")
pp4 = float.fromhex("-0x1.d747f23e32ed7p-83")
hpinv = float.fromhex("0x1.45f306dc9c883p-1")
toint = float.fromhex("0x1.8000000000000p+52")

# High and low parts of sin/cos at multiples of 1/128 radian.
TABLE = tuple(
    float.fromhex(value)
    for value in (
        "0x0.0p+0",
        "0x0.0p+0",
        "0x1.0000000000000p+0",
        "0x0.0p+0",
        "0x1.fffeaaaaeeeefp-8",
        "-0x1.e45e2ec67b77cp-62",
        "0x1.fffc000155552p-1",
        "0x1.f4a01a0196daep-55",
        "0x1.fffaaaaeeeed5p-7",
        "-0x1.2ab639a9f0777p-63",
        "0x1.fff000155549fp-1",
        "0x1.28a28a03a5ef3p-55",
        "0x1.7ff7001033255p-6",
        "0x1.efe2b51527336p-64",
        "0x1.ffdc006bff7e6p-1",
        "0x1.ae6dae86977bdp-55",
        "0x1.ffeaaaeeee86fp-6",
        "-0x1.cd406fb224ae2p-60",
        "0x1.ffc00155527d3p-1",
        "-0x1.3b54492d89b5bp-55",
        "0x1.3feb2b12d45d5p-5",
        "0x1.4ec54203d1c11p-60",
        "0x1.ff9c03414a7bap-1",
        "0x1.991f4be6c59bfp-57",
        "0x1.7fdc01032fba9p-5",
        "-0x1.599bdf46e997ap-59",
        "0x1.ff7006bfdf99fp-1",
        "-0x1.8b3b560648d5fp-56",
        "0x1.bfc6d78586dacp-5",
        "0x1.8e4fd03dbf236p-62",
        "0x1.ff3c0c8103a31p-1",
        "0x1.4856dbddc0e66p-56",
        "0x1.ffaaaeeed4edbp-5",
        "-0x1.2d16d32684b69p-59",
        "0x1.ff0015549f4d3p-1",
        "0x1.328387b99426fp-55",
        "0x1.1fc343d808befp-4",
        "-0x1.f3d32e6f3be4fp-58",
        "0x1.febc222a8ef9fp-1",
        "0x1.7934934f54c77p-58",
        "0x1.3facb12d1755bp-4",
        "-0x1.921915299468cp-58",
        "0x1.fe7034129ef6fp-1",
        "-0x1.cbf4337c96f97p-57",
        "0x1.5f911fd10b737p-4",
        "-0x1.0184f02be9102p-58",
        "0x1.fe1c4c3c873ebp-1",
        "-0x1.5a9c9057c4a02p-60",
        "0x1.7f701032550e4p-4",
        "0x1.afc2d1800501ap-60",
        "0x1.fdc06bf7e6b9bp-1",
        "0x1.31902b535f8dbp-55",
        "0x1.9f4902d55d1f9p-4",
        "0x1.2696d7eac1dc1p-58",
        "0x1.fd5c94b43e000p-1",
        "-0x1.2e768cb4f92f9p-57",
        "0x1.bf1b78568391dp-4",
        "0x1.e91841dea4cc8p-58",
        "0x1.fcf0c800e99b1p-1",
        "0x1.ea3d786d186acp-57",
        "0x1.dee6f16c1cce6p-4",
        "-0x1.50f8e2fb71673p-59",
        "0x1.fc7d078d1bc88p-1",
        "0x1.075d2447db685p-55",
        "0x1.feaaeee86ee36p-4",
        "-0x1.afcb2bcc6f03bp-59",
        "0x1.fc015527d5bd3p-1",
        "0x1.b68f35094efb8p-55",
        "0x1.0f3378ddd71d1p-3",
        "0x1.d8468724f0f9ep-57",
        "0x1.fb7db2bfe0695p-1",
        "0x1.21dadf4f65ab1p-55",
        "0x1.1f0d3d7afceafp-3",
        "-0x1.6ef95099769a5p-57",
        "0x1.faf22263c4bd3p-1",
        "-0x1.52ace133a2769p-58",
        "0x1.2ee285e4ab88fp-3",
        "-0x1.e4d0f05dee058p-57",
        "0x1.fa5ea641c36f2p-1",
        "0x1.04da6ed17cc7cp-59",
        "0x1.3eb312c5d66cbp-3",
        "0x1.47d666b66cb91p-57",
        "0x1.f9c340a7cc428p-1",
        "0x1.c5b6b063b7462p-55",
        "0x1.4e7ea4dc5f27bp-3",
        "0x1.949db2ac072fcp-58",
        "0x1.f91ff40374d01p-1",
        "-0x1.7d03f4d3a9e4cp-57",
        "0x1.5e44fcfa126f3p-3",
        "-0x1.6f443063f89b6p-57",
        "0x1.f874c2e1eecf6p-1",
        "-0x1.c6514e1332b16p-55",
        "0x1.6e05dc05a4d4cp-3",
        "-0x1.32c5c8b81c940p-66",
        "0x1.f7c1afeffde24p-1",
        "-0x1.8f55bc47540b1p-56",
        "0x1.7dc102fbaf2b5p-3",
        "0x1.5ab50e23c97c3p-59",
        "0x1.f706bdf9ece1cp-1",
        "-0x1.698c80c36dcb4p-55",
        "0x1.8d7632efaa944p-3",
        "-0x1.20fa262cbb953p-57",
        "0x1.f643efeb82acdp-1",
        "0x1.6b00ac1fe28acp-56",
        "0x1.9d252d0cec312p-3",
        "0x1.9c43d80b1137dp-58",
        "0x1.f57948cff6797p-1",
        "0x1.e3a0d3e03b1d5p-57",
        "0x1.accdb297a0765p-3",
        "-0x1.9883b57d6cdebp-58",
        "0x1.f4a6cbd1e3a79p-1",
        "0x1.13df0edaebb57p-55",
        "0x1.bc6f84edc6199p-3",
        "0x1.9c1a56a7b0cabp-57",
        "0x1.f3cc7c3b3d16ep-1",
        "-0x1.21a3ad28a3494p-57",
        "0x1.cc0a6588289a3p-3",
        "-0x1.868d09bc87c6bp-57",
        "0x1.f2ea5d753ffedp-1",
        "0x1.cc4215f56d583p-55",
        "0x1.db9e15fb5a5d0p-3",
        "-0x1.32e20d6cc6fc2p-57",
        "0x1.f20073086649fp-1",
        "0x1.b940416c1984bp-56",
        "0x1.eb2a57f8ae5a3p-3",
        "-0x1.0be06af572cebp-57",
        "0x1.f10ec09c5873bp-1",
        "0x1.d9072762c1283p-55",
        "0x1.faaeed4f31577p-3",
        "-0x1.15d88508e32b8p-57",
        "0x1.f01549f7deea1p-1",
        "0x1.d3c1e99e5cafdp-55",
        "0x1.0515cbf65155cp-2",
        "-0x1.9b8c29dfd8ec8p-56",
        "0x1.ef141300d2f26p-1",
        "-0x1.2aa1b08ded372p-55",
        "0x1.0cd00cef36436p-2",
        "-0x1.9fb0a0c93e2b5p-56",
        "0x1.ee0b1fbc0f11cp-1",
        "-0x1.bfd2380bbc3b1p-59",
        "0x1.14861aa94ddebp-2",
        "-0x1.be881b5b615a4p-57",
        "0x1.ecfa744d5efa1p-1",
        "-0x1.56d0a4af541d0p-58",
        "0x1.1c37d64c6b876p-2",
        "0x1.46076fe0dcff5p-56",
        "0x1.ebe214f76efa8p-1",
        "-0x1.02f9f12ba543ep-55",
        "0x1.23e52111aaf36p-2",
        "-0x1.4f080334eff18p-56",
        "0x1.eac2061bbaf4fp-1",
        "0x1.2c1d53e94658dp-57",
        "0x1.2b8ddc43eb49fp-2",
        "0x1.1553899f2d807p-57",
        "0x1.e99a4c3a7cd83p-1",
        "-0x1.2264b1bc53ce8p-55",
        "0x1.3331e94049f87p-2",
        "0x1.e0cb6b40c302cp-56",
        "0x1.e86aebf29a9edp-1",
        "0x1.9397afdbb58a7p-55",
        "0x1.3ad129769d3d8p-2",
        "0x1.03d5504878398p-63",
        "0x1.e733ea0193d40p-1",
        "-0x1.6428b3546ce13p-55",
        "0x1.426b7e69ee697p-2",
        "-0x1.f09c75705c59fp-56",
        "0x1.e5f54b436e9d0p-1",
        "0x1.7eb0fd02fc8bcp-55",
        "0x1.4a00c9b0f3d20p-2",
        "0x1.823ba6bb08eadp-56",
        "0x1.e4af14b2a449cp-1",
        "-0x1.68ca02e8a6833p-55",
        "0x1.5190ecf68a77ap-2",
        "0x1.b357155eef0f3p-56",
        "0x1.e3614b680d6a5p-1",
        "-0x1.27793aa015237p-56",
        "0x1.591bc9fa2f597p-2",
        "0x1.7c74bac3fe0cbp-57",
        "0x1.e20bf49acd6c1p-1",
        "-0x1.660aec7ef636cp-58",
        "0x1.60a1429078775p-2",
        "0x1.b1fd80ba89133p-58",
        "0x1.e0af15a03dbcep-1",
        "0x1.fe8e702771ae6p-58",
        "0x1.682138a38d7f7p-2",
        "-0x1.d889202444aadp-56",
        "0x1.df4ab3ebd875ep-1",
        "-0x1.e2d8a7e6736c4p-55",
        "0x1.6f9b8e33a0255p-2",
        "0x1.42bc14ee9da0dp-56",
        "0x1.ddded50f228d6p-1",
        "-0x1.e80c8d42ba2bfp-57",
        "0x1.7710255764214p-2",
        "-0x1.6ead7314bb6cep-57",
        "0x1.dc6b7eb995912p-1",
        "0x1.4b364776dcd35p-58",
        "0x1.7e7ee03c86d4ep-2",
        "-0x1.b63bcdabf5af2p-56",
        "0x1.daf0b6b888e83p-1",
        "0x1.a249e2b5e5ceap-55",
        "0x1.85e7a12826949p-2",
        "0x1.8a40e9b5face0p-56",
        "0x1.d96e82f71a9dcp-1",
        "0x1.ff61bd5d2039dp-55",
        "0x1.8d4a4a774992fp-2",
        "0x1.44a02ea766326p-56",
        "0x1.d7e4e97e17b4ap-1",
        "-0x1.3b770352bed94p-57",
        "0x1.94a6be9f546c5p-2",
        "-0x1.69ce13e683f58p-56",
        "0x1.d653f073e4040p-1",
        "-0x1.76236434bec37p-55",
        "0x1.9bfce02e80510p-2",
        "0x1.09e39a320b0a4p-56",
        "0x1.d4bb9e1c619e0p-1",
        "0x1.f34bb77858f61p-55",
        "0x1.a34c91cc50ccap-2",
        "-0x1.a310e3b50cecdp-58",
        "0x1.d31bf8d8d7c06p-1",
        "0x1.e60dd3089cbddp-56",
        "0x1.aa95b63a09277p-2",
        "-0x1.6293eb13c0381p-57",
        "0x1.d1750727d94f0p-1",
        "0x1.0d52b1ec1a48ep-55",
        "0x1.b1d8305321617p-2",
        "-0x1.ae242cb99f519p-56",
        "0x1.cfc6cfa52ad9fp-1",
        "0x1.8b5b5508f2a0dp-55",
        "0x1.b913e30dbac43p-2",
        "-0x1.e38ad2f6c3ff1p-56",
        "0x1.ce115909a82e5p-1",
        "0x1.1f139bb31109ap-55",
        "0x1.c048b17b140a3p-2",
        "0x1.19fe6757e9fa7p-57",
        "0x1.cc54aa2b2972ep-1",
        "0x1.4ee162ba83a98p-57",
        "0x1.c7767ec7fd19ep-2",
        "-0x1.eb14d1a3d5826p-58",
        "0x1.ca90c9fc67d0bp-1",
        "-0x1.46a81485e3462p-57",
        "0x1.ce9d2e3d4a51fp-2",
        "-0x1.2fc8a12dae298p-57",
        "0x1.c8c5bf8ce1a84p-1",
        "0x1.ab3d1a1590123p-56",
        "0x1.d5bca34047661p-2",
        "0x1.28a44a75fc29cp-56",
        "0x1.c6f39208be53bp-1",
        "-0x1.741dbfbaadb42p-55",
        "0x1.dcd4c15329c9ap-2",
        "0x1.0d4c6e171fd9ap-56",
        "0x1.c51a48b8b175ep-1",
        "-0x1.1bbb43b9aa880p-57",
        "0x1.e3e56c1582a69p-2",
        "-0x1.0a4821099f88fp-58",
        "0x1.c339eb01ddd81p-1",
        "-0x1.caaf5ee82c5c0p-55",
        "0x1.eaee8744b05f0p-2",
        "-0x1.789b43c9b027dp-58",
        "0x1.c1528065b7d50p-1",
        "-0x1.892111312e828p-55",
        "0x1.f1eff6bc4f97bp-2",
        "0x1.17212f8a7525cp-56",
        "0x1.bf641081e7536p-1",
        "0x1.b7bd71628a9a1p-55",
        "0x1.f8e99e76abc97p-2",
        "0x1.9d950af2d00a3p-58",
        "0x1.bd6ea310294f5p-1",
        "0x1.31bbcc88c109dp-56",
        "0x1.ffdb628d2f57ap-2",
        "0x1.f4a992e905b6ap-57",
        "0x1.bb723fe630f32p-1",
        "0x1.72bd2452d0a39p-56",
        "0x1.0362939c69955p-1",
        "-0x1.2d8cd78397b01p-55",
        "0x1.b96eeef58840ep-1",
        "0x1.45a3cc78fade0p-58",
        "0x1.06d3686946e5bp-1",
        "0x1.3f5ae4538ff1bp-55",
        "0x1.b764b84b704c2p-1",
        "-0x1.f5848c21b389bp-55",
        "0x1.0a4021e9e1001p-1",
        "-0x1.6f643a13914f6p-55",
        "0x1.b553a410c104ep-1",
        "0x1.8ff7947027a16p-58",
        "0x1.0da8b26b5672ep-1",
        "-0x1.a58def0bee909p-55",
        "0x1.b33bba89c8948p-1",
        "0x1.ea6a51d1f6ca9p-55",
        "0x1.110d0c4b69c3bp-1",
        "0x1.d918998809981p-55",
        "0x1.b11d04162a4c6p-1",
        "0x1.1dd561efbc0c2p-56",
        "0x1.146d21f8b7f82p-1",
        "0x1.bf9535e2739a8p-56",
        "0x1.aef78930bd275p-1",
        "-0x1.f836279746f94p-56",
        "0x1.17c8e5f2eedb0p-1",
        "0x1.35e57102e2488p-57",
        "0x1.accb526f69de5p-1",
        "0x1.8fb6a8dd6b6ccp-55",
        "0x1.1b204acb02fddp-1",
        "-0x1.f190c70cbb5ffp-58",
        "0x1.aa98688308913p-1",
        "-0x1.b83d607cd5070p-63",
        "0x1.1e7343236574cp-1",
        "0x1.22a3fa4f41d5ap-56",
        "0x1.a85ed4373e02dp-1",
        "0x1.9be06385ec792p-57",
        "0x1.21c1c1b0394cfp-1",
        "0x1.e5b324b23aa31p-58",
        "0x1.a61e9e72586afp-1",
        "0x1.58330e2fd453fp-55",
        "0x1.250bb93788bbbp-1",
        "0x1.ea3d02457bccep-56",
        "0x1.a3d7d0352bdcfp-1",
        "-0x1.68dbaeca19669p-55",
        "0x1.28511c917a067p-1",
        "-0x1.01df1d9a16b70p-55",
        "0x1.a18a729aee445p-1",
        "0x1.95e25736c0358p-60",
        "0x1.2b91dea88421ep-1",
        "-0x1.fa371db216ab0p-55",
        "0x1.9f368ed912f85p-1",
        "-0x1.1d200c5791606p-55",
        "0x1.2ecdf279a3082p-1",
        "0x1.d3557e0e7e37ep-55",
        "0x1.9cdc2e3f25e5cp-1",
        "0x1.3f99112993f62p-55",
        "0x1.32054b148bc4fp-1",
        "0x1.f6b42095a135bp-55",
        "0x1.9a7b5a36a6514p-1",
        "0x1.722cfcc9fa7a9p-55",
        "0x1.3537db9be0367p-1",
        "0x1.b327e7af040f0p-57",
        "0x1.98141c42e1310p-1",
        "0x1.d1ff80488f08dp-55",
        "0x1.386597456282bp-1",
        "-0x1.10fada93b07a8p-56",
        "0x1.95a67e00cb1fdp-1",
        "-0x1.0befda21f862dp-55",
        "0x1.3b8e715a2840ap-1",
        "-0x1.97653a7d2f07bp-56",
        "0x1.93328926d9e92p-1",
        "-0x1.bb77003600cdap-55",
        "0x1.3eb25d36cd53ap-1",
        "-0x1.be570e1570fc0p-58",
        "0x1.90b84784ddaf7p-1",
        "-0x1.0feb10ab93b87p-56",
        "0x1.41d14e4ba6790p-1",
        "0x1.4608fd287ecf5p-55",
        "0x1.8e37c303d9ad1p-1",
        "-0x1.463a4b53d4bf8p-57",
        "0x1.44eb381cf386bp-1",
        "-0x1.3ed6c1e6a5505p-55",
        "0x1.8bb105a5dc900p-1",
        "0x1.863e03e9474c1p-55",
        "0x1.48000e431159fp-1",
        "-0x1.b194a7463ed10p-55",
        "0x1.89241985d871fp-1",
        "0x1.c48d9c413ed84p-55",
        "0x1.4b0fc46aab761p-1",
        "0x1.0da05738cc59ap-61",
        "0x1.869108d77a6c6p-1",
        "0x1.338ffe2bfe9ddp-56",
        "0x1.4e1a4e54ed51bp-1",
        "-0x1.a492f89b7c76ap-55",
        "0x1.83f7dde701ca0p-1",
        "-0x1.152cf609bc6e8p-59",
        "0x1.511f9fd7b351cp-1",
        "-0x1.5c0e861c48831p-55",
        "0x1.8158a31916d5dp-1",
        "-0x1.de8b90b8228dep-57",
        "0x1.541facddbb724p-1",
        "0x1.232c28520d391p-56",
        "0x1.7eb362eaa1488p-1",
        "0x1.a1d65a4a5959fp-58",
        "0x1.571a6966d59b3p-1",
        "0x1.c843b4d0fb198p-58",
        "0x1.7c0827f09e54fp-1",
        "-0x1.c73d6d72aee68p-57",
        "0x1.5a0fc98813a12p-1",
        "-0x1.d82e2b7d4227bp-55",
        "0x1.7956fcd7f6543p-1",
        "-0x1.ab276e9d45ae4p-55",
        "0x1.5cffc16bf8f0dp-1",
        "0x1.96cb370eb578ap-55",
        "0x1.769fec655211fp-1",
        "-0x1.827d5cf8c68c5p-57",
        "0x1.5fea4552a9e57p-1",
        "0x1.0b6cef7ee20b7p-55",
        "0x1.73e30174efba1p-1",
        "-0x1.5d3ae3d94ad5fp-57",
        "0x1.62cf49921ac79p-1",
        "-0x1.edd9855b6241ap-55",
        "0x1.712046fa77678p-1",
        "0x1.425b0a5029c81p-55",
        "0x1.65aec2963e755p-1",
        "0x1.126f96b71053cp-55",
        "0x1.6e57c800cf55ep-1",
        "0x1.60286dedbd0a6p-55",
        "0x1.6888a4e134b2fp-1",
        "-0x1.6b7d37644d5e6p-55",
        "0x1.6b898fa9efb5dp-1",
        "0x1.15ac786ccf4b2p-56",
        "0x1.6b5ce50b7821ap-1",
        "-0x1.5d5158f702e0fp-57",
        "0x1.68b5a92eb6253p-1",
        "-0x1.9a91ad985f89cp-55",
        "0x1.6e2b77c40bde1p-1",
        "-0x1.0e729857fad53p-56",
        "0x1.65dc1fdeb8cbap-1",
        "-0x1.97c1b47337c77p-58",
        "0x1.70f451d0a8c40p-1",
        "0x1.97ede3885770dp-57",
        "0x1.62fcff20191c7p-1",
        "0x1.d9143895756efp-57",
        "0x1.73b7680dea578p-1",
        "-0x1.2248306dc12a2p-56",
        "0x1.6018526f563dfp-1",
        "0x1.46ca5e0e432d0p-55",
        "0x1.7674af6f7b524p-1",
        "0x1.e9d3f94ac84a8p-56",
        "0x1.5d2e255f1f17ap-1",
        "0x1.0314104c8892bp-55",
        "0x1.792c1d0041d52p-1",
        "-0x1.abf05eeb354ebp-55",
        "0x1.5a3e839824077p-1",
        "0x1.428aa2759be62p-55",
        "0x1.7bdda5e28b3c2p-1",
        "0x1.ad1197ccd0393p-59",
        "0x1.574978d8e83f2p-1",
        "0x1.f4714af282d23p-55",
        "0x1.7e893f5037959p-1",
        "0x1.0eefbaa650c4cp-55",
        "0x1.544f10f592ca5p-1",
        "-0x1.e7ae8e6c7a62fp-55",
        "0x1.812ede9ae4ba4p-1",
        "-0x1.7830adf402ddap-55",
        "0x1.514f57d7bf3dap-1",
        "0x1.47a108073c259p-56",
    )
)
sn3 = -1.66666666666664880952546298448555e-01
sn5 = 8.33333214285722277379541354343671e-03
cs2 = 4.99999999999999999999950396842453e-01
cs4 = -4.16666666666664434524222570944589e-02
cs6 = 1.38888874007937613028114285595617e-03


def _sincos(value_t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Evaluate glibc FMA sincos on device for finite float64 angles below 1e8."""
    if value_t.dtype != torch.float64:
        raise ValueError("Use a float64 angle tensor for exact scalar trig parity.")
    if not bool(torch.isfinite(value_t).all()) or bool((value_t.abs() >= 1e8).any()):
        raise ValueError("This bounded port requires finite angles with abs(angle) < 1e8.")

    def fma(a, b, c):
        if not isinstance(a, torch.Tensor):
            a = value_t.new_tensor(a)
        if not isinstance(b, torch.Tensor):
            b = value_t.new_tensor(b)
        if not isinstance(c, torch.Tensor):
            c = value_t.new_tensor(c)
        return torch.addcmul(c, a, b)

    table_t = value_t.new_tensor(TABLE).reshape(-1, 4)

    def sin_small(x_t, dx_t):
        xx_t = x_t * x_t
        poly_t = fma(fma(fma(fma(s5, xx_t, s4), xx_t, s3), xx_t, s2), xx_t, s1)
        t_t = fma(fma(poly_t, x_t, -0.5 * dx_t), xx_t, dx_t)
        return x_t + t_t

    def do_sin(x_t, dx_t):
        original_t = x_t
        dx_t = torch.where(x_t <= 0, -dx_t, dx_t)
        u_t = big + x_t.abs()
        index_t = ((u_t - big) * 128).long().clamp(0, 109)
        x_t = x_t.abs() - (u_t - big)
        xx_t = x_t * x_t
        s_t = x_t + fma(x_t * xx_t, fma(xx_t, sn5, sn3), dx_t)
        c_t = fma(xx_t, fma(xx_t, fma(xx_t, cs6, cs4), cs2), x_t * dx_t)
        sn_t, ssn_t, cs_t, ccs_t = table_t[index_t].unbind(-1)
        cor_t = fma(cs_t, s_t, fma(-sn_t, c_t, fma(s_t, ccs_t, ssn_t)))
        result_t = torch.copysign(sn_t + cor_t, original_t)
        return torch.where(
            original_t.abs() < 0.126,
            sin_small(original_t, torch.where(original_t <= 0, -dx_t, dx_t)),
            result_t,
        )

    def do_cos(x_t, dx_t):
        dx_t = torch.where(x_t < 0, -dx_t, dx_t)
        u_t = big + x_t.abs()
        index_t = ((u_t - big) * 128).long().clamp(0, 109)
        x_t = x_t.abs() - (u_t - big) + dx_t
        xx_t = x_t * x_t
        s_t = fma(x_t * xx_t, fma(xx_t, sn5, sn3), x_t)
        c_t = xx_t * fma(xx_t, fma(xx_t, cs6, cs4), cs2)
        sn_t, ssn_t, cs_t, ccs_t = table_t[index_t].unbind(-1)
        cor_t = fma(-sn_t, s_t, fma(-cs_t, c_t, fma(-s_t, ssn_t, ccs_t)))
        return cs_t + cor_t

    x_t = value_t
    t_t = x_t * hpinv + toint
    xn_t = t_t - toint
    n_t = xn_t.long() % 4
    y_t = (x_t - xn_t * mp1) - xn_t * mp2
    t1_t = xn_t * pp3
    t2_t = y_t - t1_t
    db_t = (y_t - t2_t) - t1_t
    t1_t = xn_t * pp4
    a_t = t2_t - t1_t
    da_t = db_t + ((t2_t - a_t) - t1_t)
    negative_t = (n_t == 1) | (n_t == 2)
    a_t = torch.where(negative_t, -a_t, a_t)
    da_t = torch.where(negative_t, -da_t, da_t)
    sin_t = do_sin(a_t, da_t)
    cos_t = do_cos(a_t, da_t)
    cos_t = torch.where(n_t & 2 != 0, -cos_t, cos_t)
    sin_t, cos_t = torch.where(n_t & 1 != 0, cos_t, sin_t), torch.where(n_t & 1 != 0, sin_t, cos_t)
    mid_y_t = hp0 - x_t.abs()
    mid_a_t = mid_y_t + hp1
    mid_da_t = (mid_y_t - mid_a_t) + hp1
    middle_t = x_t.abs() < float.fromhex("0x1.368fd00000000p1")
    sin_t = torch.where(middle_t, torch.copysign(do_cos(mid_a_t, mid_da_t), x_t), sin_t)
    cos_t = torch.where(middle_t, do_sin(mid_a_t, mid_da_t), cos_t)
    small_t = x_t.abs() < float.fromhex("0x1.b600000000000p-1")
    zero_t = torch.zeros_like(x_t)
    sin_t = torch.where(small_t, do_sin(x_t, zero_t), sin_t)
    cos_t = torch.where(small_t, do_cos(x_t, zero_t), cos_t)
    tiny_t = x_t.abs() < 2**-27
    return torch.where(tiny_t, x_t, sin_t), torch.where(tiny_t, torch.ones_like(x_t), cos_t)


def _phase_factor(angle_t: torch.Tensor) -> torch.Tensor:
    """Return exp(1j * angle) with the reference host's glibc FMA rounding.

    All angle arithmetic remains on the input device. This compatibility
    implementation accepts finite float64 angles with absolute value below
    1e8, and returns complex128 values. Exact parity has been checked against
    this project's Linux glibc FMA reference, not arbitrary host libm versions.
    """
    sine_t, cosine_t = _sincos(angle_t)
    return torch.complex(cosine_t, sine_t)
