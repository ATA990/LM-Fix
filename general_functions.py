"""LM‑Fix module: polished for GitHub publishing."""
import struct
import numpy as np
import torch

def flip_bit32(number, bit_position):
    packed = struct.pack('>f', number)
    int_rep = struct.unpack('>I', packed)[0]
    int_rep ^= 1 << int(bit_position)
    packed = struct.pack('>I', int_rep)
    return struct.unpack('>f', packed)[0]

def flip_bit16(number, bit_position):
    binary_representation = format(number.view('H'), '016b')
    integer_value = int(binary_representation, 2)
    integer_value ^= 1 << bit_position
    return np.frombuffer(np.array([integer_value], dtype=np.uint16).tobytes(), dtype=np.float16)[0]

def flip_bit8(number, bit_position, dType):
    binary_representation = format(number.view('H'), '016b')
    integer_value = int(binary_representation, 2)
    integer_value ^= 1 << bit_position
    return np.frombuffer(np.array([integer_value], dtype=np.uint8).tobytes(), dtype=dType)[0]

def flip_bit_fp8_e4m3(value: torch.Tensor, bit_position: int):
    """Flips a single bit in a torch.float8_e4m3fn tensor value"""
    value = value.to(dtype=torch.float8_e4m3fn)
    value_uint8 = value.view(torch.uint8)
    flipped_uint8 = value_uint8 ^ 1 << bit_position
    flipped_fp8 = flipped_uint8.view(torch.float8_e4m3fn)
    return flipped_fp8

def tensor_flip_bit(value: torch.Tensor, bit_position: int, dtype):
    """Flips a single bit in a torch.float8_e4m3fn tensor value"""
    value = value.to(dtype=dtype)
    uint_dtype_map = {torch.float8_e4m3fn: torch.uint8, torch.float8_e4m3fnuz: torch.uint8, torch.float8_e5m2: torch.uint8, torch.float8_e5m2fnuz: torch.uint8, torch.float16: torch.int16, torch.float32: torch.int32, torch.float64: torch.int64, torch.int8: torch.uint8, torch.int16: torch.int16, torch.int32: torch.int32, torch.int64: torch.int64, torch.uint4: torch.uint4, torch.uint8: torch.uint8, torch.uint16: torch.int16, torch.uint32: torch.int32, torch.uint64: torch.int64, torch.qint8: torch.uint8, torch.qint32: torch.int32, torch.quint8: torch.uint8, torch.bfloat16: torch.int16}
    if dtype in uint_dtype_map:
        bitwise_dtype = uint_dtype_map[dtype]
    else:
        raise TypeError(f'Unsupported dtype for bit flipping: {dtype}')
    value_uint = value.view(bitwise_dtype)
    flipped_uint = value_uint ^ 1 << bit_position
    flipped_out = flipped_uint.view(dtype)
    return flipped_out