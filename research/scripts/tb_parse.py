"""Minimal TFRecord + protobuf wire-format reader, no tensorflow/tensorboard
dependency needed. Reads length-delimited TFRecords (skips CRC verification,
trusts length fields) and walks the generic protobuf wire format for
tensorflow.Event / Summary / Summary.Value without needing compiled schemas.
"""
import struct
import glob
import sys
from collections import defaultdict

def read_varint(buf, pos):
    result = 0
    shift = 0
    while True:
        b = buf[pos]
        pos += 1
        result |= (b & 0x7f) << shift
        if not (b & 0x80):
            break
        shift += 7
    return result, pos

def parse_message(buf):
    fields = defaultdict(list)
    pos = 0
    n = len(buf)
    while pos < n:
        tag, pos = read_varint(buf, pos)
        field_num = tag >> 3
        wire_type = tag & 0x7
        if wire_type == 0:
            val, pos = read_varint(buf, pos)
        elif wire_type == 1:
            val = buf[pos:pos+8]; pos += 8
        elif wire_type == 2:
            length, pos = read_varint(buf, pos)
            val = buf[pos:pos+length]; pos += length
        elif wire_type == 5:
            val = buf[pos:pos+4]; pos += 4
        else:
            raise ValueError(f"unsupported wire type {wire_type} at pos {pos}")
        fields[field_num].append((wire_type, val))
    return fields

def read_tfrecords(path):
    with open(path, "rb") as f:
        data = f.read()
    pos = 0
    n = len(data)
    records = []
    while pos + 12 <= n:
        length = struct.unpack("<Q", data[pos:pos+8])[0]
        pos += 12  # 8-byte length + 4-byte length-crc
        if pos + length + 4 > n:
            break
        record = data[pos:pos+length]
        pos += length + 4  # data + 4-byte data-crc
        records.append(record)
    return records

def extract_scalar_from_value(value_fields):
    tag = None
    if 1 in value_fields:
        tag = value_fields[1][0][1].decode("utf-8", errors="replace")
    val = None
    if 2 in value_fields:  # simple_value: wire type 5, 4-byte float
        val = struct.unpack("<f", value_fields[2][0][1])[0]
    elif 8 in value_fields:  # tensor (TensorProto)
        tp = parse_message(value_fields[8][0][1])
        if 5 in tp:  # float_val, repeated float (wire type 5 each, or packed as bytes)
            wt, raw = tp[5][0]
            if wt == 5:
                val = struct.unpack("<f", raw)[0]
            elif wt == 2:
                val = struct.unpack("<f", raw[:4])[0]
        elif 4 in tp:  # tensor_content: raw bytes, little-endian float32
            raw = tp[4][0][1]
            if len(raw) >= 4:
                val = struct.unpack("<f", raw[:4])[0]
    return tag, val

def extract_series(path, wanted_tags=None):
    series = defaultdict(list)  # tag -> list of (step, wall_time, value)
    for rec in read_tfrecords(path):
        ev = parse_message(rec)
        wall_time = None
        if 1 in ev:
            wall_time = struct.unpack("<d", ev[1][0][1])[0]
        step = ev[2][0][1] if 2 in ev else None
        if 5 not in ev:  # no summary field
            continue
        summ = parse_message(ev[5][0][1])
        if 1 not in summ:
            continue
        for wt, raw in summ[1]:
            val_fields = parse_message(raw)
            tag, val = extract_scalar_from_value(val_fields)
            if tag is None or val is None:
                continue
            if wanted_tags is not None and tag not in wanted_tags:
                continue
            series[tag].append((step, wall_time, val))
    return series

if __name__ == "__main__":
    path = sys.argv[1]
    series = extract_series(path)
    print(f"{path}: found {len(series)} distinct scalar tags")
    for tag in sorted(series.keys()):
        print(f"  {tag}  (n={len(series[tag])})")
