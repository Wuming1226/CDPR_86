def int_to_le_bytes(value: int):
    """
    Convert an integer to 4 little-endian bytes.

    Args:
        value (int): Integer value to convert.

    Returns:
        b[bytes]: Bytes.
    """
    data_bytes = value.to_bytes(4, byteorder='little', signed=True)
    return data_bytes


def le_bytes_to_int(data_bytes):
    value = int.from_bytes(data_bytes, byteorder='little', signed=True)
    return value


if __name__ == '__main__':

    # print(int_to_le_bytes(20))
    # data = int_to_le_bytes(20)
    # print(data)
    # print(data[0])

    data = bytes([0x00, 0x00, 0x00, 0x00, 0x78, 0x56, 0x34, 0x12])
    print(le_bytes_to_int(data[4:8]))
