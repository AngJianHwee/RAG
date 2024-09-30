import base64


def to_base64(file):
    """
    Encode a binary file to base64 string
    """
    return base64.b64encode(file).decode()


def create_download_link(obj_storage_location, filename):
    """
    Generate a link to download the given file object
    """
    # read from the file object path
    with open(obj_storage_location, 'rb') as f:
        file = f.read()
    b64 = to_base64(file)
    href = f'<a href="data:application/octet-stream;base64,{b64}" download="{filename}">Download</a>'

    return href
