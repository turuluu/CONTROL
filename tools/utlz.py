import requests

def download_file(url, save_path=None):
    """
    Downloads a binary file from a URL and saves it locally.

    Args:
        url (str): The URL of the file to download.
        save_path (str, optional): Local path to save the file. If not provided, filename is derived from URL.

    Raises:
        Exception: If the download fails.
    """
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()

        # Derive filename from URL if not provided
        if save_path is None:
            local_filename = url.split("/")[-1] or "downloaded_file"
        else:
            local_filename = save_path

        with open(local_filename, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:  # filter out keep-alive chunks
                    f.write(chunk)

        return local_filename

    except Exception as e:
        print(f"Error downloading file: {e}")
        return None