
# Usa:
from google.cloud import storage
import io

def load_from_gcs(bucket_name, file_path):
    client = storage.Client()
    bucket = client.get_bucket(bucket_name)
    blob = bucket.blob(file_path)
    data = blob.download_as_bytes()
    return pd.read_csv(io.BytesIO(data))
