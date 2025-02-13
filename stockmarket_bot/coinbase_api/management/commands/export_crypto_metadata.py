# coinbase_api/management/commands/export_crypto_metadata.py
import os
import pandas as pd
from django.core.management.base import BaseCommand
from coinbase_api.models.models import CryptoMetadata
from coinbase_api.enums import ExportFolder, Database  # Adjust these as needed

class Command(BaseCommand):
    help = "Export all CryptoMetadata entries to a single Parquet file."

    def handle(self, *args, **options):
        # Define the folder where the metadata file will be stored.
        metadata_folder = ExportFolder.EXPORT_FOLDER.value  # or ExportFolder.METADATA_FOLDER.value if you have one
        if not os.path.exists(metadata_folder):
            os.makedirs(metadata_folder)
            self.stdout.write(f"Created export folder: {metadata_folder}")

        # Query all CryptoMetadata entries from the historical database (or default if appropriate)
        qs = CryptoMetadata.objects.using(Database.HISTORICAL.value).all().values("symbol", "earliest_date")
        df = pd.DataFrame(list(qs))
        if df.empty:
            self.stdout.write("No CryptoMetadata found. Exiting.")
            return

        # Ensure that earliest_date is a datetime type.
        df["earliest_date"] = pd.to_datetime(df["earliest_date"])
        # Optionally sort by symbol.
        df.sort_values("symbol", inplace=True)

        # Save the DataFrame to a single Parquet file.
        file_path = os.path.join(metadata_folder, "crypto_metadata.parquet")
        df.to_parquet(file_path)
        self.stdout.write(f"Saved CryptoMetadata to {file_path}")
