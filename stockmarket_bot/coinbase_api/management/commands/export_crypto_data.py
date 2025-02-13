# coinbase_api/management/commands/export_crypto_data.py
import os
import pandas as pd
from django.core.management.base import BaseCommand
from coinbase_api.constants import crypto_models, simulation_columns
from coinbase_api.enums import Database, ExportFolder
# Global constant for folder location (adjust this path as needed)


class Command(BaseCommand):
    help = "Export historical data for each crypto to Parquet files."

    def handle(self, *args, **options):
        # Create the export folder if it does not exist
        if not os.path.exists(ExportFolder.EXPORT_FOLDER.value):
            os.makedirs(ExportFolder.EXPORT_FOLDER.value)
            self.stdout.write(f"Created export folder: {ExportFolder.EXPORT_FOLDER.value}")

        # List of crypto models to export
        columns = simulation_columns

        for crypto in crypto_models:
            self.stdout.write(f"Exporting data for {crypto.symbol}...")
            # Query all items and order by timestamp
            qs = crypto.objects.using(Database.HISTORICAL.value).all().order_by('timestamp').values(*columns)
            df = pd.DataFrame(list(qs))
            if df.empty:
                self.stdout.write(f"No data found for {crypto.symbol}. Skipping.")
                continue
            # Ensure timestamps are converted and set as index
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
            # Optionally, sort the index just to be sure
            df.sort_index(inplace=True)

            # Write the DataFrame to a Parquet file using the crypto symbol as the filename
            file_path = os.path.join(ExportFolder.EXPORT_FOLDER.value, f"{crypto.symbol}.parquet")
            df.to_parquet(file_path)
            self.stdout.write(f"Saved data for {crypto.symbol} to {file_path}")
