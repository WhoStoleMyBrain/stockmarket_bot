# start project:
docker compose up -d --build

# setup
0. Set up a constants.py file (not to be commited to any VCS) with the following variables
    - API_KEY = CDP API Key
    - API_SECRET = CDP API Secret
    - all_crypto_models = list[AbstractOHLCV] all coinbase supported cryptos
    - crypto_models = list[AbstractOHLCV] cryptos the bot should use. See generated_models.py for options
    - crypto_features = list[str]. Indicators to be used by the bot. Need to be fields of the AbstractOHLCV table (see models.py)
    - crypto_predicted_features - Which of the predicted features (e.g. using XGBoost or LSTM) should be used.
    - crypto_extra_features - Additional features needed by XGBoost and LSTM for predictions. Currently is ['Hour', 'Day_of_Week', 'Day_of_Month', 'Month', 'Year', 'Is_Weekend']

1. run migrations on normal and historical database
    ```
    python manage.py migrate
    python manage.py migrate --database=historical
    ```
2. Initialize CryptoMetadata Table

    The following command will request data from the Coinbase API until no more data is being sent. The earliest timestamps is then stored in the CryptoMetadata table. This will help when initializing the database with all data, since we now know which timeframe needs to be filled with data for each crypto.
    ```
    python manage.py find_earliest_timestamps
    ```

3. Fetch all data from the coinbase API

    This will fetch OHLCV data from the database and add calculated parameters as well as AI predictions to the database.
    ```
    python manage.py trigger_historical_db_update
    ```