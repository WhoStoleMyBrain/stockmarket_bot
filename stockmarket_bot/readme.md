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

    This will fetch OHLCV data from the Coinbase REST API. Due to rate limits and a maximum of 300 entries per request this might take a while, depending on setting (hourly data vs 5 minute data etc) and number of cryptos 
    ```
    python manage.py trigger_historical_db_update
    ```

4. Add all calculated parameters to the fetched OHLCV data. 
    
    // TODO

    4.1 Check consistency of the database. Sometimes entries get lost or duplicate entries find their way into the database.
    After having calculated all parameters, it is therefore necessary to run 

    ```
    python manage.py clean_historical_db_duplicate_missing
    ```

    This will remove duplicate entries by timestamp and detect missing entries and fill them with previous data. 
    As usually only one entry is missing, this is hurting data quality only little. If you find a better way to produce good data quality, good for you :)

5. Fine tune training config in ml_models/training_configs. 

    Training can be split in as many phases as desired. At each phase the parameters will be loaded. If not present, some default values are used instead. Percentage defines the total number of steps, while reward_function_index allows usage of different reward functions in each phase. See the code for implemented reward functions.

6. Start training by running the following command:

    ```
    python manage.py start_training --config coinbase_api/ml_models/training_configs/training_config_name.json
    ```

    See the training progress with hopefully meaningful output. Tensorboard integration still pending. 

    Open Trainings will be stored in ml_models/active_trainings/training_config_file_{idx}. Here a copy of the training config as well as progress of the config is saved. This allows for continuation of trainings that were aborted. 

    The RL Model is stored at set intervals in the training_config_name_{idx}_checkpoint folder.

