UPDATE public.coinbase_api_ethereum
SET sma = 0 WHERE sma IS NULL;
UPDATE public.coinbase_api_ethereum
SET ema = 0 WHERE ema IS NULL;
UPDATE public.coinbase_api_ethereum
SET percentage_returns = 0 WHERE percentage_returns IS NULL;
UPDATE public.coinbase_api_ethereum
SET log_returns = 0 WHERE log_returns IS NULL;
UPDATE public.coinbase_api_ethereum
SET rsi = 0 WHERE rsi IS NULL;
UPDATE public.coinbase_api_ethereum
SET bollinger_low = 0 WHERE bollinger_low IS NULL;
UPDATE public.coinbase_api_ethereum
SET bollinger_high = 0 WHERE bollinger_high IS NULL;
UPDATE public.coinbase_api_ethereum
SET macd = 0 WHERE macd IS NULL;

UPDATE public.coinbase_api_polkadot
SET sma = 0 WHERE sma IS NULL;
UPDATE public.coinbase_api_polkadot
SET ema = 0 WHERE ema IS NULL;
UPDATE public.coinbase_api_polkadot
SET percentage_returns = 0 WHERE percentage_returns IS NULL;
UPDATE public.coinbase_api_polkadot
SET log_returns = 0 WHERE log_returns IS NULL;
UPDATE public.coinbase_api_polkadot
SET rsi = 0 WHERE rsi IS NULL;
UPDATE public.coinbase_api_polkadot
SET bollinger_low = 0 WHERE bollinger_low IS NULL;
UPDATE public.coinbase_api_polkadot
SET bollinger_high = 0 WHERE bollinger_high IS NULL;
UPDATE public.coinbase_api_polkadot
SET macd = 0 WHERE macd IS NULL;

UPDATE public.coinbase_api_bitcoin
SET sma = 0 WHERE sma IS NULL;
UPDATE public.coinbase_api_bitcoin
SET ema = 0 WHERE ema IS NULL;
UPDATE public.coinbase_api_bitcoin
SET percentage_returns = 0 WHERE percentage_returns IS NULL;
UPDATE public.coinbase_api_bitcoin
SET log_returns = 0 WHERE log_returns IS NULL;
UPDATE public.coinbase_api_bitcoin
SET rsi = 0 WHERE rsi IS NULL;
UPDATE public.coinbase_api_bitcoin
SET bollinger_low = 0 WHERE bollinger_low IS NULL;
UPDATE public.coinbase_api_bitcoin
SET bollinger_high = 0 WHERE bollinger_high IS NULL;
UPDATE public.coinbase_api_bitcoin
SET macd = 0 WHERE macd IS NULL;