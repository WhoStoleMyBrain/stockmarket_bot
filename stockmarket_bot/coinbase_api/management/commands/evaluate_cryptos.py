# your_app/management/commands/evaluate_cryptos.py
import requests
import numpy as np
from datetime import datetime, timedelta
from django.core.management.base import BaseCommand
from coinbase_api.enums import Database
from coinbase_api.models.models import AbstractOHLCV  # Assuming CryptoOHLCV is the model for historical data
from coinbase_api.constants import crypto_models

class Command(BaseCommand):
    help = 'Evaluate cryptocurrencies based on defined metrics'
    
    full_name_to_short_name = {
        "bitcoin": "BTC",
        "ethereum": "ETH",
        "solana": "SOL",
        "ondo": "ONDO",
        "shiba-inu": "SHIB",
        "dogecoin": "DOGE",
        "ripple": "XRP",
        "bonk": "BONK",
        "jasmycoin": "JASMY",
        "render-token": "RNDR",
        "fetch-ai": "FET",
        "chainlink": "LINK",
        "litecoin": "LTC",
        "avalanche-2": "AVAX",
        "injective-protocol": "INJ",
        "optimism": "OP",
        "stacks": "STX",
        "hedera-hashgraph": "HBAR",
        "near": "NEAR",
        "uniswap": "UNI",
        "lido-dao": "LDO",
        "cardano": "ADA",
        "bitcoin-cash": "BCH",
        "sui": "SUI",
        "polkadot": "DOT",
        "stellar": "XLM",
        "polygon": "MATIC",
        "tia": "TIA",
        "arbitrum": "ARB",
        "tether": "USDT",
        "filecoin": "FIL",
        "the-graph": "GRT",
        "internet-computer": "ICP",
        "ethereum-name-service": "ENS",
        "velo": "VELO",
        "jto": "JTO",
        "aptos": "APT",
        "convex-finance": "CVX",
        "sei": "SEI",
        "tellor": "TRB",
        "highstreet": "HIGH",
        "curve-dao-token": "CRV",
        "ethereum-classic": "ETC",
        "aero": "AERO",
        "oasis-network": "ROSE",
        "aioz-network": "AIOZ",
        "immutable-x": "IMX",
        "truefi": "TRU",
        "sushiswap": "SUSHI",
        "aave": "AAVE",
        "maker": "MKR",
        "quant-network": "QNT",
        "cosmos": "ATOM",
        "ocean-protocol": "OCEAN",
        "big-time": "BIGTIME",
        "vechain": "VET",
        "echelon-prime": "PRIME",
        "algorand": "ALGO",
        "karura": "KARRAT",
        "superfarm": "SUPER",
        "pirate-chain": "PIRATE",
        "livepeer": "LPT",
        "ach": "ACH",
        "render-token": "RENDER",
        "akash-network": "AKT",
        "apecoin": "APE",
        "0x": "ZRX",
        "amp": "AMP",
        "arcblock": "ABT",
        "drift": "DRIFT",
        "tezos": "XTZ",
        "kyber-network": "KNC",
        "loopring": "LRC",
        "skale": "SKL",
        "strike": "STRK",
        "synthetix-network-token": "SNX",
        "rocket-pool": "RPL",
        "api3": "API3",
        "orion-protocol": "ORN",
        "tensor": "TNSR",
        "zetachain": "ZETA",
        "the-sandbox": "SAND",
        "crypto-com-coin": "CRO",
        "blur": "BLUR",
        "pingu": "PNG",
        "eos": "EOS",
        "yearn-finance": "YFI",
        "lcx": "LCX",
        "flare": "FLR",
        "compound": "COMP",
        "auction": "AUCTION",
        "safe": "SAFE",
        "1inch": "1INCH",
        "mask-network": "MASK",
        "mango-markets": "MPL",
        "decentraland": "MANA",
        "spell-token": "SPELL",
        "gnosis": "GNO",
        "iotex": "IOTX",
        "propy": "PRO",
        "zcash": "ZEC",
        "coinbase-wrapped-staked-eth": "CBETH",
        "axelar": "AXL",
        "helium": "HNT",
        "metisdao": "METIS",
        "ankr": "ANKR",
        "orca": "ORCA",
        "coti": "COTI",
        "wrapped-bitcoin": "WBTC",
        "chiliz": "CHZ",
        "mina-protocol": "MINA",
        "dai": "DAI",
        "elrond-erd-2": "EGLD",
        "omni": "OMNI",
        "mobilecoin": "MOBILE",
        "goldfinch": "GFI",
        "qi-dao": "QI",
        "biconomy": "BICO",
        "swftcoin": "SWFTC",
        "aleph-im": "ALEPH",
        "origintrail": "TRAC",
        "axie-infinity": "AXS",
        "hashflow": "HFT",
        "cartesi": "CTSI",
        "arkham": "ARKM",
        "vulcan-forged": "PYR",
        "origin-protocol": "OGN",
        "bonfida": "FIDA",
        "golem": "GLM",
        "harvest-finance": "FARM",
        "neon": "NEON",
        "assemble-protocol": "ASM",
        "synapse": "SYN",
        "suku": "SUKU",
        "chronobank": "XCN",
        "illuvium": "ILV",
        "rlc": "RLC",
        "badger-dao": "BADGER",
        "vara": "VARA",
        "steem-dollars": "GMT",
        "parsiq": "PRQ",
        "uma": "UMA",
        "aergo": "AERGO",
        "ronin": "RONIN",
        "gala-games": "GST",
        "ribbon-finance": "RBN",
        "kusama": "KSM",
        "kava": "KAVA",
        "storj": "STORJ",
        "audius": "AUDIO",
        "dash": "DASH",
        "orchid-protocol": "OXT",
        "bluzelle": "BLZ",
        "celo": "CGLD",
        "magic": "MAGIC",
        "thevirtunetoken": "TVK",
        "request-network": "REQ",
        "osmosis": "OSMO",
        "flow": "FLOW",
        "power-ledger": "POWR",
        "honeyswap": "HONEY",
        "nkn": "NKN",
        "aurora": "AURORA",
        "fortress": "FORT",
        "gitcoin": "GTC",
        "clover": "CLV",
        "math": "MATH",
        "basic-attention-token": "BAT",
        "arpa-chain": "ARPA",
        "district0x": "DNT",
        "zencash": "ZEN",
        "shapeshift-fox-token": "FOX",
        "nct": "NCT",
        "aventus": "AVT",
        "defi-yield-protocol": "DYP",
        "deso": "DESO",
        "my-neighbor-alice": "ALICE",
        "xyo": "XYO",
        "measurable-data-token": "MDT",
        "numeraire": "NMR",
        "coin98": "C98",
        "boba-network": "BOBA",
        "aavegotchi": "GHST",
        "pond": "POND",
        "sperax": "SPA",
        "ampleforth-governance-token": "FORTH",
        "perpetual-protocol": "PERP",
        "wrapped-ampleforth": "WAMPL",
        "civic": "CVC",
        "index-coop": "INDEX",
        "alchemy-pay": "ALCX",
        "threshold": "T",
        "vechainthor": "VTHO",
        "band-protocol": "BAND",
        "seam": "SEAM",
        "project-galaxy": "GAL",
        "shadow-token": "SHDW",
        "braintrust": "BTRST",
        "msol": "MSOL",
        "shping": "SHPING",
        "pundi-x": "PUNDIX",
        "circuits-of-value": "COVAL",
        "superrare": "RARE",
        "hopr": "HOPR",
        "dia-data": "DIA",
        "liquity": "LQTY",
        "00-token": "00",
        "celer-network": "CELR",
        "rarible": "RARI",
        "dimo": "DIMO",
        "radicle": "RAD",
        "gods-unchained": "GODS",
        "melon": "MLN",
        "bancor": "BNT",
        "wrapped-cfg": "WCFG",
        "media": "MEDIA",
        "airswap": "AST",
        "ethernity-chain": "ERN",
        "wax": "WAXL",
        "balancer": "BAL",
        "adventure-gold": "AGLD",
        "inverse-finance": "INV",
        "marinade": "MNDE",
        "crux": "CTX",
        "access": "ACS",
        "pluton": "PLU",
        "idex": "IDEX",
        "kryll": "KRL",
        "stafi": "FIS",
        "muse": "MUSE",
        "paypal-usd": "PYUSD",
        "voxels": "VOXEL",
        "chrono-tech": "TIME",
        "function-x": "FX",
        "bitdao": "BIT",
        "litentry": "LIT",
        "dextools": "DEXT",
        "loko": "LOKA",
        "polkastarter": "POLS",
        "pax-gold": "PAX",
        "minedar": "DAR",
        "elastos": "ELA",
        "gemini-dollar": "GUSD",
        "gyen": "GYEN",
        "rai-finance": "RAI",
        "liquid-staked-eth": "LSETH"
        # Add more mappings as needed
    }

    def fetch_crypto_data(self, crypto_full_names):
        url = "https://api.coingecko.com/api/v3/coins/markets"
        params = {
            "vs_currency": "usd",
            "ids": ",".join(crypto_full_names),
            "order": "market_cap_desc",
            "per_page": len(crypto_full_names),
            "page": 1,
            "sparkline": "false"
        }
        response = requests.get(url, params=params)
        return response.json()

    def calculate_volatility(self, symbol):
        end_date = datetime(year=2024, month=6, day=20)
        # end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        crypto_list = [crypto for crypto in crypto_models if crypto.symbol == symbol]
        if len(crypto_list) == 0:
            self.stdout.write(self.style.WARNING(f"Crypto {symbol} was not found in crypto list"))
            return 0
        crypto = crypto_list[0]
        historical_data = crypto.objects.using(Database.HISTORICAL.value).filter(
            timestamp__range=[start_date, end_date]
        ).order_by('timestamp').values_list('close', flat=True)
        
        if len(historical_data) < 2:
            return 0  # Not enough data to calculate volatility

        log_returns = np.diff(np.log(historical_data))
        volatility = np.std(log_returns) * np.sqrt(len(log_returns))
        return volatility

    def evaluate_cryptos(self, cryptos):
        selected_cryptos = []
        for crypto in cryptos:
            # self.stdout.write(self.style.WARNING(crypto))
            market_cap = crypto.get("market_cap", 0)
            volume_24h = crypto.get("total_volume", 0)
            volatility = self.calculate_volatility(self.full_name_to_short_name[crypto['id']])
            total_volume_close_price = volume_24h * crypto.get("current_price", 0)
            # self.stdout.write(self.style.WARNING(f"Found data for crypto: {crypto['id']}. market cap: {market_cap}. volume 24h: {volume_24h}. volatility: {volatility}. total volume close price: {total_volume_close_price}"))
            
            if (market_cap >= 100e6 and volume_24h >= 10e6 and 
                volatility >= 0.05 and total_volume_close_price >= 50e6):
                selected_cryptos.append(crypto)
            else: 
                self.output_rejection_reason(crypto, market_cap, volume_24h, volatility, total_volume_close_price)
        
        return selected_cryptos
    
    def output_rejection_reason(self, crypto, market_cap, volume_24h, volatility, total_volume_close_price):
        reasons = []
        if market_cap < 100e6:
            reasons.append(f"market_cap: {market_cap:.2f} < 100e6")
        if volume_24h < 10e6:
            reasons.append(f"volume_24h: {volume_24h:.2f} < 10e6")
        if volatility < 0.05:
            reasons.append(f"volatility: {volatility:.4f} < 0.05")
        if total_volume_close_price < 50e6:
            reasons.append(f"total_volume_close_price: {total_volume_close_price:.2f} < 50e6")
        
        rejection_reasons = "; ".join(reasons)
        self.stdout.write(self.style.ERROR(f"Crypto {crypto['id']} rejected due to: {rejection_reasons}"))

    def handle(self, *args, **options):
        # crypto_symbols = ["bitcoin", "ethereum", "solana", "shiba-inu", "dogecoin", "ripple", "ondo", "bonk"]
        crypto_symbols = self.full_name_to_short_name.keys()
        crypto_data = self.fetch_crypto_data(crypto_symbols)
        selected_cryptos = self.evaluate_cryptos(crypto_data)
        
        self.stdout.write(self.style.SUCCESS(f"Selected Cryptos: {[crypto['id'] for crypto in selected_cryptos]}"))
        self.stdout.write(self.style.SUCCESS(f"Selected Cryptos: {[self.full_name_to_short_name[crypto['id']] for crypto in selected_cryptos]}"))
