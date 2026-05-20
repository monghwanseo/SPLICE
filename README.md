Replication code for "The Stablecoin Margin Convenience Yield and the Equilibrium Pricing of Crypto Perpetual Futures."

## Structure

```
code/
├── settings.py           config
├── econometrics.py       estimators
├── plot_style.py         plot style
├── render_paper.py       tables and figures
├── 01_data/              data fetchers
└── 02_experiments/       experiment scripts

data_raw/
├── bitfinex/   USDT/USDC spot
├── binance/    perpetuals, spot, funding
├── bybit/      funding
├── bitstamp/   DAI spot
└── aave/       stablecoin lending APY
```

## Data and license note

The MIT License applies only to the source code. Raw exchange and lending data remain subject to the original data providers' policies.

## Contact

monghwanseo@yonsei.ac.kr
