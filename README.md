Replication code for "The Stablecoin Margin Convenience Yield and the Equilibrium Pricing of Crypto Perpetual Futures."

## Structure

```
code/
├── settings.py           run configuration
├── econometrics.py       shared estimators
├── plot_style.py         matplotlib style
├── render_paper.py       generates all paper tables and figures
├── 01_data/              raw-data fetchers and panel builder
└── 02_experiments/       31 numbered experiment scripts

data_raw/
├── bitfinex/   USDT/USDC US-dollar spot
├── binance/    8 cryptoassets: perpetuals, spot, funding
├── bybit/      8 cryptoassets: funding
├── bitstamp/   DAI/USD spot
└── aave/       Aave V3 stablecoin lending APY
```

## Contact

monghwanseo@yonsei.ac.kr
