---
title: Volatility Radar
emoji: 📡
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
short_description: Market and sector volatility forecasts with their track record
---

# Volatility Radar

Forecasts how much the US market (SPY) and each of the eleven S&P sectors will move over the
next 10 trading days, with calibrated uncertainty bands, an out-of-sample track record, and a
news-channel overlay that scores headlines with a fine-tuned DistilBERT model.

This Space only **serves** the dashboard. A GitHub Actions job refreshes it every weekday
after the US close (new prices, model refit, news radar) and pushes the updated reports here.

Source, methodology and full results: https://github.com/toosh-legacy/quant-nlp-gen

Educational research project, **not investment advice**. It forecasts volatility (how much
prices move), never direction.
