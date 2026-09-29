# AI-Based Prediction of Disruptions in Inter-City Railway Transport

**Qing He**  
Mid Sweden University

## Motivation and project objective

**How likely is a journey to suffer a serious disruption?**

| Perspective / objective | Description |
| --- | --- |
| Passenger perspective | Estimate risk to consider an earlier train, another route or a change of plans. |
| Project objective | Predict final arrival delay and disruption severity from historical railway data and weather. |

Research focus: accuracy, the added value of weather, and how risk updates during travel.

## Example case

A simple example is:

A passenger plans to take an ICE from Frankfurt to Berlin tomorrow afternoon. The weather forecast indicates unusually high temperature. Based on historical data for similar trains, routes, times and weather conditions, can we estimate whether the journey is likely to operate normally, experience a moderate delay, suffer a major delay, or be cancelled?

## Prediction settings and scope

| Setting | Information available | Output |
| --- | --- | --- |
| Before departure | Schedule, past reliability, weather available at the prediction time | Final delay and severity probability |
| During the journey | Add current delay, earlier stops and position along the route | Updated final delay and severity probability |

**Core: one direct ICE/IC journey on one or two selected origin–destination pairs**

Extension: one connection, including missed-connection risk

### Journey and selected origin–destination pair

A **journey** is one run of one ICE or IC service on a particular date, considered between a selected origin station and destination station, without changing trains. The train must stop at the origin before reaching the destination. It may stop at other stations between them, begin before the selected origin, or continue beyond the selected destination. Intermediate stops may differ between journeys. The same train number running on another date is a different journey.

For example, a particular ICE travelling from Berlin Hbf to München Hbf on 15 October is one journey, even if it stops at Leipzig, Nürnberg and Ingolstadt. A run of the same train number on the following day is another journey.

A **corridor** means the selected origin–destination pair and the train services connecting those stations. It does not mean every track section between consecutive cities. Berlin–München is one study corridor; Berlin–Leipzig, Leipzig–Nürnberg and Nürnberg–München do not need to be counted as three separate corridors. To avoid ambiguity, use **selected origin–destination pair** from here on.

## A working definition of disruption severity

| Class | Proposed starting definition |
| --- | --- |
| Normal / minor | Final arrival delay below 20 minutes |
| Moderate | 20 to less than 60 minutes |
| Severe | 60 minutes or more |
| Cancelled | Journey cannot operate as planned |

**Initial binary target: severe delay or cancellation**

Also predict delay in minutes for completed journeys. Revisit thresholds after data exploration.

## Railway and weather datasets

| Dataset / task | Description |
| --- | --- |
| Bahn-Vorhersage | German train records since September 2021, in daily Parquet files. Reconstruct journeys from stop events. |
| DWD Climate Data Center | Historical hourly station observations: temperature, precipitation and wind. |
| Initial data task | Use a small ICE/IC subset. Join train location and time to a suitable weather station and hour. |

Data checks: access, missing records, final versus forecast updates, cancellation meaning and time zones.

## Feature groups and modelling roadmap

### Feature groups

| Feature group | Inputs |
| --- | --- |
| Service and calendar | Route, train type, time, season, duration |
| Past reliability | Earlier delay and cancellation rates |
| Weather | Temperature, rain, wind, recent extremes |
| During travel | Current delay, delay change, progress |

### Model progression

| Stage | Models |
| --- | --- |
| 0 | Historical mean / class frequency; current-delay persistence during travel |
| 1 | Linear / logistic regression |
| 2 | Random Forest; XGBoost or LightGBM |
| 3 | Optional LSTM / Transformer / GNN. Only after a stable core pipeline |

Feature ablations and model explanations help identify useful information.

## Does weather improve prediction?

**Railway features alone compared with railway features plus weather**

| Aspect | Description |
| --- | --- |
| Fair comparison | Same journeys, model and time split. Measure the change in severe-disruption detection and delay error. |
| Interpretation | A small or zero gain is a useful finding. Examine where weather helps and where it does not. |

Before departure: use past observations or forecasts issued before the prediction cutoff.

## Minimum successful project and extensions

### Minimum successful project

- A reproducible journey dataset with a clear severity target
- Baselines and standard ML models with temporal evaluation
- A fair railway-only versus railway-plus-weather experiment
- An explanation of model errors, useful features and limitations

### Optional extensions, after the core works

Dynamic risk updates, one connection, an advanced neural model, extreme-weather analysis or a simple decision-support prototype

Kickoff decisions: prediction cutoff, candidate origin–destination pairs, target and first data sample.

## Tasks for the first two weeks

- Literature review:
  - prediction targets;
  - common inputs;
  - common models;
  - evaluation methods.
- Inspect the dataset you would like to use:

  For example: the Bahn-Vorhersage dataset documentation, understand the main columns, and download only a very small sample.

  - Compare two or three possible origin–destination pairs and count the available ICE/IC journeys for each. Count each dated ICE/IC run when it stops at the origin before reaching the destination without a train change; intermediate stops may differ.
  - Select one origin–destination pair and analyse one direction first.
  - For the selected pair, examine an example journey and summarise arrival delays, cancellations and missing data.
- Investigate DWD weather data and demonstrate that they can retrieve temperature for at least one station/date and align it with a train event.
- Propose a project time plan
