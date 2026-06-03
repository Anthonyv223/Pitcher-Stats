# What Makes an Ace? — MLB Pitching Analysis

Analyzing what separates the ace of an MLB pitching staff from the rest of
the rotation, using Statcast and FanGraphs data (2018–2024).

## Approach

**Ace definition:** one ace per team per season — among starters with ≥10
games started, the pitcher with the highest fWAR (tiebreaker: innings
pitched). This frames the question as "what separates a #1 starter from a
#2–5 on the same staff" rather than comparing to league-wide talent.

## Pipeline

1. `ace_labels.py` — pulls season-level FanGraphs stats and labels the ace
   of each team's staff.
2. `statcast_features.py` — pulls Statcast pitch-level data for each labeled
   pitcher and engineers stuff/command features (velocity, whiff%, CSW%,
   chase%, extension, repertoire, contact quality), merged into one analysis
   table.

## Setup
