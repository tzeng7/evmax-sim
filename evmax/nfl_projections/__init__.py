"""Standalone NFL projection engine (game scores now, player stat lines next).

A platform feature judged on ACCURACY, not on betting EV: for every game it
projects each team's points, and from them the margin (spread), total and win
probability. It is deliberately not part of the scan pipeline (like
``evmax.golf``) and reads no market price as a model input — closing lines are
used only to SCORE it. See docs/nfl-projections-scope.md for the design and
the accuracy bar (2020-25 Vegas close: margin MAE 9.76, total MAE 10.30).

Pipeline:

    nflverse play-by-play + schedules (local parquet cache)   evmax.nfl_projections.data
      -> one row per team-game (efficiency, pace, drives)     evmax.nfl_projections.team_games
      -> point-in-time opponent-adjusted ratings              evmax.nfl_projections.ratings
         (recency-weighted ridge: metric = mu + off + def + home)
      -> projected team points -> margin / total / P(win)     evmax.nfl_projections.game_model
      -> an upcoming week (starters, roof, wind fallbacks)     evmax.nfl_projections.live
      -> model pick vs the Vegas line + graded record          evmax.nfl_projections.picks

Walk-forward evaluation: scripts/backtest_nfl_game_projections.py (accuracy),
scripts/backtest_nfl_model_picks.py (picks vs the close / ESPN openers).
CLI: ``evmax project nfl [--season S --week W]``, ``evmax project nfl-record``.
"""
