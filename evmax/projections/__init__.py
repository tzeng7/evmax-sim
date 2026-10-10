"""Dashboard Projections tab: one entrypoint for every sector's projection model.

    data/projections.yaml           which sectors the tab shows, their engine, option defaults
      -> catalog.get_catalog()       validated sector catalog (catalog.py)
      -> engine.slate_options()      what a run can be configured with (OptionSpec)
      -> engine.run_slate()          every game of the next slate (+ player rows)
      -> engine.run_game()           the deeper model for one game
      -> engine.stored()             persisted runs, for engines that store them

Engines: ``nfl.NflProjectionEngine`` (evmax.nfl_projections: game model,
player model, joint simulation) and ``point.PointProjectionEngine``
(PointProjectionModel over the Pinnacle board: NBA possession sim, Poisson +
Elo for college basketball). The web routes live in ``evmax/web/app.py``
under ``/api/projections``. Projections are not an EV input.
"""
