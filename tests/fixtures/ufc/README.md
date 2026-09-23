# UFC fixtures

- `kxufcfight_short_title_markets.json` / `kxufcfight_short_title_events.json` —
  captured live **2026-09-22** from the public (unauthenticated) Kalshi
  `GET /markets?series_ticker=KXUFCFIGHT&status=open` and
  `GET /events?series_ticker=KXUFCFIGHT&status=open`, cut to 7 fights
  (14 markets) and trimmed to the fields the parser reads; values are real.
  This is the SHORT title format Kalshi switched to in 2026-08
  ("Mickey Gall wins" — no opponent). Picked for the edge cases: alias-needing
  spellings (Norma Dumont Viana, Alateng Heili, Wang Cong), a generational
  suffix (Raul Rosas Jr), a multi-word surname the event sub_title truncates
  (Rafael Dos Anjos → "Anjos"), a numbered-card title prefix ("332: ..."), and
  sibling markets listed home-first. Used by `TestUFCShortTitleFormat` in
  `tests/test_ufc_sector.py`.

The files below were captured live on 2026-07-11 (the old long title format).

- `kxufcfight_markets.json` — Kalshi `GET /markets?series_ticker=KXUFCFIGHT&status=open`
  (UFC 318 card: Saint-Denis vs Pimblett, McGregor vs Holloway). The
  unauthenticated snapshot nulls every price field, so realistic
  `*_dollars` prices were added by hand — the parser drops price-less rows.
- `espn_scoreboard_trimmed.json` — first two bouts of
  `site.api.espn.com/apis/site/v2/sports/mma/ufc/scoreboard?dates=20240413`
  (UFC 300), trimmed to the fields `parse_scoreboard` reads.
- `espn_status.json` — core status object for UFC 300's Figueiredo vs
  Garbrandt (`…/events/600041053/competitions/401630738/status`), carrying
  `result` = Submission (Rear Naked Choke).
- `espn_athlete_trimmed.json` — core athlete object for Deiveson Figueiredo
  (id 4189320), trimmed to bio fields.

Refresh: re-run the curl calls above and re-trim (see
`evmax/clients/ufc_espn.py` for the endpoints).
