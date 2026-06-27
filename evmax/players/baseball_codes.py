"""MLB player-name aliases for prop matching.

Most MLB names match cleanly between Kalshi titles and Pinnacle special
descriptions once accents are folded (handled in normalize_player_name), so
this map only needs entries for genuine spelling divergences — nicknames,
Anglicized forms, or punctuation the two sources disagree on. Keys are
lowercase, accent-folded; values are the canonical "first last" form.
"""

# Keep keys lowercase + ASCII (post-fold). Add entries only when a real
# Kalshi↔Pinnacle mismatch is observed in shadow logs.
BASEBALL_PLAYER_ALIASES: dict[str, str] = {
    # e.g. "shohei ohtani": "shohei ohtani",  # placeholder pattern
}
