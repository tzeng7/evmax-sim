"""Name canonicalisation — the cross-source join key."""

from evmax.golf.names import canonical


class TestCanonical:
    def test_basic_lowercase(self):
        assert canonical("Matt Fitzpatrick") == "matt fitzpatrick"

    def test_last_first_order(self):
        assert canonical("Fitzpatrick, Matt") == "matt fitzpatrick"

    def test_strips_accents(self):
        assert canonical("Nicolás Echavarría") == "nicolas echavarria"
        assert canonical("Ludvig Åberg") == "ludvig aberg"

    def test_drops_suffix(self):
        assert canonical("Davis Thompson Jr.") == "davis thompson"
        assert canonical("Charl Schwartzel III") == "charl schwartzel"

    def test_removes_punctuation(self):
        assert canonical("J.J. Spaun") == "jj spaun"
        assert canonical("Byeong-Hun An") == "byeonghun an"

    def test_collapses_whitespace(self):
        assert canonical("  Tom   Kim  ") == "tom kim"

    def test_empty(self):
        assert canonical("") == ""

    def test_cross_source_match(self):
        # ESPN, PGA "Last, First", and a venue spelling all collapse to one key.
        assert (
            canonical("Tommy Fleetwood")
            == canonical("Fleetwood, Tommy")
            == canonical("Tommy  Fleetwood")
        )
