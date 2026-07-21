# Submissions log

| Version | Date | Method | Params | P@10 | MIA score | Time(s) | Notes |
|---|---|---|---|---|---|---|---|
| TIMidi_V1 | 2026-07-20 19:29 | finetune | epochs=5, lr=1e-2, split=70/15/15 | 0.6045 | 0.988 | 9 | baseline; leaderboard 82.640 (prec .633 / mia .997 / time .925) |
| TIMidi_V2 | 2026-07-20 19:xx | ssd | alpha=5, lam=2, split=70/15/15 | 0.6401 | 0.988 | 3 | SSD tuned, eval on test split |
| TIMidi_V3 | 2026-07-20 19:xx | gradasc | ascent1/repair2 (default), split=70/15/15 | 0.5924 | 0.986 | 4 | gradient ascent, weaker than baseline |
| TIMidi_V4 | 2026-07-20 19:xx | ssd | alpha=5, lam=2, split=85/15 | 0.6358 | 0.992 | 2 | SSD tuned, more train data (85%) |
