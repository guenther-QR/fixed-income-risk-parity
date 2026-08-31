# LinkedIn post

---

Last summer, before starting my MFE at UCLA Anderson, I built a macro asset
allocation study. It reported a Sharpe ratio of 1.17.

I spent the last few months rebuilding it. I found nine accounting errors. The
risk-free rate was set to zero in every period. Credit was modeled as if bonds
never default, which overstated high yield by 3.4% a year. Equity returns left
out dividends. A misaligned line of code meant the portfolio I described in the
writeup was not the portfolio I had actually computed.

After fixing all of it, the Sharpe was 0.82.

Then I did the part I had skipped entirely: a real out of sample test.

I built a walk forward engine that charges transaction costs per asset and
charges interest on any borrowed money. I added 182 predictive signals from the
academic literature. I ran eight model families, including elastic net, random
forest, gradient boosting, and regime switching models. I ran a 1,296
specification grid over every regime design choice the literature says matters.
And I locked away ten years of data in code before fitting anything, with every
access logged.

About 1,600 specifications. None of them beat a simple 60/40 portfolio out of
sample.

The reason turned out to be measurable. My seven asset universe only contained
2.88 independent bets. Gold, stocks, credit and four points on one yield curve
are not seven different things. They are closer to three. With that little
diversification, the math caps your information ratio around 0.44 before costs.

The signals were fine. The portfolio was too small to express them.

But one result held up, and it is specific to bonds.

Bond returns get harder to forecast as they get riskier. The rank correlation
between out of sample R squared and duration is negative 0.96. Short dated bonds
are predictable because their return is mostly carry, which you know in advance.
Long dated bonds are unpredictable because their return is mostly yield changes,
which you do not. The same fact makes short bonds low risk.

So I built a bond only portfolio around it. Risk parity, which sizes positions so
each one contributes the same amount of risk, instead of the same amount of
money. I also tested Hierarchical Risk Parity, a method from Marcos Lopez de
Prado that groups similar bonds together before allocating, so four nearly
identical Treasuries do not get treated as four separate bets.

Over 1982 to 2015, against an equal weight portfolio of the same 11 bonds:

Hierarchical Risk Parity: Sharpe 0.93 vs 0.74
Risk parity: Sharpe 0.88 vs 0.74

Both beat a 50/50 two year and ten year barbell, and both were ahead of the
Vanguard Total Bond Market fund. The edge survives when I match portfolio
duration, so it is not just a bet on holding shorter bonds. Turnover is under 15%
a year, so trading costs do not eat it.

Out of sample, from 2016 to 2026, both stayed ahead of equal weight, but the
margins were small and not statistically significant. That decade was hard for
every bond portfolio.

Full writeup and code in the comments. The failures are in there too, with the
log of every model I ran.

Beating the market is hard. Figuring out exactly why it is hard turned out to be
the more useful project.

#quantitativefinance #fixedincome #assetallocation #MFE

---

## Shorter version, if you want it tighter

I rebuilt a study I wrote last year that claimed a Sharpe of 1.17.

It had nine accounting errors. No risk-free rate. Credit modeled without
defaults. Equity returns without dividends. Fixed, the Sharpe was 0.82.

Then I ran the test I had skipped. About 1,600 specifications, ten years of data
sealed before I fit anything, transaction costs and borrowing costs charged.
Nothing beat 60/40 out of sample.

The reason was measurable. My seven asset universe had 2.88 independent bets. The
math caps the information ratio near 0.44. The signals were fine. The portfolio
was too small.

One thing held up. Bond returns get harder to forecast as they get riskier. The
rank correlation between out of sample R squared and duration is negative 0.96.

So I built a bond only book on that idea, using risk parity and Lopez de Prado's
Hierarchical Risk Parity. Over 1982 to 2015 it returned a Sharpe of 0.93 against
0.74 for equal weight, and it beat a Treasury barbell and the Vanguard Total Bond
fund. The edge holds after matching duration, so it is not just a bet on shorter
bonds.

Code and full writeup below, failures included.

---

## Posting notes

- Pick one link as the main call to action. The GitHub repo is probably better
  than the report page for an interviewer audience.
- The negative 0.96 number is the most quotable thing here. If you cut for
  length, keep it.
- If someone asks why you led with the failures: the failures are what make the
  one positive result believable. Anyone can show you a backtest that worked.
