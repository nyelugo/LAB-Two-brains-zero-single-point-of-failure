# Lab summary

Author: Nnanyelugo Ahukannah

The hard part of this lab was not calling two APIs, it was making the second one
useful when the first dies. I put both providers behind a single interface so
either can summarize or classify, then had a `ProviderPool` try a preferred
provider and fall back to the other, recording every substitution so a reviewer
can see failover rather than take it on trust. Two smaller problems shaped the
design: NewsAPI's free tier truncates article text to about 200 characters, so I
added a `content -> description -> title` fallback and a minimum-length check to
avoid paying to summarize a bare headline; and the lab page shipped none of the
code it referenced, so every module was written from the spec rather than filled
in. What I learned is that "fallback" is only real if you can prove it — the
tests that mattered were the ones that kill a provider and assert the other one
picked up, and the `--simulate-outage` flag that does the same thing against the
live APIs. Given more time I would cache by article URL to stop re-processing
unchanged stories, and batch the sentiment calls, since input tokens dominate
cost at roughly $0.00011 per article.
