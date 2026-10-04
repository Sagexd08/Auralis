# apps/web

The public Auralis site: two self-contained pages (`index.html` and `whitepaper.html`) with no build step
and no dependencies.

```bash
# serve locally
python3 -m http.server 8080 --directory apps/web
# then open http://localhost:8080
```

Deploys as-is to any static host — GitHub Pages, Vercel, Cloudflare Pages —
with `apps/web` as the output directory and no build command.

## Scope

The PRD describes a larger Next.js site (`/playground`, `/benchmarks`,
`/models`, `/docs`, an interactive architecture diagram and a live inference
demo). This is deliberately not that: it is one honest landing page for what
Auralis actually does today, with no build toolchain to maintain while the
dictation loop is still the thing being worked on.

**It quotes no accuracy numbers.** The benchmark harness exists but has not been
run against a real corpus, so there is nothing to publish. If you add numbers
here, add the command that produced them alongside.
