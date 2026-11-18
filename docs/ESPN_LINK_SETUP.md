# Setting up "auto-refresh" for private ESPN leagues

This is the one dashboard step Claude can't do from here — everything else (the Worker code,
the client UI) is already built and tested. This only needs doing once, ever, for the whole site.

## What this turns on

Once this is set up, a user can run "Set up auto-refresh" (paste their `espn_s2`/`SWID` cookies,
copied once from their browser's DevTools), and from then on the **Refresh from ESPN** button
works by itself — from their phone, or right inside the Home Screen app — with no browser trip,
until ESPN eventually invalidates their cookie (weeks/months later).

Until this is set up, the feature just says "Auto-refresh isn't set up on this server yet" and
nothing else about the site changes. Every other existing import method (public league ID, the
phone bookmarklet, pasting the Rosters page) keeps working exactly as before regardless.

## Step 1 — create a KV namespace

1. Go to the [Cloudflare dashboard](https://dash.cloudflare.com) → **Workers & Pages** → **KV** (in
   the left sidebar, under Storage & Databases).
2. Click **Create a namespace**. Name it something like `fantasymania-espn-links`. Create it.

## Step 2 — bind it to the Pages project

1. Go to **Workers & Pages** → your Pages project (fantasy-analyzer / fantasymania) → **Settings**
   → **Functions**.
2. Scroll to **KV namespace bindings** → **Add binding**.
3. Variable name: `ESPN_KV` (must match exactly — this is the name the code looks for).
4. KV namespace: pick the one you just created.
5. Save. Do this for both the **Production** and **Preview** environments if it asks.

## Step 3 — generate and set the encryption key

This key encrypts the stored cookies at rest. Generate it on your own computer (not in this chat,
so it never appears anywhere else) — on Mac/Linux:

```
openssl rand -base64 32
```

On Windows (PowerShell):

```
[Convert]::ToBase64String((1..32 | ForEach-Object { Get-Random -Maximum 256 }))
```

Then:

1. Same **Settings** → **Functions** (or **Environment variables**) page as above.
2. Add an environment variable: name `ESPN_ENC_KEY`, value = the string you just generated.
3. **Encrypt** it (there's usually a toggle/button for this — Cloudflare calls it a "secret").
4. Save, for both Production and Preview.

Write that key down somewhere safe (a password manager) outside of Cloudflare too — if you ever
lose it, every stored link breaks at once (everyone just sees "expired" and has to re-link, no
worse than a normal cookie expiry, but still — no reason to lose it for nothing).

## Step 4 — redeploy

The next deploy of `main` picks up both settings automatically — no code change needed beyond
what's already on the `espn-import` branch. Once that's merged and deployed, "Set up auto-refresh"
will work.

## Checking it worked

After setup, "Set up auto-refresh" on a real private league should succeed and hand back a token
instead of saying "not configured" (if it still says that, double check the binding name is
exactly `ESPN_KV` and the variable name is exactly `ESPN_ENC_KEY`, both on the Production
environment).

## What's actually stored

Per linked league: `espn_s2`, `SWID`, the league ID, and the season year — encrypted (AES-GCM)
before it's written to KV, keyed by a random token the browser holds (not by any Fantasy Mania
account, since the site doesn't have accounts). Anyone holding that token can read or delete that
one entry; nothing else is derivable from it. There's a "Turn off auto-refresh" button in the app
that deletes the entry.
