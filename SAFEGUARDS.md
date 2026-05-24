# Safeguards & Trademark Policy

> **Not legal advice.** This document explains how the project protects its
> responsible-use Safeguards. Have a lawyer review it for your jurisdiction.

## The short version

- **The code is Apache-2.0.** You may use, modify, and redistribute hfl
  freely, including commercially. See [LICENSE](LICENSE).
- **The name "hfl" and the official builds are not.** Using the "hfl" name,
  logo, or releasing something *as* "hfl" is conditioned on keeping the
  Safeguards below active and functional. This is the project's trademark
  policy, which Apache-2.0 explicitly does **not** grant (Section 6).

This gives the code maximum adoption (a true open-source license) while
keeping a fork from stripping the user-protecting features *and still trading
on the hfl name and reputation*.

## The Safeguards

A distribution that calls itself "hfl" (or uses the marks / implies it is the
official project) must keep these active, with full compliance as the default:

1. **License Verification** — verify and surface a model's license before
   download.
2. **Provenance Tracking** — record where each model came from
   (repo, revision, hash).
3. **AI-Output Disclaimer** — convey that generated content is AI-produced and
   that the user assumes responsibility.
4. **Privacy Protection** — never persist auth tokens or secrets to disk/logs.
5. **Model Gating Respect** — do not bypass or automate around a model's
   gated-access requirements.

Per-operation override flags for automation (e.g. `--skip-license`) are fine,
provided the default is full compliance and a warning is shown.

## What this means for forks

| You want to… | Allowed under Apache-2.0? | May you call it "hfl"? |
|---|---|---|
| Use / embed / sell hfl, safeguards intact | Yes | Yes (attribution appreciated) |
| Modify the code, safeguards intact | Yes | Yes |
| Remove or disable the Safeguards | **Yes** (the code is Apache) | **No** — rename it and drop the hfl marks / any implied affiliation |

So the Safeguards cannot be quietly stripped *from something presented as
hfl*. A renamed fork is free to do as it wishes with the Apache-2.0 code — it
simply can't borrow the hfl name or reputation to do so.

## Trademark

"hfl" and the hfl logo are trademarks of the project author. Nominative,
truthful references ("compatible with hfl", "a fork of hfl") are always fine.
What requires keeping the Safeguards is presenting your distribution *as* hfl.

## History

hfl was previously released under the custom **HRUL v1.0** (source-available,
which enforced the Safeguards through copyright). It moved to **Apache-2.0 +
this policy** to be a real open-source project while still protecting users
and model authors. The historical text is kept at
[`LICENSE-HRUL-1.0.txt`](LICENSE-HRUL-1.0.txt) for reference.
