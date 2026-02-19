# License FAQ — HRUL v1.0

Frequently asked questions about the hfl Responsible Use License (HRUL) v1.0.

## Can I use hfl for commercial purposes?

**YES.** You can use hfl in commercial products, services, and internal tools without restriction.

## Can I sell a product based on hfl?

**YES.** You can build, brand, and sell derivative products. You can charge money for them. The only requirement is that your product must keep the five Compliance Modules active (license checking, provenance tracking, AI disclaimers, privacy protection, gating respect).

## Can I modify the code?

**YES.** You can modify, extend, refactor, or completely rewrite any part of hfl, including the Compliance Modules themselves — as long as the modified versions still perform the same protective functions.

## Can I remove the license checker?

**NO.** The License Verification Module must remain active in any publicly distributed derivative work. You can rewrite it, improve it, replace the classification database, or change the UI — but it must still verify and display model licenses before download.

## Can I remove the AI disclaimer?

**NO.** The AI Output Disclaimer Module must remain active. You can change the wording to be more specific or comprehensive, but it must still convey that content is AI-generated and that the user assumes responsibility.

## Can I disable compliance features with a config flag?

**NO global disable.** You cannot provide a setting that turns off all compliance features at once. However, you CAN provide per-operation override flags (like `--skip-license`) for individual commands, as long as the default behavior is full compliance and a warning is shown when the flag is used. This exception exists for CI/CD and automation workflows.

## Can I use hfl internally without the Compliance Modules?

**YES.** The Compliance Module requirements only apply to publicly distributed derivative works. Internal use within your organization is fully exempt (Section 4.3). That said, keeping the modules active is recommended for your own legal protection regarding model licenses.

## Can I fork hfl and use a different name?

**YES.** You can rebrand, rename, and change the visual design. You must include attribution ("Based on hfl — Licensed under HRUL v1.0") in a reasonably prominent location and keep the Compliance Modules active.

## Can I offer hfl as a hosted service (SaaS)?

**YES**, but it counts as Distribution. The Compliance Modules must be active in the hosted version, and the HRUL terms apply.

## Is HRUL an open-source license?

**No, not in the formal OSI/FSF sense.** The HRUL imposes behavioral requirements on derivative works (maintaining compliance features), which goes beyond what the Open Source Definition allows. hfl is best described as **source-available with responsible use requirements**. You have full access to the source code and broad rights to use, modify, and distribute — with the condition that safety features survive.

## What licenses is HRUL based on?

The HRUL draws from several established licenses:

- **Apache License 2.0** — Overall structure, grant of rights, disclaimer, limitation of liability
- **GPL-3.0** — Copyleft concept (downstream preservation of obligations), cure period mechanism
- **RAIL (Responsible AI License)** — Behavioral requirements that flow to derivative works
- **MPL 2.0** — Module-level scope (obligations apply to specific components, not the entire codebase)
- **Jacobsen v. Katzer precedent** — License conditions enforced as copyright conditions, not merely contractual covenants

## Why not just use Apache 2.0 or MIT?

Permissive licenses cannot require derivative works to maintain specific functionality. A fork under MIT could legally strip all license verification, remove AI disclaimers, bypass model gating, and persist tokens to disk. The entire compliance stack — which exists to protect users and model authors — could be deleted with no legal consequence.

## Why not use GPL?

GPL requires sharing all source code of derivative works. This is both too much (hfl doesn't need to force source disclosure) and too little (GPL doesn't mandate that specific safety features remain functional — a derivative could include the code but never call it).

## What happens if I violate the license?

You have a **30-day cure period** to fix the violation. If you restore the Compliance Modules within 30 days, your rights continue uninterrupted. If not, your distribution rights terminate. A second violation within 12 months of reinstatement results in permanent termination.

## Does the HRUL affect the licenses of hfl's dependencies?

**NO.** The HRUL explicitly separates its scope from dependency licenses (Section 9.5). All dependencies (typer, FastAPI, torch, etc.) retain their own MIT/Apache 2.0/BSD licenses. If you extract a dependency from hfl and use it independently, only that dependency's own license applies.

## Where can I read the full license?

See [LICENSE](LICENSE) in this repository. The license includes a Compliance Verification Checklist (Appendix A) and a Rationale section (Appendix B) explaining the legal context.
