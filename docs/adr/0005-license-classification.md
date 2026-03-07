# ADR-0005: License Classification System

## Status

Accepted

## Context

HuggingFace Hub hosts models with various licenses, from permissive open-source to restrictive commercial licenses. Users downloading models through HFL need to:
- Understand the legal implications
- Make informed decisions
- Comply with license requirements
- Be warned about high-risk licenses

We need a classification system that is:
- Easy to understand
- Legally accurate (not legal advice)
- Actionable for users
- Maintainable

## Decision

We implement a 5-level license risk classification system:

| Level | Name | Description | Example Licenses |
|-------|------|-------------|------------------|
| 1 | PERMISSIVE | Free use, minimal restrictions | MIT, Apache-2.0, CC0 |
| 2 | CONDITIONAL | Some conditions (attribution, share-alike) | CC-BY, GPL, LGPL |
| 3 | NON_COMMERCIAL | Non-commercial use only | CC-BY-NC, Llama 2 |
| 4 | RESTRICTED | Significant restrictions, requires review | Custom licenses |
| 5 | UNKNOWN | License not found or unrecognized | N/A |

**User experience:**
- Level 1-2: Download proceeds with brief notice
- Level 3: Warning about commercial restrictions
- Level 4-5: Explicit user confirmation required

**Implementation:**
```python
class LicenseRisk(Enum):
    PERMISSIVE = 1
    CONDITIONAL = 2
    NON_COMMERCIAL = 3
    RESTRICTED = 4
    UNKNOWN = 5
```

## Consequences

### Positive

- Clear risk levels for users
- Consistent handling across all models
- Automatic classification for common licenses
- Explicit consent for risky licenses

### Negative

- Not legal advice, users still responsible
- May miss license nuances
- Requires maintenance as new licenses appear
- Some edge cases hard to classify

### Neutral

- Classification stored in model manifest
- Provenance tracking for compliance
- User can override with explicit consent

## Alternatives Considered

### Option A: Binary Allow/Block

Simple allow or block list.

**Pros:**
- Very simple
- Clear decisions

**Cons:**
- Too coarse
- Blocks legitimate use cases
- User frustration

### Option B: Full Legal Parser

Parse and analyze license text programmatically.

**Pros:**
- Accurate analysis
- Handles any license

**Cons:**
- Extremely complex
- Error prone
- Maintenance nightmare

### Option C: No Classification

Let users handle licenses themselves.

**Pros:**
- No liability for HFL
- Simpler implementation

**Cons:**
- Poor user experience
- Users may unknowingly violate licenses
- Reputation risk

## References

- SPDX license list
- HuggingFace license metadata
- Open source license compatibility
