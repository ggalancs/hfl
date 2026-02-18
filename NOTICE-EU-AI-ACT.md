# EU AI Act Notice

This document provides information about hfl in relation to the European Union Artificial Intelligence Act (Regulation 2024/1689).

## What is hfl?

hfl is a tool for downloading and running AI models locally. It:

- Downloads open-weight AI models from HuggingFace Hub
- Converts model formats for hardware compatibility
- Provides local inference (no cloud processing)
- Exposes APIs for integration with applications

hfl does **NOT**:
- Train or fine-tune AI models
- Create or modify AI model behavior
- Host AI models for third-party access
- Process data on external servers

## hfl's Role Under EU AI Act

hfl is classified as an **inference tool**, not as:
- An AI system provider (Article 3(3))
- A GPAI model provider (Article 3(63))
- A deployer in the primary sense (Article 3(4))

hfl acts as a local execution environment similar to a Python runtime or container platform.

## User Responsibilities

Users who deploy AI models using hfl within the European Union may have obligations under the EU AI Act, including:

### For All AI Systems

- **Transparency** (Article 50): Inform end-users when they interact with AI
- **Human oversight**: Implement appropriate human review for automated decisions
- **Record keeping**: Maintain logs of AI system usage where applicable

### For High-Risk AI Systems

If you deploy models in high-risk domains (Annex III), additional obligations apply:

- Risk management system (Article 9)
- Data governance (Article 10)
- Technical documentation (Article 11)
- Record-keeping and logging (Article 12)
- Transparency and user information (Article 13)
- Human oversight measures (Article 14)
- Accuracy, robustness, cybersecurity (Article 15)

### For GPAI Models

Models trained with >10²³ FLOPs may be classified as General-Purpose AI (GPAI) models with additional requirements:

- Technical documentation
- Information for downstream providers
- Copyright compliance policy
- Training data summary

## Model Provider Compliance

Before deploying models in the EU, verify that model providers have met their obligations:

1. **Check model documentation** on HuggingFace for EU AI Act compliance statements
2. **Review the model card** for intended uses and limitations
3. **Verify license compatibility** with your deployment context
4. **Document your risk assessment** for the specific use case

## hfl Compliance Features

hfl includes features to support EU AI Act compliance:

| Feature | Purpose |
|---------|---------|
| License verification | Displays model license before download |
| Provenance logging | Records model origin and conversions |
| Disclaimer headers | Indicates AI-generated content in API responses |
| Local processing | No data leaves your infrastructure |

## Disclaimer

This notice is for informational purposes only and does not constitute legal advice. The EU AI Act is complex and evolving. Consult qualified legal counsel for compliance guidance specific to your situation.

## Resources

- [EU AI Act Full Text](https://eur-lex.europa.eu/eli/reg/2024/1689/oj)
- [AI Office Guidelines](https://digital-strategy.ec.europa.eu/en/policies/ai-office)
- [HuggingFace Model Cards](https://huggingface.co/docs/hub/model-cards)

---

*Last updated: February 2026*
