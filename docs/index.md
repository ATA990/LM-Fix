# LM-Fix

**LM-Fix** is a lightweight framework for **bit-flip detection** and **rapid recovery** in LLMs.

## TL;DR
- Detects silent bit flips using golden and layer-wise hashes.
- Localizes faults down to parameters.
- Recovers by cache clearing, layer search, and integer-view weight repair.

## Citation
Please cite the software and the paper (ICCAD 2025). See `CITATION.cff`.

## JSON-LD (for better indexing)

```html
<script type="application/ld+json">
{
  "@context": "https://schema.org",
  "@type": ["SoftwareApplication", "ScholarlyArticle"],
  "name": "LM-Fix",
  "headline": "Lightweight Bit-Flip Detection and Rapid Recovery Framework for Language Models",
  "url": "https://YOUR_USERNAME.github.io/LM-Fix/",
  "codeRepository": "https://github.com/ATA990/LM-Fix",
  "license": "https://opensource.org/licenses/MIT",
  "applicationCategory": "AI Security",
  "programmingLanguage": "Python",
  "creator": [{"@type": "Person", "name": "Add your name"}],
  "datePublished": "2025-10-22",
  "isBasedOn": {"@type": "CreativeWork", "name": "ICCAD 2025 paper"},
  "keywords": ["bit-flip", "fault injection", "reliability", "LLM", "PyTorch", "model recovery"]
}
</script>
```
