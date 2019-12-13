# Instructions

Build using:

```bash
swift build
```

# Configurations

For `medical-causes` and `medical-treats` we are using LIA
configured with the following options:

- Predictor Embedding Size: `32`
- Instance Hidden Unit Counts: `[32, 32, 32, 32]`
- Predictor Hidden Unit Counts: `[]`
- Confusion Latent Size: `1`
- Gamma: `0.0`
- Entropy Weight: `0.0`
- Use Soft Predictions: `true`
- Learning Rate: `1e-4`
- Learning Rate Decay Factor: `1.0`
- Batch Size: `512`
- M Step Count: `1000`
- EM Step Count: `2`
- Marginal Step Count: `1000`
