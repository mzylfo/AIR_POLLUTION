# VAE Two-Feature Generation

This document describes the new functionality that allows VAE models to receive two features as input reference and generate synthetic data.

## Overview

The requirement was to enable the VAE model to "ricevere in ingresso due feature per ref e generare i dati tramite modello VAE" (receive two features as input for reference and generate data through the VAE model).

## Solution

We have implemented a comprehensive solution consisting of:

1. **VAEGenerator Class**: Core functionality for generating data from trained VAE/CVAE models
2. **TwoFeatureVAEInterface**: Simplified interface specifically for two-feature input
3. **ModelPredictionEnhanced**: Enhanced prediction capabilities with two-feature support
4. **Fixed existing CVAE issues**: Resolved parameter mismatches and implementation bugs

## Files Added/Modified

### New Files:
- `src/NeuroCorrelation/Models/VAEGenerator.py` - Core VAE generation functionality
- `src/NeuroCorrelation/ModelPrediction/ModelPredictionEnhanced.py` - Enhanced prediction interface
- `complete_example_two_feature_vae.py` - Complete usage example
- `test_vae_generator.py` - Test suite for the new functionality

### Modified Files:
- `src/NeuroCorrelation/ModelTraining/ModelTraining.py` - Fixed CVAE parameter mismatch (line 890)
- `src/NeuroCorrelation/Models/ConditionalVariationalAutoEncoderModels.py` - Fixed duplicate code and forward method
- `test.py` - Added new experiment option for two-feature generation

## Usage Examples

### Basic Usage

```python
from src.NeuroCorrelation.ModelPrediction.ModelPredictionEnhanced import TwoFeatureVAEInterface

# Create interface with your trained VAE model
vae_interface = TwoFeatureVAEInterface(your_trained_vae_model, model_type="VAE")

# Generate data from two reference features
synthetic_data = vae_interface.generate_data(
    ref_feature1=0.5,     # First reference feature
    ref_feature2=1.2,     # Second reference feature  
    num_samples=10        # Number of samples to generate
)

print(f"Generated data shape: {synthetic_data.shape}")
```

### Callable Interface

```python
# The interface is callable for convenient usage
vae_interface = TwoFeatureVAEInterface(trained_model)
samples = vae_interface(0.5, 1.2, 10)  # (feature1, feature2, num_samples)
```

### Batch Generation

```python
# Generate for multiple feature pairs
feature_pairs = [
    (0.1, 0.3),  # First pair
    (0.5, 0.7),  # Second pair
    (0.9, 1.1),  # Third pair
]

batch_samples = vae_interface.generate_multiple(
    feature_references=feature_pairs,
    samples_per_reference=5  # 5 samples per pair
)
```

### CVAE Conditional Generation

```python
# For CVAE models with conditioning
cvae_interface = TwoFeatureVAEInterface(trained_cvae_model, model_type="CVAE")

conditioned_samples = cvae_interface.generate_data(
    ref_feature1=0.4,
    ref_feature2=0.6,
    num_samples=8
)
```

## Integration with Existing Code

### Using with ModelTraining

```python
# After training your VAE model
from src.NeuroCorrelation.ModelPrediction.ModelPredictionEnhanced import TwoFeatureVAEInterface

# Your existing training code...
model_training = ModelTraining(...)
trained_model = model_training.model

# Create generator interface
generator = TwoFeatureVAEInterface(trained_model, "VAE")

# Generate synthetic data from features
new_data = generator.generate_data(feature_a, feature_b, 100)
```

### Using with InstancesGeneration

```python
# In your InstancesGeneration workflow
vae_generator = TwoFeatureVAEInterface(self.model_trained["VAE"])

# Generate samples for analysis
synthetic_samples = vae_generator.generate_multiple(
    feature_references=[(f1, f2) for f1, f2 in feature_pairs],
    samples_per_reference=50
)
```

## Command Line Usage

Added new experiment option to `test.py`:

```bash
# Show information about two-feature generation
python test.py --exp generateTwoFeatures --num_case 1 --experiment_name_suffix test --main_folder output --repeation_b 1 --repeation_e 2 --optimization false --load_model false --train_models false

# Or use short form
python test.py --exp gtf --num_case 1 --experiment_name_suffix test --main_folder output --repeation_b 1 --repeation_e 2 --optimization false --load_model false --train_models false
```

## Testing

Run the complete test suite:

```bash
# Test the new functionality
python complete_example_two_feature_vae.py

# Run specific tests
python test_vae_generator.py
```

## Architecture Details

### VAEGenerator Class

The core `VAEGenerator` class provides:
- Model loading and initialization
- Multiple generation modes (latent sampling, direct conditioning)
- Batch processing capabilities
- Device management (CPU/GPU)

### TwoFeatureVAEInterface Class

Simplified interface specifically for the two-feature requirement:
- Clean API focused on the specific use case
- Callable interface for convenience
- Support for both VAE and CVAE models
- Batch processing for multiple feature pairs

### Generation Modes

1. **Latent Sampling**: Samples from the latent space and uses the decoder
2. **Direct Conditioning**: Uses features as direct conditioning input (CVAE only)

## Fixed Issues

1. **CVAE Parameter Mismatch**: Fixed `ModelTraining.py` line 890 where `x_in_timeweather` parameter was incorrectly passed instead of `condition`
2. **Duplicate Code**: Removed duplicate code in `ConditionalVariationalAutoEncoderModels.py`
3. **Forward Method**: Fixed nn_Model forward method to properly handle condition parameters

## Performance Considerations

- Models are set to evaluation mode for generation
- GPU support available (specify device="cuda")
- Batch processing for efficient multiple sample generation
- Memory-efficient tensor operations

## Future Enhancements

Potential improvements:
1. Support for more than two reference features
2. Advanced conditioning strategies
3. Model-specific optimizations
4. Caching for repeated generations
5. Integration with data visualization tools

## Dependencies

Required packages:
- torch
- numpy
- pathlib (standard library)
- typing (standard library)

## License

Same as the main project license.