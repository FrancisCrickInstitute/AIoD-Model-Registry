# AIoD Model Registry
This repository contains the central manifests/schemas that defines the models available within AIoD.

In addition, are:

- Tests for validating schemas
- Utility functions for ingesting the schemas and filtering by whether a user has access to each model, enabling us to [automatically write the UI for our Napari plugin](https://franciscrickinstitute.github.io/aiod_docs/sections/development/#automatic-ui-construction)

## Installation

```
git clone https://github.com/FrancisCrickInstitute/AIoD-Model-Registry.git
cd AIoD-Model-Registry/
pip install .
```

or

```
pip install git+https://github.com/FrancisCrickInstitute/AIoD-Model-Registry.git
```

## Contact Details
* cameron.shand@crick.ac.uk
* jon.smith@crick.ac.uk

## Contribution Guidelines
See our [documentation](https://franciscrickinstitute.github.io/aiod_docs/sections/contributing/expanding/) for guidance on adding models to this repo.

### Local Validation
To locally test whether a new manifest is eligible, simply run `pytest -v tests/`, where any errors will be detailed by Pydantic.
