# AIoD Model Registry
This repository contains the central manifests/schemas that define the models available within [AIoD](https://franciscrickinstitute.github.io/aiod_docs).

In addition, are:

- Tests for validating schemas
- Utility functions for ingesting the schemas and filtering by whether a user has access to each model, enabling us to [automatically write the UI for our Napari plugin](https://franciscrickinstitute.github.io/aiod_docs/sections/development/#automatic-ui-construction)

## Installation

Install the PyPI version with:
```
pip install aiod_registry
```

Note that you should do this from within a conda, uv etc. environment.


### Development
For development/contributing, you should clone and install the dev version:
```
git clone https://github.com/FrancisCrickInstitute/aiod_registry.git
cd aiod_registry
pip install -e ".[dev]"
```

## Contact Details
* cameron.shand@crick.ac.uk
* jon.smith@crick.ac.uk

## Contribution Guidelines
See our [documentation](https://franciscrickinstitute.github.io/aiod_docs/sections/contributing/expanding/) for guidance on adding models to this repo.

### Local Validation
To locally test whether a new manifest is eligible, simply run `pytest -v tests/`, where any errors will be detailed by Pydantic. Note that this will need the [development](#development) version. Otherwise, the tests will be run automatically on pull requests.
