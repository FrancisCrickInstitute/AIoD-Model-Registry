# AIoD Model Registry
This repository contains the central document that lists all AI models available within AIoD.

## Code Overview:
This repository contains the core description for all AI models available within AIoD. All submissions for AI models must go through the PR process and conform to the existing template design.

## Contact Details:
* cameron.shand@crick.ac.uk
* jon.smith@crick.ac.uk

## Development Setup:
This repository is used in conjunction with the AIoD front ends available here:
* https://github.com/FrancisCrickInstitute/ai-on-demand/

## Production Setup:
Install one of the AIoD front-ends or wrappers from:
* https://github.com/FrancisCrickInstitute/ai-on-demand/
 
## Usage:
Refer to your installed front-end or wrapper.

## Contribution Guidelines:
See the [contributing page on the Wiki](https://github.com/FrancisCrickInstitute/AIoD-Model-Registry/wiki/Model-Registry#contributing) for the steps necessary on how to contribute a model.

### Local Validation
To locally test whether a new manifest is eligible, simply run `pytest -v tests/`, where any errors will be detailed by Pydantic.
